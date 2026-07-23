"""BiRefNet background removal via CoreML.

Provides a CoreML inference path for the BiRefNet model shipped with
TripoSplat. If a converted `.mlpackage` is not present, it can be created
from the original PyTorch weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..vendor.triposplat.model import BiRefNet
from .coreml_utils import CoreMLPredictor, convert_torch_to_coreml, load_coreml_model

logger = logging.getLogger(__name__)


class BiRefNetCoreML:
    """CoreML backend for BiRefNet background removal."""

    INPUT_SIZE = (1024, 1024)
    _NORM_MEAN = (0.485, 0.456, 0.406)
    _NORM_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        coreml_path: Union[str, Path],
        fallback_torch_path: Optional[Union[str, Path]] = None,
        device: str = "mps",
    ):
        self.coreml_path = Path(coreml_path)
        self.device = torch.device(device)
        self._predictor: Optional[CoreMLPredictor] = load_coreml_model(self.coreml_path)
        self._torch_model: Optional[BiRefNet] = None
        if self._predictor is None and fallback_torch_path is not None:
            self._load_torch(fallback_torch_path)

    def _load_torch(self, path: Union[str, Path]):
        model = BiRefNet()
        model.load_safetensors(str(path))
        model.to(self.device).eval()
        self._torch_model = model

    def convert_from_torch(
        self,
        torch_ckpt_path: Union[str, Path],
        force: bool = False,
    ) -> Optional[Path]:
        """Convert the original PyTorch checkpoint to CoreML."""
        if self.coreml_path.exists() and not force:
            logger.info("CoreML model already exists: %s", self.coreml_path)
            return self.coreml_path

        model = BiRefNet()
        model.load_safetensors(str(torch_ckpt_path))
        model.to(self.device).eval()

        example = torch.randn(1, 3, *self.INPUT_SIZE, device=self.device)
        out = convert_torch_to_coreml(
            model,
            example,
            self.coreml_path,
            compute_precision="fp16",
        )
        if out is not None:
            self._predictor = load_coreml_model(out)
        return out

    @property
    def is_coreml_available(self) -> bool:
        return self._predictor is not None

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        W, H = image.size
        arr = np.array(image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=self.INPUT_SIZE, mode="bilinear", align_corners=True)
        mean = torch.tensor(self._NORM_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._NORM_STD).view(1, 3, 1, 1)
        return ((t - mean) / std).to(self.device)

    def _postprocess(self, alpha: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
        H, W = original_size[1], original_size[0]
        alpha_t = torch.from_numpy(alpha).float().unsqueeze(0).unsqueeze(0)
        alpha_t = F.interpolate(alpha_t, size=(H, W), mode="bilinear", align_corners=True)
        a = (alpha_t.clamp(0, 1) * 255).to(torch.uint8)[0, 0].numpy()
        return Image.fromarray(a, mode="L")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Return an RGBA image with the predicted alpha matte."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        t = self._preprocess(image)
        original_size = image.size

        if self._predictor is not None:
            alpha = self._predictor.predict(t.cpu())
            alpha = np.asarray(alpha).squeeze()
        elif self._torch_model is not None:
            with torch.no_grad():
                alpha = self._torch_model(t)
            alpha = alpha.float().cpu().numpy().squeeze()
        else:
            raise RuntimeError("Neither CoreML nor PyTorch BiRefNet is available")

        alpha_img = self._postprocess(alpha, original_size)
        rgba = image.copy()
        rgba.putalpha(alpha_img)
        return rgba


def create_birefnet_coreml(
    triposplat_dir: Union[str, Path],
    device: str = "mps",
) -> BiRefNetCoreML:
    triposplat_dir = Path(triposplat_dir)
    coreml_path = triposplat_dir / "birefnet.mlpackage"
    torch_path = triposplat_dir / "background_removal" / "birefnet.safetensors"
    model = BiRefNetCoreML(coreml_path=coreml_path, fallback_torch_path=torch_path, device=device)
    if not model.is_coreml_available:
        model.convert_from_torch(torch_path)
    return model
