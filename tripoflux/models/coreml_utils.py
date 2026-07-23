"""CoreML conversion and inference utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def convert_torch_to_coreml(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: Union[str, Path],
    compute_precision: str = "fp16",
    minimum_deployment_target: str = "macOS15",
) -> Optional[Path]:
    """Convert a PyTorch module to CoreML `.mlpackage`.

    Returns the output path on success, otherwise ``None``.
    """
    try:
        import coremltools as ct
    except ImportError:
        logger.warning("coremltools is not installed")
        return None

    output_path = Path(output_path)
    if output_path.exists():
        logger.info("CoreML model already exists at %s", output_path)
        return output_path

    model = model.eval()
    traced = torch.jit.trace(model, example_input)

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_input.shape)],
            compute_precision=(
                ct.precision.FLOAT16 if compute_precision == "fp16" else ct.precision.FLOAT32
            ),
            convert_to="mlprogram",
            minimum_deployment_target=getattr(ct.target, minimum_deployment_target, ct.target.macOS15),
        )
        mlmodel.save(str(output_path))
        logger.info("Saved CoreML model to %s", output_path)
        return output_path
    except Exception as exc:  # pragma: no cover - conversion failure
        logger.warning("CoreML conversion failed: %s", exc)
        return None


class CoreMLPredictor:
    """Lightweight wrapper around a CoreML `.mlpackage`."""

    def __init__(self, model_path: Union[str, Path]):
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError("coremltools is not installed") from exc

        self.model_path = Path(model_path)
        self._model = ct.models.MLModel(str(self.model_path))
        self._spec = self._model.get_spec()
        self._input_name = self._spec.description.input[0].name
        self._output_name = self._spec.description.output[0].name

    def predict(self, input_array) -> any:
        import numpy as np

        if isinstance(input_array, torch.Tensor):
            input_array = input_array.detach().cpu().numpy()
        input_array = np.ascontiguousarray(input_array)
        out = self._model.predict({self._input_name: input_array})
        return out[self._output_name]


def load_coreml_model(path: Union[str, Path]) -> Optional[CoreMLPredictor]:
    try:
        return CoreMLPredictor(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load CoreML model %s: %s", path, exc)
        return None
