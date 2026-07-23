#!/usr/bin/env python3
"""Benchmark FLUX.2-klein-9B generation time on MLX."""

import argparse
import time
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.flux2 import Flux2Klein


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A red sports car, studio lighting, product photography")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, default=None, choices=[None, 4, 8])
    parser.add_argument("--output", default="benchmark_flux.png")
    parser.add_argument("--model", default="flux2-klein-9b", choices=["flux2-klein-4b", "flux2-klein-9b"])
    args = parser.parse_args()

    print(f"Loading {args.model} (quantize={args.quantize})...")
    t0 = time.time()

    if args.model == "flux2-klein-9b":
        model_config = ModelConfig.flux2_klein_9b()
    else:
        model_config = ModelConfig.flux2_klein_4b()

    model = Flux2Klein(model_config=model_config, quantize=args.quantize)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print(f"Generating {args.width}x{args.height} image...")
    t1 = time.time()
    result = model.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
    )
    gen_time = time.time() - t1
    print(f"Generation finished in {gen_time:.2f}s")

    img = result.image if hasattr(result, "image") else result
    img.save(args.output)
    print(f"Saved to {args.output}")

    print(f"\nSummary:")
    print(f"  Load time:      {load_time:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Total:           {load_time + gen_time:.2f}s")


if __name__ == "__main__":
    main()
