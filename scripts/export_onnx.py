"""Export a Hive or Zertz .pt checkpoint to ONNX format.

Usage:
    uv run python scripts/export_onnx.py model.pt              # auto-detect game, output model.onnx
    uv run python scripts/export_onnx.py model.pt -o out.onnx  # explicit output path
"""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Export a .pt checkpoint to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", help="Output .onnx path (default: same name with .onnx)")
    args = parser.parse_args()

    onnx_path = args.output or args.checkpoint.rsplit(".", 1)[0] + ".onnx"

    # Peek at checkpoint to detect game type
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    game = checkpoint.get("game", "hive")

    if game == "zertz":
        from zertz.nn.model import load_checkpoint, export_onnx
    else:
        from hive.nn.model import load_checkpoint, export_onnx

    model, ckpt = load_checkpoint(args.checkpoint)
    model.eval()
    export_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
