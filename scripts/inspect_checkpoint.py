#!/usr/bin/env python3
"""Print properties of a .pt checkpoint file."""
import sys
import torch
import json


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    for key, value in checkpoint.items():
        if key == "model_state_dict":
            continue
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
