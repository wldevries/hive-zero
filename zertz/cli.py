"""CLI entry point for the Zertz AI engine."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Zertz AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Self-play training (stub)
    subparsers.add_parser("train", help="Run self-play training")

    args = parser.parse_args()

    if args.command == "train":
        raise NotImplementedError("Zertz training not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
