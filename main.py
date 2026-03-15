"""Entry point for the Hive UHP engine."""

from hive.uhp.engine import UHPEngine


def main():
    engine = UHPEngine()
    engine.run()


if __name__ == "__main__":
    main()
