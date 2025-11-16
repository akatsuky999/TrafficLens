"""
Entry point for TrafficLens.
"""

from trafficlens.logging_config import setup_logging
from trafficlens.gui import run_app


def main() -> None:
    setup_logging()
    run_app()


if __name__ == "__main__":
    main()


