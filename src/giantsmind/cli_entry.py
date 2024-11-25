#!/usr/bin/env python3
import argparse
import sys
from typing import List, Optional

from giantsmind.core.interact_papers import one_question_chain
from giantsmind.utils.logging import logger


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GiantsMind CLI")
    # parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the application.

    Args:
        args: List of command line arguments. Defaults to sys.argv[1:] if None.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parsed_args = parse_arguments(args)

    try:
        logger.info("Starting application...")

        # Your application logic goes here
        # logger.debug(f"Using configuration file: {parsed_args.config}")

        # Example: Load configuration
        # config = load_config(parsed_args.config)

        # Example: Initialize your application
        # app = Application(config)
        # app.run()

        one_question_chain(1)

        logger.info("Application completed successfully")
        return 0

    except Exception as e:
        logger.exception("An error occurred:")
        return 1


if __name__ == "__main__":
    # The following line allows the script to be both imported and run directly
    sys.exit(main())
