#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Optional

from giantsmind.scripts.interact_papers import one_question_chain
from giantsmind.scripts.parse_papers import parse_papers
from giantsmind.utils.logging import logger


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GiantsMind CLI")
    parser.add_argument(
        "--parse",
        metavar="PDF_PATH",
        help="Parse PDFs from the specified folder path",
        nargs="?",
        const=os.getenv("DEFAULT_PDF_PATH"),
    )

    args = parser.parse_args(args)
    return args


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the application."""
    parsed_args = parse_arguments(args)

    try:
        if parsed_args.parse is not None:
            return parse_papers(parsed_args.parse)
        else:
            one_question_chain(1)
            return 0

    except Exception as e:
        logger.exception("An error occurred:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
