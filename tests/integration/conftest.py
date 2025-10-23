import os
import sys

import pytest

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def setup_logging():
    """Fixture to setup logging for tests"""
    from giantsmind.utils.logging import logger

    logger.setLevel("ERROR")  # Reduce logging noise during tests
