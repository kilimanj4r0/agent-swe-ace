"""Shared test configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--config",
        default=None,
        help="Path to override config YAML (relative to project root). "
        "If omitted, uses base config.yaml only.",
    )
