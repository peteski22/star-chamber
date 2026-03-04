"""Star-chamber: Multi-LLM council protocol SDK."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("star-chamber")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
