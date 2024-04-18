import importlib.metadata

try:
    __version__ = importlib.metadata.version("fmbench")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed, fallback to a default version
    __version__ = "None"
