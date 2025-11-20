"""Location loaders for the Universe modality.

This module automatically imports all location sources to register them.
"""

from .universe import UniverseLoader
from . import sources

__all__ = ['UniverseLoader', 'sources']
