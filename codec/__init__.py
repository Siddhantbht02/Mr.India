"""
Codec module containing Error-Correction Coding and Distribution-Preserving Embedders.
"""

from codec.distribution_preserving import DistributionPreservingCodec
from codec.ecc_engine import ECCEngine

__all__ = ["DistributionPreservingCodec", "ECCEngine"]
