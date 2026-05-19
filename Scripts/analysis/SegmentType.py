"""
Author: Constant ROUX
Team: SAMoVA
Date: 2022
"""

# imports
from enum import Enum


class SegmentType(Enum):
    """Represents the segment type of the segment. It can be SILENCE (1) or VOICE (2)."""

    SILENCE = 1
    VOICE = 2
