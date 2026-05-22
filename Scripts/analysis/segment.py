"""
Author: Constant ROUX
Team: SAMoVA
Date: 2022
"""
from enum import Enum


class SegmentType(Enum):
    """Represents the segment type of the segment. It can be SILENCE (1) or VOICE (2)."""
    SILENCE = 1
    VOICE = 2


class Segment:
    """Represents a "segment" of audio frames by the start timecode, the end timecode, the duration / length of the
    segment and hys type (SILENCE or VOICE)."""

    def __init__(self, start: float, end: float, duration: float, type: SegmentType) -> None:
        self.start = start
        self.end = end
        self.duration = duration
        self.type = type

    def __str__(self) -> str:
        return (
            str(self.type)
            + " : "
            + str(self.start)
            + " -> "
            + str(self.end)
            + " ["
            + str(self.duration)
            + "]"
        )
