"""
Author: Constant ROUX
Team: SAMoVA
Date: 2022
"""


class Segment:
    """Represents a "segment" of audio frames by the start timecode, the end timecode, the duration / length of the
    segment and hys type (SILENCE or VOICE)."""

    def __init__(self, start, end, duration, type):
        self.start = start
        self.end = end
        self.duration = duration
        self.type = type

    def __str__(self):
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
