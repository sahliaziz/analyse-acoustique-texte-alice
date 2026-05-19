"""
Author: Constant ROUX
Based on: https://github.com/wiseman/py-webrtcvad example file
Date: 2022
"""



class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, data, timestamp, duration):
        self.data = data
        self.timestamp = timestamp
        self.duration = duration

    def __str__(self):
        return str(self.data) + ", " + str(self.timestamp)
