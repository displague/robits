"""Stream utilities."""


class TeeStream:
    """Write-through multiplexer that mirrors output to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        """Write text to all streams and flush each immediately."""
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        """Flush all underlying streams."""
        for stream in self.streams:
            stream.flush()
