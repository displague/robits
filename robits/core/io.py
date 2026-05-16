"""Stream utilities and structured JSONL logger."""
import json
import threading
from datetime import datetime, timezone


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


class JsonlLogger:
    """Write structured JSONL log entries to a file."""

    def __init__(self, f):
        self._f = f
        self._lock = threading.Lock()

    def write_event(self, event_type, **kwargs):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **kwargs,
        }
        with self._lock:
            print(json.dumps(record, default=str), file=self._f)
            self._f.flush()


class _NullLogger:
    def write_event(self, *_, **__): pass


_null = _NullLogger()
# Module-level singleton — safe for single-process CLI use. Concurrent main()
# invocations (e.g. parallel tests) would clobber each other; set_logger(None)
# in a finally block could silence a still-running caller. Tests that exercise
# logging should patch get_logger directly rather than calling set_logger.
_logger = None


def get_logger():
    return _logger if _logger is not None else _null


def set_logger(logger):
    global _logger
    _logger = logger
