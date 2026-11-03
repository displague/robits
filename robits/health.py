"""Liveness check for a headless robits process that never calls the model.

The headless loop touches ROBITS_HEARTBEAT_FILE on every pass, and idle sleeps
are capped by ROBITS_IDLE_MAX_SLEEP_SECONDS (300 by default), so an old file
means the loop is stuck. Use as an exec probe:

    python -m robits.health --max-age 900

Exit 0 when the file was touched within max-age seconds, 1 otherwise. Pick
max-age above the idle cap plus the longest model call you expect.
"""

import argparse
import os
import sys
import time


def check(path, max_age, now=None):
    """Return (ok, message) for the heartbeat file at path."""
    if not path:
        return False, "ROBITS_HEARTBEAT_FILE is not set"
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(path)
    except OSError as error:
        return False, f"heartbeat unreadable: {error}"
    if age > max_age:
        return False, f"heartbeat is {age:.0f}s old (limit {max_age:.0f}s)"
    return True, f"heartbeat {age:.0f}s old"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robits.health", description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=os.environ.get("ROBITS_HEARTBEAT_FILE", ""),
                        help="heartbeat path (default: ROBITS_HEARTBEAT_FILE)")
    parser.add_argument("--max-age", type=float, default=900.0, help="seconds (default 900)")
    args = parser.parse_args(argv)
    ok, message = check(args.file, args.max_age)
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
