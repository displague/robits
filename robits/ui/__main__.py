"""CLI entry point for running the Robits TUI."""
import argparse
import os
import sys
from pathlib import Path

from robits.core.config import _config
from robits.ui.app import RobitsTuiApp


def main():
    parser = argparse.ArgumentParser(
        description="Robits Terminal User Interface (TUI) for observability."
    )
    
    # Try to resolve default database path
    default_db = getattr(_config, "memory_db_path", None)
    if not default_db:
        default_db = Path.home() / ".local" / "share" / "robits" / "memory.db"
    
    parser.add_argument(
        "--db",
        type=str,
        default=str(default_db),
        help="Path to the SQLite database file."
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="UUID of the session to view/replay. Defaults to the most recent session."
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        choices=["full", "restricted", "public-only"],
        default="full",
        help="Observer policy for private thoughts and DMs (default: full)."
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (start a live session with chat and agent execution)."
    )

    args = parser.parse_args()

    # Expand user/relative paths for the DB
    db_path = os.path.abspath(os.path.expanduser(args.db))

    if not os.path.exists(db_path):
        if args.interactive:
            # Create directories so SQLite can initialize it
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        else:
            print(f"Error: Database file not found at '{db_path}'.", file=sys.stderr)
            print("Please run a simulation first to initialize the database or use --interactive.", file=sys.stderr)
            sys.exit(1)

    app = RobitsTuiApp(
        db_path=db_path,
        session_id=args.session_id,
        policy=args.policy,
        interactive=args.interactive
    )
    app.run()


if __name__ == "__main__":
    main()
