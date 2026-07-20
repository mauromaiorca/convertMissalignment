"""Allow ``python -m convertMissalignment``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
