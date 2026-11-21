"""``python -m pymetrix.exercises``: the same as ``pymetrix-exercises``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
