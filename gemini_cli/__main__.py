"""
Entry point for running Gemini CLI as a module
Allows: python -m gemini_cli
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())