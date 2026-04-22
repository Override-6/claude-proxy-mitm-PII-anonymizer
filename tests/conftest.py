"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add /app to path so imports work
app_dir = Path("/app")
if app_dir.exists() and str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
