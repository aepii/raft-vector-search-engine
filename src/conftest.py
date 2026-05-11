"""
pytest configuration file for the src/ directory.

Adds the src/ directory to sys.path so that test files within this directory
can import sibling modules using bare module names (e.g., `import coordinator`)
rather than requiring package-relative imports.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
