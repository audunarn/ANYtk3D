"""Exact module alias for the historic private selection import path."""

import sys

from any3dview import selection as _shared_selection

# ANYfem's scale gates instrument private geometry predicates on this module.
# Returning the shared module object preserves that monkeypatch behaviour as
# well as public class identity.
sys.modules[__name__] = _shared_selection
