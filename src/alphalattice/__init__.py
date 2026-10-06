"""AlphaLattice product package owned by this Playpen worktree.

This package deliberately does not extend its path. It once called
`pkgutil.extend_path`, which merged the outer AlphaLattice checkout into
`__path__`: a module missing here resolved there silently, the product loaded
source no reviewer of this worktree had seen, and an edit outside this worktree
could change what the product ran. Every module the product imports now lives
here, so a missing module is an import error rather than a silent fallback.
"""

from __future__ import annotations

__version__ = "0.0.0"
