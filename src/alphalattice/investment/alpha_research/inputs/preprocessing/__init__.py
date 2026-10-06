"""Alpha-owned, training-only input transforms.

Fold-fitted preprocessing belongs to the Desk that consumes it, not to the Desk
that produces the state being transformed. These transformers read an
authoritative Sector surface; they never recompute Sector state.
"""
