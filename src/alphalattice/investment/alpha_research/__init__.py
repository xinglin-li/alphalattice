"""Responsibility-partitioned Alpha Research package.

External research code imports the deterministic ``inputs``, ``targets``,
``experiments``, and ``evaluation`` owners directly. Agent orchestration,
publication, scores, verification, and narrow legacy readback remain explicit
sibling boundaries; this initializer deliberately has no eager imports.
"""
