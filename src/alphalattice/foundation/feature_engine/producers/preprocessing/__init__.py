"""Panel preprocessing: the sole owner of cross-sectional transformation.

Separated from series production so one responsibility has one active owner:
``feature_kernels`` computes single-series formulas, and everything that
reshapes a cross-section -- winsorization, sector neutralization,
standardization -- lives here behind an explicitly installed recipe.
"""

from __future__ import annotations
