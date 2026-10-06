"""Cross-fitted stock-return calibration capabilities."""

from .authority import (
    AlphaReturnUnitCapabilityBinding,
    ResolvedAlphaReturnUnitCalibration,
    installed_alpha_return_unit_capability,
    readback_alpha_return_unit_calibration,
    resolve_alpha_return_unit_calibration,
    seal_installed_return_unit_calibration,
)
from .return_unit import (
    AlphaReturnUnitCalibration,
    AlphaReturnUnitCalibrationError,
    AlphaReturnUnitCalibrationEvidence,
    applied_expected_return_matrix,
    calibrate_alpha_return_unit_signal,
)
from .stock_returns import (
    StockCalibrationEvidence,
    StockCalibrationRecipe,
    anchored_expanding_calibration_folds,
    calibrate_stock_returns_cross_fitted,
)

__all__ = [
    "AlphaReturnUnitCalibration",
    "AlphaReturnUnitCalibrationError",
    "AlphaReturnUnitCalibrationEvidence",
    "AlphaReturnUnitCapabilityBinding",
    "ResolvedAlphaReturnUnitCalibration",
    "StockCalibrationEvidence",
    "StockCalibrationRecipe",
    "anchored_expanding_calibration_folds",
    "applied_expected_return_matrix",
    "calibrate_alpha_return_unit_signal",
    "calibrate_stock_returns_cross_fitted",
    "installed_alpha_return_unit_capability",
    "readback_alpha_return_unit_calibration",
    "resolve_alpha_return_unit_calibration",
    "seal_installed_return_unit_calibration",
]
