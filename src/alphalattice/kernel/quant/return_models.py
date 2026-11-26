"""Pure deterministic fit and predict engines for WP60D return models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from alphalattice.kernel.knowledge.catalog import SystemResourceCatalog
from alphalattice.kernel.knowledge.contracts import PackageArgSpecRef
from alphalattice.kernel.knowledge.enums import ResolutionStatus
from alphalattice.kernel.knowledge.errors import KnowledgeError
from alphalattice.kernel.quant.errors import QuantError
from alphalattice.kernel.quant.return_model_contracts import (
    ReturnModelPackageCall,
    ReturnModelParameter,
)

FloatArray = npt.NDArray[np.float64]
SEED = 1729


def _error(message: str, *, code: str, cause: BaseException | None = None) -> NoReturn:
    chain = () if cause is None else (f"{type(cause).__name__}: {cause}",)
    raise QuantError(message, code=code, cause_chain=chain) from cause


def audit_return_model_package_call(
    catalog: SystemResourceCatalog | None,
    references: tuple[PackageArgSpecRef, ...],
    stable_id: str,
    arguments: Mapping[str, JsonValue],
) -> ReturnModelPackageCall:
    """Validate a package invocation against its registered argument spec."""
    if catalog is None:
        _error(
            "return-model package execution requires a system resource catalog",
            code="quant.return_model_invalid_input",
        )
    reference = next((item for item in references if item.stable_id == stable_id), None)
    if reference is None:
        _error(
            f"return-model package authority is unavailable: {stable_id}",
            code="quant.return_model_invalid_input",
        )
    audit = catalog.validate_package_arguments(reference, arguments)
    if audit.status is not ResolutionStatus.RESOLVED:
        failure = audit.failure
        if failure is None:
            _error(
                f"return-model package audit omitted its failure: {stable_id}",
                code="quant.return_model_invalid_input",
            )
        raise KnowledgeError(
            f"return-model package arguments are unsupported: {stable_id}",
            code=failure.code,
            cause_chain=failure.cause_chain,
        )
    return ReturnModelPackageCall(
        stable_id=stable_id,
        arguments=tuple(
            ReturnModelParameter(name=name, value=value)
            for name, value in sorted(arguments.items())
        ),
        audit=audit,
    )


__all__ = ["audit_return_model_package_call"]
