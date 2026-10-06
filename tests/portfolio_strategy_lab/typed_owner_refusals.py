"""Public constructor cases for the Web's typed owner refusal contract."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

CODE = "storage.managed_capacity_exceeded"
DETAIL = "owner-private-detail-7b2e"

# These classes are not public typed refusal cases; each exclusion has its source reason.
EXCLUDED_POSITIVE = {
    "alphalattice.control.observation_runtime.telemetry.process_metrics._ProcessExited": (
        "process_metrics.py:195-197; enumeration race is swallowed as an absent process, "
        "not a refusal"
    ),
    "alphalattice.control.product_host.composition.decision_advancement._Cancelled": (
        "decision_advancement.py:286; private cancellation control flow, with no code protocol"
    ),
    "alphalattice.control.product_host.composition.decision_advancement._Deferred": (
        "decision_advancement.py:290-293; internal WAIT carries StageExecutionResult, "
        "not a public failure"
    ),
    "alphalattice.kernel.knowledge.model_store._RetryableStatus": (
        "model_store.py:464-465,482; numeric HTTP status controls transfer retry, not an owner code"
    ),
    "alphalattice.kernel.shared_kernel.domain.errors.RegisteredCodeError": (
        "domain/errors.py:48-50,65-68; abstract category/label/codes require "
        "a concrete registered subclass"
    ),
    "alphalattice.control.task_control.registry.TaskNotFoundError": (
        "registry.py:99-105; typed UUID absence in KeyError.args, no owner-code field or text"
    ),
}


@dataclass(frozen=True)
class ExceptionSpec:
    module: str
    name: str
    path: Path
    line: int
    bases: tuple[str, ...]

    @property
    def fqn(self) -> str:
        return f"{self.module}.{self.name}"


def exception_specs(source_root: Path) -> tuple[ExceptionSpec, ...]:
    """Resolve local exception subclasses statically, including imported bases.

    source_root is the repository's src/alphalattice directory. No module is imported.
    """
    definitions: dict[str, ExceptionSpec] = {}
    builtin_exceptions = {
        name
        for name, value in vars(builtins).items()
        if isinstance(value, type) and issubclass(value, BaseException)
    }
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root.parent).with_suffix("")
        components = list(relative.parts)
        if components[-1] == "__init__":
            components.pop()
        module = ".".join(components)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        names: dict[str, str] = {}
        local_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                if node.level:
                    package = (
                        module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                    )
                    imported_module = ".".join(
                        package[: len(package) - node.level + 1]
                        + (imported_module.split(".") if imported_module else [])
                    )
                for alias in node.names:
                    names[alias.asname or alias.name] = f"{imported_module}.{alias.name}"
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for base in node.bases:
                raw = ast.unparse(base)
                first, dot, rest = raw.partition(".")
                if first in names:
                    bases.append(names[first] + (dot + rest if dot else ""))
                elif first in local_classes:
                    bases.append(f"{module}.{raw}")
                else:
                    bases.append(raw.removeprefix("builtins."))
            spec = ExceptionSpec(module, node.name, path, node.lineno, tuple(bases))
            definitions[spec.fqn] = spec
    selected: set[str] = set()
    while True:
        additions = {
            name
            for name, spec in definitions.items()
            if any(base in builtin_exceptions or base in selected for base in spec.bases)
        } - selected
        if not additions:
            break
        selected.update(additions)
    return tuple(definitions[name] for name in sorted(selected))


def load_exception(spec: ExceptionSpec) -> type[Exception]:
    """Load the discovered exception class for its public constructor."""
    cls = getattr(importlib.import_module(spec.module), spec.name)
    assert isinstance(cls, type) and issubclass(cls, Exception)
    return cls


def make_owner_exception(cls: type[Exception]) -> tuple[Exception, str]:
    """Construct one positive typed-code case using the actual public constructor.

    Required typed fixtures use public constructors, including the StorageInventoryError
    cause, DocumentRejectionCode enum and SealedBook record. The synthetic qualified
    code checks handling across exception bases; it makes no claim about every
    production call of the class.
    """
    fqn = f"{cls.__module__}.{cls.__name__}"
    if fqn in EXCLUDED_POSITIVE:
        raise ValueError(EXCLUDED_POSITIVE[fqn])
    name = cls.__name__
    module = importlib.import_module(cls.__module__)
    class_code = getattr(cls, "code", None)
    registered_codes = getattr(cls, "codes", None)
    if name == "LiveEvidenceError":
        registered_codes = module.LIVE_EVIDENCE_FAILURE_CODES
    code = (
        str(class_code)
        if isinstance(class_code, str)
        else (min(registered_codes) if registered_codes else CODE)
    )
    if name == "AlternativeEvidenceDocumentQualityError":
        rejection = module.DocumentRejectionCode.INVALID_MEDIA_TYPE
        return cls(rejection, DETAIL), rejection.value
    if name == "AlternativeEvidenceCommitRefused":
        from alphalattice.control.product_host.storage.inventory import StorageInventoryError

        return cls(StorageInventoryError(CODE, DETAIL)), CODE
    if name == "PortfolioReviewInputIncomplete":
        book = module.SealedBook(
            authority=module.BookAuthority.DEVELOPMENT_RESULT,
            report=None,
            result_hash=None,
            candidate_hash=None,
            handoff_hash=None,
        )
        return cls(book), "product_host.evidence_review_input_incomplete"
    if name == "UnparsableDocument":
        # The module already imports yaml; create its public unlocated exception.
        return cls(module.yaml.YAMLError()), "research_authoring.document_unparsable"
    if name == "ChildStartFailed":
        return cls(OSError(5, DETAIL)), "task_control.child_start_failed"
    if name == "SourceFileUnavailable":
        return cls("source_artifact_root", Path("."), OSError(2, DETAIL)), (
            "evidence_review.source_file_unavailable:source_artifact_root"
        )
    if name == "DeliveryBudgetBelowMinimumUnit":
        return cls(
            "packet", 2, 1
        ), "alternative_evidence.delivery_budget_below_minimum_unit:packet:2>1"
    if name == "UnitSourcesShort":
        return cls(held=0, issuers=1, needed=1, uncovered=()), (
            "alternative_evidence.minimum_entity_coverage_not_met:"
            "0 of 1 issuers hold a source, 1 needed"
        )
    if name == "TaskRecordAuthorityError":
        return cls(()), "task_control.database_authority_unreadable"
    if name == "TaskQueueHeadAuthorityError":
        return cls(), "task_control.database_authority_unreadable"
    if name == "AlphaTrainingInputAuthorityError":
        return cls(
            "FEATURE_AXIS_AUTHORITY_MISMATCH"
        ), "INVALID_EXPERIMENT:FEATURE_AXIS_AUTHORITY_MISMATCH"

    # Inherited builtin exception constructors accept arbitrary args, not keyword
    # fields. A class-owned code (SplitAdjustedPriceIntegrityError) must survive a
    # prose detail; otherwise use the synthetic code to prove base-independent handling.
    if cls.__init__ in (
        BaseException.__init__,
        Exception.__init__,
        ValueError.__init__,
        RuntimeError.__init__,
        KeyError.__init__,
        FileNotFoundError.__init__,
        TimeoutError.__init__,
        OSError.__init__,
    ):
        return cls(DETAIL if isinstance(class_code, str) else CODE), code
    signature = inspect.signature(cls.__init__)
    kwargs: dict[str, object] = {}
    values = {
        "retryable": False,
        "failure_code": code,
        "message": DETAIL
        if ("failure_code" in signature.parameters or "code" in signature.parameters)
        else code,
        "detail": DETAIL,
        "reason": code,
        "category": "storage",
        "threads": 1,
        "execution_attempt_count": 0,
        "prior_failure_codes": (),
        "domain_tool_call_sequence": (),
        "agent_executions": (),
        "model_context_audits": (),
        "evidence_as_of": datetime(2026, 1, 1, tzinfo=UTC),
        "entities": (),
        "failed": 0,
        "option": "source_artifact_root",
        "type_name": "RuntimeError",
        "failure_class": None,
    }
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.name == "code":
            if isinstance(parameter.default, str):
                code = parameter.default
            kwargs[parameter.name] = code
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif parameter.name in values:
            kwargs[parameter.name] = values[parameter.name]
        else:
            raise AssertionError(f"No factory parameter for {fqn}.{parameter.name}")
    return cls(**kwargs), code


@dataclass(frozen=True)
class OwnerRefusalCase:
    """One refusal raised through a handler's public owner method."""

    name: str
    factory: Callable[[], Exception]
    code: str
    private_markers: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def typed_owner_cases() -> tuple[OwnerRefusalCase, ...]:
    """Cover discovered owner exception classes and code-bearing raw builtins.

    The parser admits new exception subclasses automatically. Required constructor
    parameters without a known public fixture fail loudly, rather than silently
    removing a class from the Web contract matrix.
    """
    source_root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    cases = []
    for spec in exception_specs(source_root):
        if spec.fqn in EXCLUDED_POSITIVE:
            continue
        cls = load_exception(spec)
        exemplar, code = make_owner_exception(cls)

        def factory(exception_class: type[Exception] = cls) -> Exception:
            return make_owner_exception(exception_class)[0]

        cases.append(
            OwnerRefusalCase(
                name=spec.fqn,
                factory=factory,
                code=code,
                private_markers=(DETAIL,) if DETAIL in str(exemplar) else (),
            )
        )
    for cls in (ValueError, RuntimeError, FileNotFoundError):
        cases.append(
            OwnerRefusalCase(
                name=f"builtins.{cls.__name__}",
                factory=lambda exception_class=cls: exception_class(CODE),
                code=CODE,
            )
        )
    return tuple(cases)
