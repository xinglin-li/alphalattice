"""Identity of one Risk development Program.

A development Program is experiment authority, not a surface: it says which
installed implementation, under which declared parameter domain, over which
frozen source closure, a development run is allowed to execute. It is deliberately
separate from the published Risk identities, and nothing here is ever written to
a current or admitted pointer.

Why the implementation identity is carried here rather than referenced: a catalog
that hashes only method identifiers lets the same identifier change code while
old evidence still looks reusable. ``ordered_numerical_binding_hashes`` is the
catalog's per-capability implementation identity, and because
``development_binding_hash`` covers it, a changed implementation under an
unchanged identifier produces a different Program and therefore cannot reuse the
earlier evidence.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH_PATTERN = r"^[0-9a-f]{64}$"

RISK_EXPERIMENT_KIND = "risk.covariance-development"
"""The authored kind this Desk answers for.

Declared here rather than in the compiler because the *verifier* needs it too,
and importing the compiler to reach a string would have pulled the estimator
catalog and the covariance adapter into the verifier's import closure. Replay
resolves a verifier and must not be able to reach anything that computes; a
constant is not a reason to weaken that.
"""


class RiskDevelopmentProgramBinding(BaseModel):  # type: ignore[misc]
    """Everything a development run must agree on before any numerical call.

    Two identities live here and they answer different questions. Conflating
    them was the defect:

    ``selected_method_binding_hash``
        *What is about to compute.* The chosen capability, its sealed recipe,
        its declared parameter domain, the content of its executable code, and
        the numerical environment. Installing an unrelated adapter does not
        move it, because an adapter nobody selected does not change what runs.

    ``catalog_hash``
        *What the Host has installed.* Governance. It moves whenever the
        installed set changes, which is correct and is why it cannot be the
        numerical identity: folding the whole catalog into the selected method
        made "the Host installed something else" indistinguishable from "the
        numbers would come out differently".

    ``development_binding_hash`` still covers both, so a governance change does
    move the Program. That is intentional and separately recorded: it is a
    reseal question, not evidence that the methodology changed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["RiskDevelopmentProgramBinding"] = "RiskDevelopmentProgramBinding"
    catalog_hash: str = Field(pattern=_HASH_PATTERN)
    selected_adapter_id: str = Field(min_length=1, max_length=96)
    selected_numerical_binding_hash: str = Field(pattern=_HASH_PATTERN)
    recipe_hash: str = Field(pattern=_HASH_PATTERN)
    parameter_domain_hash: str = Field(pattern=_HASH_PATTERN)
    source_closure_hash: str | None = Field(
        default=None, pattern=_HASH_PATTERN, exclude_if=lambda v: v is None
    )
    development_source_closure_hash: str | None = Field(
        default=None, pattern=_HASH_PATTERN, exclude_if=lambda v: v is None
    )
    """The code closures a binding sealed before the binding plan's P.

    A Program binds meaning only: the code that runs a study is bound by its plan's
    implementation hash, whose moves are recorded, so an edit that leaves the numbers
    alone moves no Program. A binding sealed earlier still carries both closures and
    parses with them; its Program reads back as recorded and is historical
    (``earlier_scheme``). New bindings carry neither.
    """

    numerical_environment_hash: str | None = Field(
        default=None, pattern=_HASH_PATTERN, exclude_if=lambda v: v is None
    )
    """Only on a binding sealed before E0: the environment is provenance (LAWS.md ID6)."""
    selected_method_binding_hash: str = Field(pattern=_HASH_PATTERN)
    development_binding_hash: str = Field(pattern=_HASH_PATTERN)

    @property
    def earlier_scheme(self) -> bool:
        """Sealed before P: the binding carries the code closures."""

        return self.source_closure_hash is not None

    @staticmethod
    def selected_method_identity(
        *,
        selected_adapter_id: str,
        selected_numerical_binding_hash: str,
        recipe_hash: str,
        parameter_domain_hash: str,
        numerical_environment_hash: str | None = None,
        source_closure_hash: str | None = None,
        development_source_closure_hash: str | None = None,
    ) -> str:
        """Identity of the computation itself, with the catalog left out."""

        identity: dict[str, str | None] = {
            "kind": "RiskSelectedMethodBinding",
            "selected_adapter_id": selected_adapter_id,
            "selected_numerical_binding_hash": selected_numerical_binding_hash,
            "recipe_hash": recipe_hash,
            "parameter_domain_hash": parameter_domain_hash,
        }
        if numerical_environment_hash is not None:
            identity["numerical_environment_hash"] = numerical_environment_hash
        if source_closure_hash is not None or development_source_closure_hash is not None:
            identity |= {
                "source_closure_hash": source_closure_hash,
                "development_source_closure_hash": development_source_closure_hash,
            }
        return str(canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> Self:
        if self.selected_method_binding_hash != self.selected_method_identity(
            selected_adapter_id=self.selected_adapter_id,
            selected_numerical_binding_hash=self.selected_numerical_binding_hash,
            recipe_hash=self.recipe_hash,
            parameter_domain_hash=self.parameter_domain_hash,
            source_closure_hash=self.source_closure_hash,
            development_source_closure_hash=self.development_source_closure_hash,
            numerical_environment_hash=self.numerical_environment_hash,
        ):
            raise ValueError("risk_research.selected_method_binding_identity_invalid")
        expected = canonical_hash(
            self.model_dump(mode="json", exclude={"development_binding_hash"})
        )
        if self.development_binding_hash != expected:
            raise ValueError("risk_research.development_binding_identity_invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        catalog_hash: str,
        selected_adapter_id: str,
        selected_numerical_binding_hash: str,
        recipe_hash: str,
        parameter_domain_hash: str,
    ) -> RiskDevelopmentProgramBinding:
        selected = cls.selected_method_identity(
            selected_adapter_id=selected_adapter_id,
            selected_numerical_binding_hash=selected_numerical_binding_hash,
            recipe_hash=recipe_hash,
            parameter_domain_hash=parameter_domain_hash,
        )
        fields = {
            "kind": "RiskDevelopmentProgramBinding",
            "catalog_hash": catalog_hash,
            "selected_adapter_id": selected_adapter_id,
            "selected_numerical_binding_hash": selected_numerical_binding_hash,
            "recipe_hash": recipe_hash,
            "parameter_domain_hash": parameter_domain_hash,
            "selected_method_binding_hash": selected,
        }
        return cls(**fields, development_binding_hash=str(canonical_hash(fields)))


__all__ = ["RiskDevelopmentProgramBinding"]
