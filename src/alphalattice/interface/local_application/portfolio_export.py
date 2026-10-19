"""A Portfolio result's export manifest: what was asked, what answered, how to ask again.

Owner: the local application's Portfolio research service (`portfolio_research`), whose
`_manifest` builds it for an export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class PortfolioExportManifest:
    """One compact record of what was asked, what answered, and how to ask again.

    The command needs the workspace's folder and the spec, saved from ``spec_document``, and
    nothing else. The predecessor's root/hash argument list is deliberately not restored: a
    hash on a command line is an identity being replayed as an authority, and the compiler
    never saw it.
    """

    workspace_id: str
    spec_hash: str
    holdings_spec_hash: str
    program_hash: str
    numerical_input_assembly_hash: str | None
    result_hash: str
    execution_ledger_hash: str
    economic_ledger_hash: str
    report_hash: str
    authorities_hash: str | None
    originating_task_id: str | None
    execution_mode: str
    controls: tuple[tuple[str, str], ...]
    command: str
    spec_document: dict[str, object]

    def as_json(self) -> str:
        """Serialize the manifest with stable keys and readable indentation."""
        body = {field.name: getattr(self, field.name) for field in fields(self)}
        body["kind"] = "PortfolioExportManifest"
        body["spec"] = body.pop("spec_document")
        return json.dumps(body, indent=2, sort_keys=True)
