"""A Portfolio result's export manifest: what was asked, what answered, how to ask again.

Owner: the local application's Portfolio research service (`portfolio_research`), whose
`_manifest` builds it for an export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


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
        return json.dumps(
            {
                "kind": "PortfolioExportManifest",
                "workspace_id": self.workspace_id,
                "spec_hash": self.spec_hash,
                "holdings_spec_hash": self.holdings_spec_hash,
                "program_hash": self.program_hash,
                "numerical_input_assembly_hash": self.numerical_input_assembly_hash,
                "result_hash": self.result_hash,
                "execution_ledger_hash": self.execution_ledger_hash,
                "economic_ledger_hash": self.economic_ledger_hash,
                "report_hash": self.report_hash,
                "authorities_hash": self.authorities_hash,
                "originating_task_id": self.originating_task_id,
                "execution_mode": self.execution_mode,
                "controls": [list(pair) for pair in self.controls],
                "command": self.command,
                "spec": self.spec_document,
            },
            indent=2,
            sort_keys=True,
        )
