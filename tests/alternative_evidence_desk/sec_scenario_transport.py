"""A controlled SEC transport for incremental-acquisition scenarios.

Serves an official-shaped submissions inventory per issuer (`filings.recent`
and, when declared, older `filings.files` shards), the registry, and filing
bodies by their official locator; counts every request by kind so a
scenario can prove what was and was not fetched; and takes declared
metadata transitions (a filing added, an amendment, a body that fails or
exceeds the cap) between passes. Synthetic bodies unless a scenario loads
retained originals into it; no network anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from alphalattice.evidence.alternative_evidence.sources.sec_edgar import SecOfficialResponse

NOW = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ScenarioFiling:
    accession: str
    form: str
    filed_on: str
    accepted_at: str
    primary_document: str
    report_date: str = ""
    size: int | None = None


class SecScenarioTransport:
    """Official-shaped responses from a declared inventory; every call counted."""

    def __init__(
        self,
        *,
        registry: dict[str, tuple[str, str]],
        filings: dict[str, list[ScenarioFiling]],
        bodies: dict[str, bytes] | None = None,
        shards: dict[str, dict[str, list[ScenarioFiling]]] | None = None,
        shard_ranges: dict[str, dict[str, tuple[str, str]]] | None = None,
    ) -> None:
        # registry: ticker -> (cik, title); filings: cik -> recent filings
        # (newest first as the SEC lists them); bodies: locator -> bytes;
        # shards: cik -> {shard file name -> filings}.
        self.registry = registry
        self.filings = filings
        self.bodies = dict(bodies or {})
        self.shards = shards or {}
        self.shard_ranges = shard_ranges or {}
        self.calls: list[str] = []
        self.failures: dict[str, Callable[[], Exception]] = {}
        self.retrieved_at = NOW
        self.before_get: Callable[[str], None] | None = None
        """A scenario's hook before each request -- to hold a transfer until
        another caller has acted, never to change what is served."""

    # ---------------------------------------------------------- accounting
    @property
    def registry_calls(self) -> int:
        return sum(1 for url in self.calls if url.endswith("company_tickers.json"))

    @property
    def inventory_calls(self) -> int:
        return sum(1 for url in self.calls if "/submissions/" in url)

    @property
    def body_calls(self) -> list[str]:
        return [url for url in self.calls if "/Archives/edgar/data/" in url]

    def body_calls_for(self, accession: str) -> int:
        compact = accession.replace("-", "")
        return sum(1 for url in self.body_calls if f"/{compact}/" in url)

    def reset_calls(self) -> None:
        self.calls = []

    # ---------------------------------------------------------- transitions
    def add_filing(self, cik: str, filing: ScenarioFiling, body: bytes) -> None:
        self.filings[cik] = [filing, *self.filings.get(cik, [])]
        self.bodies[self.locator(cik, filing)] = body

    @staticmethod
    def locator(cik: str, filing: ScenarioFiling) -> str:
        compact = filing.accession.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{compact}/{filing.primary_document}"

    def fail_body(self, cik: str, filing: ScenarioFiling, error: Callable[[], Exception]) -> None:
        self.failures[self.locator(cik, filing)] = error

    def heal_body(self, cik: str, filing: ScenarioFiling) -> None:
        self.failures.pop(self.locator(cik, filing), None)

    # ---------------------------------------------------------- the seam
    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        if self.before_get is not None:
            self.before_get(url)
        self.calls.append(url)
        failure = self.failures.get(url)
        if failure is not None:
            raise failure()
        if url.endswith("company_tickers.json"):
            payload = {
                str(index): {"cik_str": int(cik), "ticker": ticker, "title": title}
                for index, (ticker, (cik, title)) in enumerate(sorted(self.registry.items()))
            }
            return self._json(payload)
        if "/submissions/" in url:
            name = url.rsplit("/", 1)[-1]
            if name.startswith("CIK") and name.endswith(".json") and "-submissions-" not in name:
                cik = name[3:-5]
                return self._json(self._submissions(cik))
            # an older shard: CIK##########-submissions-001.json
            cik = name[3:13]
            shard = self.shards.get(cik, {}).get(name)
            if shard is None:
                raise ValueError(f"scenario shard not declared: {name}")
            return self._json({"filings": {"recent": self._columns(shard)}})
        if "/Archives/edgar/data/" in url:
            body = self.bodies.get(url)
            if body is None:
                raise ValueError(f"scenario body not declared: {url}")
            if len(body) > maximum_bytes:
                raise ValueError("alternative_evidence.sec_response_too_large")
            return SecOfficialResponse(
                status_code=200,
                content_type="text/html",
                content=body,
                retrieved_at=self.retrieved_at,
            )
        if "/companyfacts/" in url:
            return self._json({"facts": {"us-gaap": {}}})
        raise ValueError(f"scenario url not declared: {url}")

    def _submissions(self, cik: str) -> dict[str, object]:
        recent = self.filings.get(cik, [])
        files = [
            {
                "name": name,
                "filingCount": len(shard),
                "filingFrom": self.shard_ranges.get(cik, {}).get(
                    name, ("1900-01-01", "1900-01-01")
                )[0],
                "filingTo": self.shard_ranges.get(cik, {}).get(name, ("1900-01-01", "1900-01-01"))[
                    1
                ],
            }
            for name, shard in self.shards.get(cik, {}).items()
        ]
        return {"cik": cik, "filings": {"recent": self._columns(recent), "files": files}}

    @staticmethod
    def _columns(filings: list[ScenarioFiling]) -> dict[str, list[object]]:
        columns: dict[str, list[object]] = {
            "accessionNumber": [f.accession for f in filings],
            "filingDate": [f.filed_on for f in filings],
            "acceptanceDateTime": [f.accepted_at for f in filings],
            "form": [f.form for f in filings],
            "primaryDocument": [f.primary_document for f in filings],
            "reportDate": [f.report_date for f in filings],
        }
        if any(f.size is not None for f in filings):
            columns["size"] = [f.size if f.size is not None else 0 for f in filings]
        return columns

    def _json(self, payload: object) -> SecOfficialResponse:
        return SecOfficialResponse(
            status_code=200,
            content_type="application/json",
            content=json.dumps(payload).encode(),
            retrieved_at=self.retrieved_at,
        )


def filing_body(accession: str, *, words: int = 400) -> bytes:
    """A distinct synthetic filing body, long enough to canonicalize."""

    text = " ".join(
        f"Filing {accession} states item {index} of its ordinary narrative."
        for index in range(words // 10)
    )
    return f"<html><body><p>{text}</p></body></html>".encode()


__all__ = ["NOW", "ScenarioFiling", "SecScenarioTransport", "filing_body"]
