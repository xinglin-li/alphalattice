"""The development evidence set: twenty recorded issuer documents, ten in English and ten in
Chinese, over two issuers, with one document revised and one retired by its source.

Development and verification read these; the full corpus runs only in the nightly. Each case is
named: `REVISED` is one document's later revision, and `RETIRED` is one the source offered before
`RETIRED_AT` and no longer offers after it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import AcquiredEvidenceDocument
from alphalattice.evidence.alternative_evidence.sources.recorded import (
    RecordedEvidenceDocument,
    RecordedEvidenceSource,
)

NOW = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
ENTITIES = ("AAPL", "MSFT")
RETIRED_AT = NOW - timedelta(days=3)
"""After this the source no longer offers `RETIRED`; the documents it already gave stay."""

_EN = (
    "Revenue in the quarter rose {n} percent on services demand while hardware units fell.",
    "Management expects gross margin between {n} and {m} percent in the next quarter.",
    "The company repurchased {n} million shares and declared a dividend of {m} cents.",
    "A supplier disruption in region {n} delayed shipments by about {m} weeks.",
    "The board approved a credit facility of {n} hundred million dollars maturing in {m} years.",
)
# The Chinese comma is written by its code point, so no reader takes it for an ASCII comma.
_ZH = (
    "本季度营业收入同比增长{n}%\uff0c主要来自服务业务\uff0c硬件出货量有所下降。",
    "管理层预计下一季度毛利率在{n}%至{m}%之间。",
    "公司回购了{n}百万股股票\uff0c并宣布每股派发{m}分的股息。",
    "第{n}地区的供应中断使发货推迟了约{m}周。",
    "董事会批准了一项{n}亿美元的信贷额度\uff0c期限为{m}年。",
)


def _text(entity: str, index: int, chinese: bool) -> str:
    """A filing with a title, five sections and unique lines, so no quality rule reads it as
    boilerplate; Chinese sentences end in 。 and carry no spaces, as Chinese filings do."""
    sentences = _ZH if chinese else _EN
    title = f"# {entity} 第{index}号公告" if chinese else f"# {entity} report {index}"
    sections = [
        (f"## 第{part}节" if chinese else f"## Section {part}")
        + "\n\n"
        + sentence.format(n=index * 7 + part, m=index + part * 3)
        for part, sentence in enumerate(sentences, start=1)
    ]
    return "\n\n".join([title, *sections]) + "\n"


def _document(
    entity: str, index: int, *, chinese: bool, revision: str | None = None, text: str | None = None
) -> RecordedEvidenceDocument:
    at = NOW - timedelta(days=40 - index)
    return RecordedEvidenceDocument(
        entity_id=entity,
        source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
        evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
        document_type="8-K" if index % 2 else "10-Q",
        revision=revision or f"{entity.lower()}-{'zh' if chinese else 'en'}-{index:02d}",
        published_at=at - timedelta(days=1),
        captured_at=at,
        available_at=at,
        text=text or _text(entity, index, chinese),
        immutable_source=True,
    )


DOCUMENTS = tuple(
    _document(entity, index, chinese=chinese)
    for entity in ENTITIES
    for chinese in (False, True)
    for index in range(1, 6)
)
"""Twenty documents: each issuer five in English and five in Chinese."""

REVISED = _document(
    "AAPL",
    3,
    chinese=False,
    revision="aapl-en-03-revised",
    text=_text("AAPL", 3, False).replace("rose 22 percent", "rose 21 percent"),
).model_copy(
    update=dict(
        published_at=NOW - timedelta(days=11),
        captured_at=NOW - timedelta(days=10),
        available_at=NOW - timedelta(days=10),
    )
)
"""AAPL's third English document, revised: the same type, a new label and a corrected figure."""

RETIRED = DOCUMENTS[-1]
"""MSFT's fifth Chinese document, which the source stops offering at `RETIRED_AT`."""


def offered(at: datetime) -> tuple[RecordedEvidenceDocument, ...]:
    """What the issuer's source offers at `at`: the revision once it is available, and the
    retired document only before its retirement."""
    kept = tuple(item for item in DOCUMENTS if at < RETIRED_AT or item != RETIRED)
    return (*kept, REVISED) if at >= REVISED.available_at else kept


def acquired(at: datetime) -> tuple[AcquiredEvidenceDocument, ...]:
    """What the recorded source acquires from the offer at `at`, for a request read at `NOW`."""
    request = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=ENTITIES,
        evidence_as_of=NOW,
        acquisition_deadline=NOW + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.RECORDED,
    )
    return RecordedEvidenceSource().acquire_documents(request=request, documents=offered(at))
