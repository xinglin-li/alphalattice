"""What each committee member is given to judge a date's positions.

Its questions and the floor's facts, each sealed at the floor's open with the record it reads
and the reference it reads against, and a named absence where no record holds one (PENDING
for live data, UNAVAILABLE otherwise). The floor's admission of messages is `committee`'s;
this composes the material only, by reading the sealed records and taking differences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.bundles import moment_text as when

AGENDA_LIMIT: Final = 12
AGENDA_RANKING: Final = "Affected holding weight, then adverse count, then original alias."


def holdings(readback: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The date's held and exited names by weight (H1 the largest).

    Each carries its component scores at formation from the publication's sealed input.
    """
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioUpdatePublication as Publication,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        portfolio_update_positions,
    )

    value = Publication.model_validate(readback["publication"])
    history = tuple(Publication.model_validate(v) for v in readback.get("history", []))
    positions = portfolio_update_positions(value, history)
    previous = positions.preceding or positions.weights
    book = None if value.pending_proposal is None else value.pending_proposal.input
    components = getattr(book, "components", None) or (() if book is None else (book,))
    names = [f"C{n}" for n in range(1, len(components) + 1)]  # a component's floor alias
    scores: dict[str, dict[str, float]] = {}
    for name, component in zip(names, components, strict=False):
        for listing, score in _scores(component).items():
            scores.setdefault(listing, {})[str(name)] = score
    rows = [
        {
            "listing_id": listing,
            "name": label,
            "weight": weight,
            "change": weight - previous[i],
            "scores": scores.get(listing, {}),
        }
        for i, ((listing, label), weight) in enumerate(
            zip(readback["listing_labels"].items(), positions.weights, strict=True)
        )
        if weight > 0 or previous[i] > 0
    ]
    rows.sort(key=lambda r: (-r["weight"], -abs(r["change"]), r["name"]))
    return [{"alias": f"H{n}", **row} for n, row in enumerate(rows, 1)]


def agenda(
    points: Sequence[Mapping[str, Any]], held: Sequence[Mapping[str, Any]], limit: int
) -> dict[str, Any]:
    """A bounded, identity-bearing agenda; a small floor keeps its original record exactly."""
    if len(points) <= limit:
        return {}
    weights = {h["alias"]: h["weight"] for h in held}
    ranked = sorted(
        points,
        key=lambda p: (
            -sum(weights.get(a, 0) for a in p["targets"])
            if p["targets"]
            else -sum(weights.values()),
            -p.get("words", {}).get("n", 0),
            int(p["alias"][1:]),
        ),
    )
    policy = {
        "rule_version": 1,
        "limit": limit,
        "rank_keys": (
            ("affected_holding_weight", "desc"),
            ("adverse_count", "desc"),
            ("original_alias_number", "asc"),
        ),
    }
    return {
        "agenda": {
            **policy,
            "ranking": AGENDA_RANKING,
            "rule_hash": canonical_hash(policy),
            "aliases": [p["alias"] for p in ranked[:limit]],
            "below": [p["alias"] for p in ranked[limit:]],
        }
    }


QUESTIONS: Final = {
    "PM": {
        "PM.change": "What changed, and what did it cost?",
        "PM.envelope": "Is the book inside its research envelope?",
        "PM.feasibility": "Can it be carried out?",
        "PM.synthesis": "What does the committee hand on?",
    },
    "ALPHA": {
        "ALPHA.clean": "Is the signal clean?",
        "ALPHA.held": "Why is each name held?",
        "ALPHA.cost": "Was today's change worth its cost?",
        "ALPHA.bet": "Is it one bet?",
        "ALPHA.strength": "How strong is today's signal?",
    },
    "RISK": {
        "RISK.volatility": "How volatile is the book against its own history?",
        "RISK.decomposition": "Where does its risk come from?",
        "RISK.concentration": "Is it concentrated beyond its caps?",
        "RISK.correlation": "Do its names move together?",
        "RISK.liquidity": "Can it be traded at its size?",
        "RISK.reliability": "Can the risk read be trusted?",
    },
    "CRO": {
        "CRO.evidence": "What is wrong with the names held?",
        "CRO.coverage": "What did the review not see?",
        "CRO.process": "Did the update run by its declared method?",
        "CRO.rules": "Are the positions within the declared rules?",
        "CRO.dissents": "What stands against the book, and what goes to the person?",
    },
}
"""Each member's questions; the floor answers every one with a fact or a named absence."""

_ASKED: Final = {q: words for qs in QUESTIONS.values() for q, words in qs.items()}

_ABSENT: Final = (
    ("PM.envelope", "PENDING", "the replay baseline and live against expected need live sessions."),
    ("ALPHA.clean", "UNAVAILABLE", "data decisions on held names need the data platform's join."),
    ("ALPHA.bet", "UNAVAILABLE", "sector and style concentration are in no record yet."),
    (
        "ALPHA.strength",
        "UNAVAILABLE",
        "score dispersion needs raw pre-rank scores; hit rate pending.",
    ),
    (
        "RISK.volatility",
        "UNAVAILABLE",
        "volatility history needs the date's Risk kept each session.",
    ),
    ("RISK.correlation", "UNAVAILABLE", "pairwise correlation and clusters are in no record."),
    ("RISK.liquidity", "UNAVAILABLE", "position against ADV needs a declared capital."),
    ("CRO.process", "UNAVAILABLE", "repairs and data decisions need the data platform's join."),
)
"""What no record holds yet, said as a fact: PENDING for live data, UNAVAILABLE otherwise."""

_WORDS: Final = {
    "PENDING": "pending",
    "UNAVAILABLE": "not available",
    "EVALUATED": "evaluated",
    "NOT_EVALUATED": "not evaluated",
    "NOT_EVALUATED_FOR_THIS_PROPOSAL": "not evaluated for this proposal",
    "REVIEWED": "reviewed",
    "NOT_REVIEWED": "not reviewed",
    "NOT_REVIEWED_FOR_THIS_PROPOSAL": "not reviewed for this proposal",
    "NO_MATERIAL_OBJECTION": "no material objection",
    "ACCEPT_WITH_LIMITS": "accepted with limits",
    "REQUEST_EVIDENCE_REFRESH": "evidence refresh requested",
    "HUMAN_REVIEW_REQUIRED": "human review required",
    "MATERIAL_OBJECTION": "material objection",
}
"""The floor's words for each status its facts and views name. The code stays in the records'
fields: an absence's `state`, the review standing, the date's Risk."""


def status_word(code: Any) -> str:
    """A status in the floor's words; one no table names reads as its code."""
    return _WORDS.get(str(code), str(code))


_FACTS: Final = {
    "turnover": (
        "PM.change",
        "positions",
        "Turnover {t:.1%}, settled median {median}; cost {c5:.3%} at 5, {c10:.3%} at 10 bps.",
    ),
    "cash": (
        "PM.feasibility",
        "publication",
        "Cash {cash:.2%}; missed executions settled so far: {missed}.",
    ),
    "floor": (
        "PM.synthesis",
        "floor",
        "The floor itself: stances, rulings and items for the person.",
    ),
    "moves": (
        "ALPHA.clean",
        "publication",
        "{c} score moves on the previous session: {moved}; excluded from score comparisons: "
        "{unscored}.",
    ),
    "rule": (
        "ALPHA.held",
        "publication",
        "{c} holds {n}, top {top} by rank; past the top {top}, inside the exit line at {out}: "
        "{kept}; past the exit line at {out}, held by tranches formed on earlier sessions until "
        "each rebalances: {tranche}.",
    ),
    "spread": (
        "ALPHA.cost",
        "publication",
        "{c} entries and exits: score spread {spread:.2f} against a {cost:.3%} cost.",
    ),
    "curve": (
        "ALPHA.strength",
        "publication",
        "{c} matured return by rank: top fifth {top:.2%}, bottom {bottom:.2%}.",
    ),
    "shared": ("ALPHA.bet", "publication", "Held by two or more components: {shared}."),
    "risk": ("RISK.decomposition", "date_risk", "{view}"),
    "weights": (
        "RISK.concentration",
        "positions",
        "Largest {w:.2%} against the {cap} cap; top ten {ten:.2%}; {spread}",
    ),
    "covered": ("RISK.reliability", "date_risk", "Covered weight {covered:.0%}, Risk as of {at}."),
    "card": ("CRO.evidence", "evidence", "{handle} affects {held}; not held here: {outside}."),
    "flagged": (
        "CRO.evidence",
        "evidence",
        "The holdings the issue cards affect hold {flagged:.2%} of the book, across {cards} cards.",
    ),
    "scope": (
        "CRO.coverage",
        "evidence",
        "Of the book's ending weight: filings read {read}; checked with nothing filed in "
        "{days} days {nothing}; not reached {unread}. Read and checked together {together}; "
        "the review's in-scope weight {reviewed}, against the minimum {minimum}.",
    ),
    "scope_total": (
        "CRO.coverage",
        "evidence",
        "The review's in-scope weight {reviewed}, against the minimum {minimum}; the split by "
        "part is not recorded for this review.",
    ),
    "review": ("CRO.coverage", "review_standing", "{now}"),
    "package": (
        "CRO.process",
        "publication",
        "Package hash {same} the installed book's; observed through {seen}; enters {entry} "
        "from the {formation} close; the update advanced its data through the {target} session.",
    ),
    "caps": (
        "CRO.rules",
        "publication",
        "Largest {w:.2%} against the {cap:.0%} cap; research positions only.",
    ),
    "dissents": (
        "CRO.dissents",
        "floor",
        "The floor itself: standing dissents and items for the person.",
    ),
}
"""Each fact's question, source record and the Host's words; a writer gives its values."""


def _view(readback: Mapping[str, Any], held: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The book's sealed records as the facts read them, by reading and differencing only."""
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        PortfolioCostPolicy as Policy,
    )
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        PortfolioPerSideCostAssumption as Side,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioUpdatePublication as Publication,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
        FROZEN_RESEARCH_BOOK_RECIPES as RECIPES,
    )

    moved = np.asarray([0.5 * sum(abs(h["change"]) for h in held)])
    side = [Side.from_bps_per_side(b).platform_cost_bps for b in ("5", "10")]
    cost = [Policy().cost_fraction(one_way_turnover=moved, cost_bps=b)[0] for b in side]
    view: dict[str, Any] = {"components": [], "settled": [], "turnover": moved[0], "cost": cost}
    try:
        value = Publication.model_validate(readback["publication"])
        earlier = [Publication.model_validate(v) for v in readback.get("history", [])]
    except (KeyError, ValueError):
        return view
    events = {e.formation_session: e for v in (*earlier, value) for e in v.events}.values()
    proposal, package = value.pending_proposal, readback.get("strategy_package_id")
    before = _previous_proposal(proposal, earlier)
    return view | {
        "value": value,
        "proposal": proposal,
        "recipe": next((r for r in RECIPES if r.strategy_id == package), None),
        "components": _components(readback, proposal, before) if proposal else [],
        "settled": sorted(e.turnover for e in events if e.phase == "ENTRY_SETTLED"),
        "missed": sum(e.missed_executions for e in events),
    }


def _previous_proposal(proposal: Any, earlier: Sequence[Any]) -> Any:
    """The latest earlier proposal formed on a session before this one's: a history can hold this
    very publication, whose scores are today's, never the session before."""
    formed = getattr(getattr(proposal, "schedule", None), "formation_session", None)
    return next(
        (
            p.pending_proposal
            for p in reversed(earlier)
            if p.pending_proposal
            and formed is not None
            and p.pending_proposal.schedule.formation_session < formed
        ),
        None,
    )


def _components(readback: Mapping[str, Any], proposal: Any, before: Any) -> list[dict[str, Any]]:
    """Each component's held names, scores and ranks, and its scores the session before."""
    axis, sleeves = list(readback["listing_labels"]), proposal.book.sleeves
    counts, out, first = proposal.book.component_sleeve_counts or (len(sleeves),), [], 0
    for n, part in enumerate(_parts(proposal.input)):
        mine = {axis[i] for s in sleeves[first : first + counts[n]] for i, w in enumerate(s) if w}
        first += counts[n]
        old = _parts(before.input)[n : n + 1] if before else ()
        score = _scores(part)
        rank = {k: r for r, k in enumerate(sorted(score, key=lambda k: -score[k]), 1)}
        was = _scores(old[0]) if old else {}
        out.append({"alias": f"C{n + 1}", "mine": mine, "score": score, "rank": rank, "was": was})
        out[-1]["excluded"] = set(old[0].ordered_listing_ids) - was.keys() if old else set()
        out[-1]["curve"] = [v for v in part.bucket_means if v is not None]
    return out


def _scores(part: Any) -> dict[str, float]:
    """Only decision-eligible, present scores support comparisons, including signed scores."""
    return {
        k: score
        for k, score, eligible in zip(
            part.ordered_listing_ids, part.scores, part.decision_eligible, strict=True
        )
        if eligible and score is not None
    }


def _parts(book_input: Any) -> tuple[Any, ...]:
    """A book input's component inputs, one for a single-component book."""
    return tuple(getattr(book_input, "components", None) or (book_input,))


def _listed(items: Iterable[str]) -> str:
    return ", ".join(items) or "none"


def facts(
    readback: Mapping[str, Any], evidence: Mapping[str, Any], held: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The floor's facts: an S alias, the Host's words, its question and its source record.

    Each reads against a reference; a question no record answers gets a named absence.

    Args:
        readback: The update's readback, with its publication, history and date Risk.
        evidence: The Evidence projection on its positions.
        held: The floor's holdings.

    Returns:
        The facts in question order, each with its alias, question, words and source.
    """
    view = _book(readback, held)
    said = [*_read(_pm_values, readback, view), *_read(_alpha_values, held, view)]
    said += _read(_risk_values, readback, held, view)
    said += _read(_cro_values, readback, evidence, held, view)
    rows = [(*_FACTS[k][:2], _FACTS[k][2].format(**v), None) for k, v in said if v is not None]
    rows += [(q, "absence", f"{status_word(c).capitalize()}: {text}", c) for q, c, text in _ABSENT]
    clean = "no record that reads cleanly answers this"
    rows += [
        (q, "absence", f"{status_word('UNAVAILABLE').capitalize()}: {clean}: {w}", "UNAVAILABLE")
        for q, w in _ASKED.items()
        if all(r[0] != q for r in rows)
    ]
    rows.sort(key=lambda r: list(_ASKED).index(r[0]))
    return [
        {"alias": f"S{n}", "question": q, "text": t, "source": s} | ({"state": c} if c else {})
        for n, (q, s, t, c) in enumerate(rows, 1)
    ]


Said = list[tuple[str, dict[str, Any] | None]]


def _book(readback: Mapping[str, Any], held: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The book's view, or an empty one where its records cannot be read."""
    try:
        return _view(readback, held)
    except (ArithmeticError, LookupError, TypeError, ValueError, AttributeError):
        return {"components": [], "settled": [], "turnover": 0.0, "cost": [0.0, 0.0]}


def _read(writer: Any, *records: Any) -> Said:
    """A writer's values, or none where a record cannot be read: its questions are then named
    absences, and the floor opens all the same."""
    try:
        return list(writer(*records))
    except (ArithmeticError, LookupError, TypeError, ValueError, AttributeError):
        return []


def _fraction(value: Any) -> float:
    """A weight as the Evidence projection gives it: a fraction, or a percentage as text."""
    text = str(value or 0).strip()
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)


def _pm_values(readback: Mapping[str, Any], view: Mapping[str, Any]) -> Said:
    """What changed against its settled median, and the cash and misses."""
    past, proposal, (c5, c10) = view["settled"], view.get("proposal"), view["cost"]
    median = f"{past[len(past) // 2]:.1%} over {len(past)}" if past else "none settled"
    cash = dict(cash=proposal.book.cash, missed=view["missed"]) if proposal else None
    return [
        ("turnover", dict(t=view["turnover"], median=median, c5=c5, c10=c10)),
        ("cash", cash),
        ("floor", {}),
    ]


def _alpha_values(held: Sequence[Mapping[str, Any]], view: Mapping[str, Any]) -> Said:
    """Whether a clean, weak signal is carried out faithfully and cheaply, per component."""
    alias, recipe = {h["listing_id"]: h["alias"] for h in held}, view.get("recipe")
    top, out = (recipe.top_k, recipe.exit_rank) if recipe else (0, 0)
    moves = [h["listing_id"] for h in held if h["change"] and h["weight"] in (0, h["change"])]
    rules = (alias, top, out, moves, view["cost"][0])
    said = [s for c in view["components"] for s in _component_values(c, *rules)]
    names = [k for c in view["components"] for k in c["mine"]]
    shared = _listed(sorted({alias[k] for k in names if names.count(k) > 1}))
    return [*said, ("shared", dict(shared=shared) if names else None)]


def _component_values(
    c: Mapping[str, Any],
    alias: Mapping[str, str],
    top: int,
    out: int,
    moves: list[str],
    cost: float,
) -> Said:
    """One component's score moves, rule lines, entry-exit spread and rank curve."""
    score, was, rank, curve, name = c["score"], c["was"], c["rank"], c["curve"], c["alias"]
    # Every selection orders by its value, then by the listing: one source, one text.
    mine = sorted(c["mine"])
    both = [k for k in mine if None not in (was.get(k), score.get(k))]
    jumps = sorted(both, key=lambda k: (-abs(score[k] - was[k]), k))
    moved = _listed(f"{alias[k]} {was[k]:.2f}>{score[k]:.2f}" for k in jumps[:3])
    ties = sum(
        abs(score[k] - was[k]) == abs(score[j] - was[j]) for k in jumps[3:] for j in jumps[2:3]
    )
    moved += f", and {ties} others moved equally" if ties else ""
    unscored = _listed(
        alias.get(k, k)
        for k in sorted(set(mine) | set(moves))
        if k not in rank or k in c.get("excluded", ())
    )
    past = sorted((k for k in mine if top < rank.get(k, 0)), key=lambda k: (rank[k], k))
    kept = _listed(f"{alias.get(k, k)} {rank[k]}" for k in past if rank[k] <= out)
    tranche = _listed(f"{alias.get(k, k)} {rank[k]}" for k in past if rank[k] > out)
    gap = [score[k] for k in moves if k in score]
    ranks = dict(c=name, top=sum(curve[:4]) / 4, bottom=sum(curve[-4:]) / 4)
    return [
        ("moves", dict(c=name, moved=moved, unscored=unscored)),
        ("rule", dict(c=name, n=len(c["mine"]), top=top, out=out, kept=kept, tranche=tranche)),
        ("spread", dict(c=name, spread=max(gap) - min(gap), cost=cost) if gap else None),
        ("curve", ranks if len(curve) > 7 else None),
    ]


def _risk_values(
    readback: Mapping[str, Any], held: Sequence[Mapping[str, Any]], view: Mapping[str, Any]
) -> Said:
    """How the book can hurt today, each against its cap or its weight."""
    risk, recipe = readback.get("date_risk") or {}, view.get("recipe")
    w = sorted((h["weight"] for h in held), reverse=True) or [0.0]
    cap = f"{recipe.aggregate_name_cap:.0%} name" if recipe else "undeclared name"
    view_ = dict(view=risk_view(risk, held)) if risk.get("risk_status") == "EVALUATED" else None
    covered = risk.get("covered_weight")
    return [
        ("risk", view_),
        ("weights", dict(w=w[0], cap=cap, ten=sum(w[:10]), spread=concentration(held))),
        ("covered", None if covered is None else dict(covered=covered, at=risk.get("risk_as_of"))),
    ]


def _cro_values(
    readback: Mapping[str, Any],
    evidence: Mapping[str, Any],
    held: Sequence[Mapping[str, Any]],
    view: Mapping[str, Any],
) -> Said:
    """What is wrong or undisclosed, each against the book, the minimum or the declaration."""
    scope, split, recipe, proposal = (
        evidence.get("scope_coverage") or {},
        evidence.get("coverage") or {},
        view.get("recipe"),
        view.get("proposal"),
    )
    weight = {h["alias"]: h["weight"] for h in held}
    cards = [
        (c.get("issue_handle"), *card_holdings(c, evidence, held))
        for c in evidence.get("issue_cards") or ()
    ]
    flagged = sum(weight[a] for a in {a for _, hit, _ in cards for a in hit})
    sealed = (readback.get("realized_performance") or {}).get("strategy_package_hash")
    used = getattr(getattr(proposal, "input", None), "strategy_package_hash", sealed)
    same = "matches" if not sealed or used == sealed else "differs from"
    largest = max((h["weight"] for h in held), default=0.0)
    when = getattr(proposal, "schedule", None)
    seen = getattr(view.get("value"), "observed_through", None)
    target = (readback.get("update") or {}).get("target_session")
    dates = dict(entry=when.entry_session, formation=when.formation_session) if when else {}
    return [
        *(
            ("card", dict(handle=h, held=_listed(hit), outside=_listed(out)))
            for h, hit, out in cards
        ),
        ("flagged", dict(flagged=flagged, cards=len(cards))),
        *_coverage(scope, split),
        ("review", dict(now=review_now(readback))),
        ("package", dict(same=same, seen=seen, target=target, **dates) if when else None),
        ("caps", dict(w=largest, cap=recipe.aggregate_name_cap) if recipe else None),
        ("dissents", {}),
    ]


def _coverage(scope: Mapping[str, Any], split: Mapping[str, Any]) -> Said:
    """The reviewed weight against its minimum, by part where the review recorded its split, else
    that the split is not recorded (a review published before it): never a missing part printed."""
    total = (
        {
            "reviewed": scope.get("whole_book_reviewed_weight"),
            "minimum": scope.get("minimum_required_weight"),
        }
        if scope
        else None
    )
    parts = {k: split.get(f"{f}_ending_weight") for k, f in _SPLIT}
    parts["days"] = split.get("nothing_filed_window_days")
    whole = total is not None and None not in parts.values()
    if whole:  # the total is its named parts, in the parts' own form
        read, nothing = (float(str(parts[k]).rstrip("%")) for k in ("read", "nothing"))
        parts["together"] = f"{read + nothing:.3f}%"
    return [
        ("scope", {**total, **parts} if total and whole else None),
        ("scope_total", None if whole else total),
    ]


_SPLIT: Final = (("read", "reviewed"), ("nothing", "nothing_filed"), ("unread", "unreached"))
"""Each part of the book's ending weight a review's coverage records, by the fact's word."""


def _held(held: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The names held at the date's positions, weight above zero; an exit is never held."""
    return [h for h in held if h["weight"] > 0]


def card_holdings(
    card: Mapping[str, Any], evidence: Mapping[str, Any], held: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    """The held aliases an issue card affects, and the entities it names that no holding is.

    Each affected entity reaches its listings through its issuer row's tickers, which name the
    floor's holdings; nothing is matched by company name.

    Args:
        card: One issue card of the Evidence projection.
        evidence: The projection, with its issuer rows.
        held: The floor's holdings.

    Returns:
        The affected held aliases in the floor's order, and the entities held by none.
    """
    tickers = {
        r.get("entity_id"): r.get("tickers") or () for r in evidence.get("issuer_rows") or ()
    }
    aliases = {h["name"]: h["alias"] for h in _held(held)}
    hit = {
        e: [aliases[t] for t in tickers.get(e, ()) if t in aliases]
        for e in card.get("affected_entities") or ()
    }
    order = [h["alias"] for h in held]
    return sorted({a for a_ in hit.values() for a in a_}, key=order.index), [
        e for e, a in hit.items() if not a
    ]


def below_findings(
    points: Sequence[Mapping[str, Any]],
    below: Sequence[str],
    evidence: Mapping[str, Any],
    held: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Each HIGH issue card on a holding whose Evidence point sits below the agenda line.

    No stance judges such a card, so the report carries it whole, as an appendix the person reads.

    Args:
        points: The floor's tension points.
        below: The aliases of the points below the agenda line.
        evidence: The Evidence projection, with its issue cards and issuer rows.
        held: The floor's holdings.

    Returns:
        Each such card's handle, its below-line holdings and the CRO's inference.
    """
    adverse = {
        t
        for p in points
        if p["alias"] in below and p["kind"] == "HELD_WITH_ADVERSE_EVIDENCE"
        for t in p["targets"]
    }
    found = []
    for card in evidence.get("issue_cards") or ():
        hit = [h for h in card_holdings(card, evidence, held)[0] if h in adverse]
        if card.get("severity_if_true") == "HIGH" and hit:
            text = str(card.get("cro_inference") or "")
            found.append({"handle": card.get("issue_handle"), "holdings": hit, "text": text})
    return found


def issue_lines(evidence: Mapping[str, Any], held: Sequence[Mapping[str, Any]]) -> list[str]:
    """The CRO view's issue cards, each with the held aliases it affects.

    Args:
        evidence: The Evidence projection on the positions.
        held: The floor's holdings.

    Returns:
        One line per issue card.
    """
    return [
        f"{c.get('issue_handle')} (affects {_listed(card_holdings(c, evidence, held)[0])}): "
        f"{c.get('cro_inference')} Severity if true: {c.get('severity_if_true')}."
        for c in evidence.get("issue_cards") or ()
    ]


def _turnovers(view: Mapping[str, Any]) -> list[float]:
    """The settled sessions' one-way turnovers, smallest first."""
    return list(view.get("settled") or ())


def sufficiency(floor: Any) -> dict[str, list[str]]:
    """Each role's questions no fact on the floor answers; empty when every one has a fact."""
    asked = {f["question"] for f in floor.opened.get("facts", ())}
    return {
        r: missing for r, qs in QUESTIONS.items() if (missing := [q for q in qs if q not in asked])
    }


def _outliers(
    held: Sequence[Mapping[str, Any]], moved: list[str], view: Mapping[str, Any]
) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    """The date's exceptions as points, from the facts' own records."""
    past, top = view.get("settled") or [], getattr(view.get("recipe"), "top_k", 0)
    ranks = [(c["rank"], c["mine"]) for c in view.get("components") or ()]
    kept = [
        h["alias"]
        for h in held[:5]
        if top
        and any(
            h["listing_id"] in mine and rank.get(h["listing_id"], top + 1) > top
            for rank, mine in ranks
        )
    ]
    points = [
        (
            "TURNOVER_ABOVE_MEDIAN",
            "Turnover is above its settled median; the largest changes: {holdings}.",
            {"holdings": ", ".join(moved)},
            moved,
            bool(past) and view["turnover"] > past[len(past) // 2],
        ),
        (
            "KEPT_BY_RULE",
            "{holdings} sit among the five largest weights though past the top of that formation "
            "session's ranking by score: the exit line or an earlier tranche holds them.",
            {"holdings": ", ".join(kept)},
            kept,
            bool(kept),
        ),
    ]
    largest = (
        "LARGEST_POSITION",
        "The largest position, {holding}, and the five largest together.",
        {"holding": "H1"},
        [h["alias"] for h in held[:5]],
    )
    return [p[:4] for p in points if p[4]] or [largest]


def evidence_gap(evidence: Mapping[str, Any]) -> str | None:
    """Why no Evidence stands for the positions (a refusal, or a state with no analysis)."""
    if evidence.get("status") == "REFUSED" or evidence.get("failure_code"):
        return str(evidence.get("failure_code") or "REFUSED")
    if evidence.get("evidence_as_of") is None:
        return str(evidence.get("state") or "NOT_PREPARED")
    return None


def review_now(readback: Mapping[str, Any]) -> str:
    """The review of this proposal as it stands now; the sealed status only when none stands."""
    standing = readback.get("review_standing") or {}
    route, published = (standing.get("cro") or {}).get("route"), standing.get("published_at")
    if standing.get("status"):
        return (
            f"Review of this proposal now: {status_word(standing['status'])}"
            + (f", published {when(published)}" if published else "")
            + (f"; CRO route {status_word(route)}" if route else "")
            + "."
            + "".join(f" {said}" for said in standing.get("cutoff_words") or ())
        )
    sealed = status_word(readback["publication"].get("cro_status"))
    return f"Review of this proposal at publication: {sealed}."


def risk_view(risk: Mapping[str, Any] | None, held: Sequence[Mapping[str, Any]]) -> str:
    """The date's predicted risk as the Host renders it, report only; members type no digit."""
    if not risk or risk.get("risk_status") != "EVALUATED":
        reason = (risk or {}).get("reason") or "risk_research.date_risk_unavailable"
        said = status_word("NOT_EVALUATED")
        return f"Dated positions Risk assessment: {said}; unavailable ({reason}); "
    alias = {h["listing_id"]: h["alias"] for h in held}
    top = ", ".join(
        f"{alias.get(c['listing_id'], c['listing_id'])} {c['share']:.0%}"
        for c in risk["top_contributors"]
    )
    later = risk["sessions_before_the_positions"]
    return (
        f"Dated positions Risk assessment: {status_word('EVALUATED')}. "
        f"Predicted risk at {risk['risk_as_of']}"
        + (f", {later} sessions before them" if later else "")
        + f" (the installed recipe, report only): volatility {risk['volatility_per_session']:.2%} "
        f"a session, {risk['volatility_annualized']:.1%} a year; systematic share "
        f"{risk['systematic_share']:.0%}; covered weight {risk['covered_weight']:.0%}; largest "
        f"contributors {top}; "
    )


def concentration(held: Sequence[Mapping[str, Any]]) -> str:
    """Concentration from the date's weights alone."""
    weights = sorted((h["weight"] for h in held if h["weight"] > 0), reverse=True)
    if not weights:
        return "the date's positions hold no name."
    hhi = sum(w * w for w in weights)
    return (
        f"concentration from the date's weights only: {len(weights)} holdings, largest "
        f"{weights[0]:.2%}, five largest {sum(weights[:5]):.2%}, Herfindahl {hhi:.4f}, effective "
        f"names {1 / hhi:.1f}."
    )


def tension_points(
    held: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    risk: Mapping[str, Any] | None = None,
    readback: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """What the committee argues: this date's exceptions in its facts.

    Held names with Evidence flags; turnover above its settled median; large weights kept by
    the exit line rather than by score; the largest risk
    contributors (or why none stands); a gap in the Evidence. With no outlier, the largest
    position stands in for one.
    """
    adverse = {
        h["alias"]: int(row["adverse_issue_count"])
        for row in evidence.get("issuer_rows") or ()
        for h in _held(held)
        if h["name"] in (row.get("tickers") or ()) and int(row.get("adverse_issue_count") or 0)
    }
    moved = [h["alias"] for h in sorted(held, key=lambda h: -abs(h["change"]))[:5]]
    gap = evidence_gap(evidence)
    points = [
        *(
            (
                "HELD_WITH_ADVERSE_EVIDENCE",
                "{holding} is held while its Evidence names {n} major negative(s).",
                {"holding": a, "n": n},
                [a],
            )
            for a, n in adverse.items()
        ),
        *_outliers(held, moved, _book(readback, held) if readback else {}),
        *_risk_point(held, risk),
        *(
            [
                (
                    "EVIDENCE_GAP",
                    "No Evidence stands for these positions: {reason}.",
                    {"reason": gap},
                    [],
                )
            ]
            if gap
            else []
        ),
    ]
    return [
        {
            "alias": f"T{n}",
            "kind": kind,
            "text": word.format(**words),
            "word": word,
            "words": words,
            "targets": targets,
        }
        for n, (kind, word, words, targets) in enumerate(points, 1)
    ]


def _risk_point(
    held: Sequence[Mapping[str, Any]], risk: Mapping[str, Any] | None
) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    """The date's largest risk contributors as a point, or the reason no predicted risk stands."""
    if risk is None:
        return []
    alias = {h["listing_id"]: h["alias"] for h in held}
    top = [
        alias[c["listing_id"]]
        for c in risk.get("top_contributors") or ()
        if c["listing_id"] in alias
    ]
    if risk.get("risk_status") == "EVALUATED" and top:
        return [
            (
                "TOP_RISK_CONTRIBUTORS",
                "The largest risk contributors: {holdings}.",
                {"holdings": ", ".join(top)},
                top,
            )
        ]
    reason = risk.get("reason") or "none of the held names contributes"
    return [
        (
            "RISK_GAP",
            "No predicted risk stands for these positions: {reason}.",
            {"reason": reason},
            [],
        )
    ]
