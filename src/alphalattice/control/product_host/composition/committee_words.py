"""The committee floor's words: its refusals, rules, routes, report phrases and road ends.

Owned by the committee floor (`committee.py`), which reads and re-exports them, so readers
of `committee.MEMBER_COMMANDS` and the like are unchanged; split from it for size, so the
floor's logic and the sentences agents learn from are kept apart.
"""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.project_layout import command_prefix

COMMAND: Final = " ".join(command_prefix())
RULE_WORDS: Final = {
    "ROLE": "Only its senders send this kind, each with the key in its own bundle: a STANCE or "
    "REPLY any member, a CHALLENGE or PASS a specialist, a RULING, VERDICT or PERSON_ANSWER the "
    "PM. This sender may send now: {subject}",
    "STANCES": "The floor is in its blind first round, which takes only one STANCE from each "
    f"member; follow `{COMMAND} committee wait` until the stances are revealed.",
    "DEBATE": "The floor is past its stances; it takes challenges, replies, passes, rulings and "
    "the verdict, not a STANCE.",
    "CLOSED": "The floor is closed, by its verdict or its time box, and takes only the PM's "
    "PERSON_ANSWER to an item handed to the person. A specialist sends nothing more and returns "
    "its receipt to its lead; the PM sends the offered `report` request as it stands.",
    "STANCE_IN": "This member's stance is on the floor already, one each; speak again by a "
    "CHALLENGE or REPLY once the stances are revealed.",
    "CAP": "This member has sent its three challenges or replies after its stance; the floor "
    "takes no more from it.",
    "TURN": "A PASS yields this member's one discussion turn: it comes from a member whose stance "
    "is on the floor, before it challenged or replied, and names no target or position; after it "
    "the member sends nothing more.",
    "TARGET": "A target, position or alias names nothing on the floor; name only the points, "
    f"holdings, messages and facts `aliases` lists (`{COMMAND} committee show` shows them). A "
    "member's role is never a target: answer its message by M alias. A point typed by its kind, "
    "a holding by its name or a role named as a target is offered corrected.",
    "REPLY_TO": "The message named in reply_to cannot be answered by this kind: a REPLY answers "
    "any message, a RULING a challenge not yet ruled, a PERSON_ANSWER an item handed to the "
    "person. Send the corrected message the answer offers.",
    "SHAPE": "This kind's shape does not fit: reply_to is only for a REPLY, RULING or "
    "PERSON_ANSWER, naming the message it answers, so drop it from a STANCE, CHALLENGE or PASS; "
    "an outcome is only for a RULING or the VERDICT, which each require one. Send the corrected "
    "message the answer offers.",
    "UNRULED": "A verdict closes the floor, so the PM rules on every challenge first and one "
    "stands unruled; read it, send its ruling in your own words, then the verdict.",
    "MEMBER": "This bundle's member submits from its own session with its key: run the command "
    "under Your route in its bundle's README.md as it stands, which carries `--key` and the key, "
    "and take the Host's corrections yourself. A lead starts the member with its bundle, then "
    "runs this again for its receipt.",
    "KEY": "This key is not this bundle's: a member's key admits only its own bundle's answer, "
    "with the command its route names.",
    "ENDED": "This bundle's answer is accepted and its key has ended; only that answer reads its "
    "receipt again. Return the receipt it gave to your lead.",
    "FACTS": "A RESERVE rests on the floor's facts: cite by S alias the facts, or the named "
    f"absences, that `{COMMAND} committee show` lists under this member's questions; resubmit.",
    "NUMBER": "A digit is typed outside an alias ({subject}), and the Host renders every number: "
    "every figure is written as its floor alias, which `aliases` lists: resubmit the same message "
    "with each typed figure or reference replaced by its alias on the floor.",
    "LATE": "This member's window has passed: a stance after the blind round's window, or a "
    "turn after the discussion window, is not recorded, and a silent turn counts as passed. "
    "Nothing more is taken from this member; return to your lead what you would have said, so "
    "the lead can carry it to the person.",
    "AGENDA": "This message's positions do not fit the agenda of {subject} points: key positions "
    "by agenda point or holding aliases, never by fact aliases, which are cited in "
    "the text. The PM may raise one below-line point through a REPLY's positions; other messages "
    "name only agenda or raised points. Send the offered correction, or revise the named text "
    "field.",
}
"""Each floor rule's own words, as its refusal says them."""

VERDICT_RULE: Final = (
    "not an approval or advice. PROCEED_WITH_NOTES by default, members' reservations carried as "
    "notes; a problem with particular holdings is a note and a suggestion to avoid or reduce them, "
    "never a DECLINE. DECLINE endorsement only for a book-level hard defect (a data error "
    "affecting the book, a breached book Risk bound, or key evidence missing for the book as a "
    "whole), "
    "naming it; FOR_THE_PERSON only for a named decision of the person; PROCEED when nothing "
    "needs noting."
)
"""How the PM chooses its verdict: the committee is a showcase of discussion, never a gate."""

ALIAS_RULE: Final = (
    "Name every target and position by its floor alias: a point (T1), a holding (H3), a message "
    "(M4) or a fact (S2); a member's role (ALPHA, RISK, CRO) is never a target, so answer its "
    "message by M alias. Type no other digit, since the Host renders every number."
)
"""How every member and the PM name what they speak of; NUMBER and TARGET refusals hold it."""
BELOW_FINDING: Final = "Below the agenda line, a HIGH Evidence negative on {holdings}"
"""The report's appendix row for a HIGH issue no stance judged, in the CRO's own words."""
TAKE_TURN: Final = (
    "Your one discussion turn is open: take it now, before {at}, when the window closes and "
    "silence counts as a pass. Send the offered challenge with your words, a REPLY, or the offered "
    "pass; never wait again with your turn open."
)
"""The open turn's words, with the discussion window's close."""

COMPLETE_FIRST_USE: Final = "The first use's road ends here: complete its goal."
"""What the first use's last answer says while its goal stands open."""

COMPLETE_DATE: Final = (
    "This date's road ends once the person has its report or the PM's view: give it to them, "
    "then complete the Session's open goal."
)
"""What a date's last answer says while the Session's goal stands open."""

COMPLETE_HOW: Final = (
    "Its offered `goal` read writes the completion to fill (`goal show <goal-id> "
    "--save-declaration <file>`): write your judgment, send it, and send each attach its answer "
    "offers for a reference it lacks."
)
"""How a goal's completion is written, from the read its road's end offers."""

NO_PM_STANCE: Final = (
    "The PM gave no stance, so the floor never reached its debate; its time box closed it "
    "without a verdict."
)

SUGGESTIONS: Final = {
    "OBJECT": "Consider avoiding or reducing {holdings}: {member}",
    "SUPPORT": "Could keep or add to {holdings}: {member}",
}
"""The report's lead: a member's standing OBJECT or SUPPORT on holdings, as a suggestion."""
NO_SUGGESTION: Final = "No member suggested avoiding, reducing or adding a holding"
NO_NUMBER: Final = (
    "The committee changed no number; whether to act on its suggestions is the person's decision."
)

MEMBER_COMMANDS: Final = ("committee submit", "committee wait")
"""What a member's own route runs; its card's allowlist names the same (materialize)."""

OPENER_WORDS: Final = (
    "The PM's key and route belong to the session that opened this floor; its wait goes on "
    "there. Read the floor here, or continue in that session."
)
"""Why another session's open names no key, and its way on."""

DAILY_ASK: Final = (
    "Should each daily update also convene the investment committee, about ten more minutes? "
    "Without it, the PM gives its view on the reviewed positions."
)
"""The one question the first use's report asks, word for word, until the person answers it."""
DAILY_WORDS: Final = {
    "ON": "Each daily update's published review convenes the investment committee.",
    "OFF": "A daily update's published review gets the PM's view on the reviewed positions; the "
    "committee is offered, not convened.",
    "UNANSWERED": "Never answered, so off: a daily update's review gets the PM's view, and the "
    "committee is offered.",
}

LEAD_PUBLISHED: Final = (
    "Nothing was submitted for this bundle; the date's review is already published, so its "
    "committee is offered."
)
"""A refusal to the lead where the review stands published: the committee, not a new bundle."""

RELAY_WORD: Final = "Relayed by the agent"
PASSED: Final = "{member} passed its discussion turn"
"""A yielded turn, as the report reads it."""

PASSED_SILENT: Final = "{member} passed its discussion turn (silent)"
"""A turn the discussion window closed on silence, as the report reads it."""
UNRULED_ON: Final = "Challenge stood unruled on {targets}"

ROUTE_STANCE: Final = (
    "On the committee's floor you are {role}. Send one STANCE first (the round is blind); "
    "its position on each {point} point (T1, never a question) is about the date's "
    "positions: SUPPORT, they may stand; OBJECT, they should not stand as they are; RESERVE, "
    "this bundle cannot tell, citing by S alias the facts it rests on. A standing OBJECT or "
    "SUPPORT leads the report as a suggestion on the holdings its point names, less any you "
    "set apart: give one its own position (H5: RESERVE), or name it in a sentence on that "
    "point with that one position word. You may suggest avoiding, reducing or adding "
    "specific holdings, with reasons from the materials; the person decides."
)
"""A member route's first step, its stance, for its role and the floor's kind of point."""

ROUTE_END: Final = (
    "This route is the bundle's only final action: there is no answer file and nothing for "
    "the lead to submit. Its steps: send your STANCE, wait, then take the turn the wait "
    "offers (TAKE_YOUR_DISCUSSION_TURN: a CHALLENGE or REPLY, or its offered pass) and wait "
    "again until the floor settles or closes. The lead starts you as a general subagent."
)
"""A member route's last step: the route is its only final action."""
