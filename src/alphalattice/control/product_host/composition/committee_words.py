"""The committee floor's words: each rule's refusal, the verdict's rule and the alias rule.

Owned by the committee floor (`committee.py`), which reads them; split from it for size, so
the floor's logic and the sentences agents learn from are kept apart.
"""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.project_layout import command_prefix

COMMAND: Final = " ".join(command_prefix())
RULE_WORDS: Final = {
    "ROLE": "Only its senders send this kind, each with the key in its own bundle: a STANCE or "
    "REPLY any member, a CHALLENGE or PASS a specialist, a RULING, VERDICT or PERSON_ANSWER the "
    "PM.",
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
    "MEMBER": "This bundle's member submits from its own session with its key and takes the Host's "
    "corrections itself; start it with its bundle, then run this again for its receipt.",
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
