# Committee refusals and their fixes

A refused committee message answers `committee.message_refused:<RULE>` with the rule's words in
`detail`, its `next_action` and, where only the message's shape or a typed name is wrong, the same
message corrected under `next_requests.corrected`: send that as it stands. Otherwise fix the field
the rule names and send again. A member's key is in its own bundle; the PM's is the open's `pm_key`.

Messages name points (T1), holdings (H3), messages (M4) and facts (S2) by alias, and type no other
digit. `--seen` takes a message's number: the answer's `seen`, so message M7 is `--seen 7`.

| Rule | What was wrong | Corrected example |
|---|---|---|
| ROLE | The kind is not this member's: STANCE and REPLY any member; CHALLENGE and PASS a specialist; RULING, VERDICT and PERSON_ANSWER the PM. Its `may_send` lists what this sender may send now. | A specialist that disagrees sends `kind: CHALLENGE`, never `kind: RULING`; the PM has no turn to pass, so it sends a REPLY, a RULING or the VERDICT. |
| STANCES | The blind round takes only one STANCE from each member. | Send the STANCE; run `committee wait` until the stances are revealed, then challenge or reply. |
| DEBATE | The stances are revealed; a STANCE is no longer taken. | Send `kind: REPLY` with `reply_to: M2` instead. |
| CLOSED | The floor closed by its verdict or its time box. | A specialist sends nothing more and returns its receipt. The PM sends the offered `report` request, or `kind: PERSON_ANSWER` with `reply_to` naming the item handed to the person. |
| LATE | A stance after the blind round's window, or a turn after the discussion window. | Nothing more is taken from this member: return what you would have said to your lead. Take a turn before the answer's `discussion_closes_at`. |
| STANCE_IN | This member's stance is already on the floor. | After the reveal, send `kind: CHALLENGE` or `kind: REPLY` with `reply_to: M3`. |
| CAP | This member has sent its three challenges or replies. | Nothing more: the floor takes no further message from it. |
| TURN | A PASS came after the member spoke, before its stance, or with a target or positions. | `kind: PASS`, `text: Nothing to add.`, with no `targets`, `positions` or `reply_to`. |
| TARGET | A target, position or alias names nothing on the floor, or names a role. | `targets: [M4]`, not `targets: [CRO]`; `positions: {T1: OBJECT}`, not `{"Turnover": OBJECT}`. |
| REPLY_TO | The named message cannot be answered by this kind. | A RULING answers an unruled challenge: `reply_to: M5` where M5 is a CHALLENGE. |
| SHAPE | `reply_to` or `outcome` on a kind that does not take it, or missing where required. | A CHALLENGE drops `reply_to`; a RULING carries `reply_to: M5` and `outcome: ADOPT`; a VERDICT carries `outcome: PROCEED_WITH_NOTES` and no `reply_to`. |
| UNRULED | The verdict came while a challenge stands unruled. | Send the offered `kind: RULING`, `reply_to: M5`, `outcome: REJECT` with your words, then the VERDICT. |
| FACTS | A RESERVE cites none of the floor's facts for this member's questions. | `positions: {T2: RESERVE}` with text that cites `S3`. |
| NUMBER | A digit typed outside an alias. | `text: H3 carries the most of T1's risk.`, not `AAPL carries 12% of the risk.` |
| AGENDA | Positions keyed outside the agenda, or by a fact alias. | `positions: {T1: SUPPORT}` or `{H3: OBJECT}`, never `{S2: SUPPORT}`: an S alias is cited in the text. The PM may raise one below-line point through a REPLY's positions. |
| MEMBER | A member's submit came without its key, or from another session. | Run the command under Your route in the bundle's README.md as it stands: it carries `--key` and the key. A lead starts the member with its bundle instead. |
| KEY | The key belongs to another member or bundle. | Use the key in this member's own bundle; the PM's is the open's `pm_key`. |
| ENDED | The member's thread has ended. | Return the receipt it gave to your lead. |

`committee.not_open` names the update without an open floor: run `committee open --update <task>`
on the date's published review. `committee.daily_setting_human_only` means the daily setting is the
person's: ask once and relay their words with `--person-said`.
