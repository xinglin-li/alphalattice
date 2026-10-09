# Native sessions and Team
Date: 2026-10-08

A Session binds itself when it first works on a workspace, and its first research request opens a goal when it holds none. Binding starts no model, agent or Task and grants no execution or write rights. Each host Session has its own ignored local record, holding host, Session, workspace and roles, never a token or research authority; a new Session binds independently. Run `alphalattice session bind` only to rebind or to turn usage reading off, and `alphalattice session unbind` only to change your own Session's workspace, usage or roles. Never edit another Session's record.

`scripts/materialize_claude_host.py` derives the Claude cards and the byte-identical Skill copy from the Codex cards and this Skill; `--check` reports drift. The Analyst and CRO cards keep their restricted read and answer tools.

## Team and the Goal's Conversation

Team and the Goal's Conversation show what the product recorded of a bound Session's work: its requests, the bundles it prepared, the answers it submitted and those the Host accepted, and its usage. There is no message command; record a decision worth keeping with `goal note` ([goals](goals.md)). The lead submits every answer, so an accepted answer is filed as the lead's, its author `NOT_OBSERVED`. A prepared bundle with no accepted answer is an open assignment in `goal show`; it reminds and never blocks submission.

## Usage

With reading on, the Host reads the bound Session's own session file and the specialists that file records, when a goal is taken or submitted, an answer is submitted, a Team or Goal page opens, or `alphalattice session usage` asks; never on a timer and never another Session's file. It keeps models, efforts, token counts and times and discards conversation and tools. A format it does not know reads as unavailable, never as zero, and research continues. `session bind --usage off` keeps one Session unread; the person's switch on **Settings** turns reading off for every Session of the workspace, and only a person sets it.
