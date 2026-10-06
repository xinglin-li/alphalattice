# Product Browser Operation
Date: 2026-09-25

Playwright and its CLI are product dependencies for human/agent browser operation
of Local Web, and are also used for QA. They are declared in `dependencies`, not
`devDependencies`. Numerical execution and ordinary manual browser access do not
need to launch Playwright. This is the maintained Windows source-checkout setup,
not a claim that a public installer or other operating systems have been accepted.

## Setup and use

Install Node.js >=20 with npm, then from the product checkout:

```powershell
./scripts/playwright.ps1 setup
./scripts/playwright.ps1 check
./scripts/playwright.ps1 run --help
./scripts/playwright.ps1 run -s=my-research open <local-web-url>
./scripts/playwright.ps1 run -s=my-research snapshot
./scripts/playwright.ps1 run -s=my-research close
```

`setup` uses the committed npm lock (including integrity hashes), installs production
dependencies, and obtains the matching Chromium headless shell if absent. It may
access the package/browser distribution endpoints; it does not enable business-data
network access. `check` and `run` never install or upgrade. A failed check reports
the missing dependency and setup action instead of relying on a global CLI.

The exact existing pins are Microsoft `@playwright/cli` 0.1.19 and Playwright
`1.63.0-alpha-2026-08-31`, Apache-2.0. The alpha library pin is inherited from that
CLI's exact dependency, not chosen as a floating release. The lock and installed
browser manifest own the browser revision. A version upgrade requires its own
compatibility check; this promotion does not upgrade package or browser bytes.

Node modules and managed browsers live under this directory and are ignored.
The Python `.venv`, Python dependency lock, research identities and workspace are
not modified by installation. Use a unique browser session per task; close only
your session. Keep screenshots/traces in private ignored output, never source Git.
Neither browser access nor a CLI request grants permission for product actions.

## Acceptance boundary

`check` establishes package versions and presence of the managed headless browser;
it does not prove a user journey or browser launch. Use `run -s=<session> open`
and `snapshot` for that proof, then `close` only the owned session. The
operating-flow QA record contains the pinned-browser observation for this
checkout. Product onboarding uses this setup/check path; it does not assume a
developer's global Chrome or CLI.
