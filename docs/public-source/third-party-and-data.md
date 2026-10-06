# Downloads, data sources and terms
Date: 2026-10-03

AlphaLattice is Apache-2.0. Downloaded software, models and acquired data keep their own terms; [NOTICE](../../NOTICE) lists included third-party material and licenses of separately installed model packs.

## Network and installation authority

Workspace network access starts closed. The first-use `FIRST_USE` goal's 24-hour delegation, later person confirmation, offline override and the separate consent for model installation/live Evidence are described in [Getting started](getting-started.md). `ALPHALATTICE_NETWORK_DISABLED=1` keeps workspace acquisition offline even when the control permits access; research calculations are held offline. Model installation downloads only with its explicit `--network` flag, separately from workspace authority and the offline override. Use local import when you already hold the pinned files.

## Local retrieval model packs

Evidence uses local encoders/rerankers. An installer selects the two packs a recipe needs rather than downloading all supported packs. With admitted installation, it fetches the named files over HTTPS from `https://huggingface.co/<repository>/resolve/<revision>/<file>` and verifies pinned hashes. Byte estimates are code `approximate_bytes`, not measured transfer totals; revisions are exact product pins and licenses follow NOTICE.

| Repository | Pinned revision | Estimated bytes | License in NOTICE |
| --- | --- | ---: | --- |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | 134,037,628 | MIT |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | 475,370,661 | Apache-2.0 |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1,207,469,357 | Apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 1,207,471,008 | Apache-2.0 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | `a09144355adeed5f58c8ed011d209bf8ee5a1fec` | 91,707,303 | Apache-2.0; ONNX weights converted from `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Verified packs are reused; interrupted installation retains staged files and can resume. On Windows the default model store is `%LOCALAPPDATA%/AlphaLattice/models`; `ALPHALATTICE_MODEL_STORE` or installer `--store` changes it. Workspaces share linked pack files. Runtime loaders read local files; a study never silently installs a missing model. Locked Python packages install during checkout setup; Playwright browser tooling is separate. Model files and Python packages retain their own licenses.

## Market data: Yahoo Finance

The installed US profile uses `yfinance` for USD research on the `XNYS_XNAS` calendar, with split-adjusted daily prices and validity `CURRENT_UNIVERSE_RESEARCH_ONLY`.

| Acquired | Provider call | Timing |
| --- | --- | --- |
| Daily OHLCV, adjusted-close observations and corporate actions for admitted listings | `yfinance.Ticker(...).history(...)`; hydration also reads adjustment history | First preparation and maintenance when required history is absent/stale |
| SPY market-reference prices and adjustment history | Same provider through market-reference maintainer | When required reference is not verified |
| Current Sector labels (`sector`, `sectorKey`) | `yfinance.Ticker(...).get_info()` | When Sector reference is absent or refresh is due |

The data comes from Yahoo Finance; Yahoo's terms govern its use. Observations stay in the workspace for that workspace's research and are not redistributed. The library license is not a data license. Sector labels are current observations, not historical GICS; newly observed labels have effective sessions and published sessions retain prior values. Current index membership and historical backfill do not establish a survivorship-free point-in-time universe.

## Current membership: Wikipedia

Candidate membership is the union of three current tables fetched with `httpx`:

| Index | Source |
| --- | --- |
| S&P 500 | [List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) |
| NASDAQ-100 | [List of NASDAQ-100 companies](https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies) |
| DJIA | [List of Dow Jones Industrial Average companies](https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies) |

Membership is captured during first preparation and a due maintenance check; a readiness read does not fetch it. Records retain source URL, retrieval time, response hash and membership evidence. Wikipedia page text is under CC BY-SA 4.0 and GFDL; reuse credits authors with a source link. The workspace retains captured membership with its URL for local research and does not redistribute it.

## Official filings: SEC EDGAR

Admitted live Evidence preparation uses the `httpx` SEC client to read `https://www.sec.gov/files/company_tickers.json`, filing inventories at `https://data.sec.gov/submissions/CIK<10-digit-CIK>.json`, and selected filing bodies under `https://www.sec.gov/Archives/edgar/data/`. The client supports explicitly requested XBRL company facts at `https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit-CIK>.json`; Local Web's live Evidence policy requests filings.

The client sends the declared `SEC_USER_AGENT` after trimming whitespace and requiring an `@` contact; there is no default contact. It allows at most 5 requests/second per client (at least 0.2 seconds between starts), caps responses/attempts, retries 429 and server errors with delays and refuses URLs outside allowed SEC paths. This is not a machine-wide rate limit.

Live admission requires explicit source consent, permitted workspace network access and the declared SEC contact. Without live admission, Evidence uses its recorded package offline; verified local filings can be reused without claiming fresh acquisition. Locally supplied official documents are marked `USER_PROVIDED_FOR_LOCAL_RESEARCH`; their declared access right is not a license granted by AlphaLattice.

## Cost and scope

Built-in acquisition and retrieval use no paid data feed or hosted model service. Product paths have no billing step; Codex/Claude Code access is separate. For updates, local storage, backup and restore, see [Data and care](data-and-care.md).
