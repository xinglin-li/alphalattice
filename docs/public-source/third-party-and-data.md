# Downloads, data sources and terms
Date: 2026-10-08

AlphaLattice is Apache-2.0. Software, models and data retain their terms; [NOTICE](../../NOTICE) lists included material and separately installed model licenses.

## Network and installation

The [agent guide](../../AGENTS.md) owns first-use authority and [your choices](../../AGENTS.md#what-only-a-person-decides). Acquisition starts closed; `ALPHALATTICE_NETWORK_DISABLED=1` forces it offline; calculations stay offline. Installation's `--network` consent is separate from workspace authority and that override. Local import accepts held pinned files.

## Local retrieval models

Each Evidence recipe selects two packs for its local encoder and reranker. Installation fetches pinned files from `https://huggingface.co/<repository>/resolve/<revision>/<file>` and verifies their hashes. The code's `approximate_bytes` values are estimates rather than measured transfers. NOTICE lists the licenses.

| Repository | Pinned revision | Estimated bytes | License in NOTICE |
| --- | --- | ---: | --- |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | 134,037,628 | MIT |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | 475,370,661 | Apache-2.0 |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1,207,469,357 | Apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 1,207,471,008 | Apache-2.0 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | `a09144355adeed5f58c8ed011d209bf8ee5a1fec` | 91,707,303 | Apache-2.0; ONNX weights converted from `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Verified packs are reused, and interrupted installations resume their staged files. On Windows, `%LOCALAPPDATA%/AlphaLattice/models` is the default store. `ALPHALATTICE_MODEL_STORE` or `--store` changes its location. Workspaces share linked pack files. Runtime loaders read local files and never silently install missing models. Locked Python packages install during setup, while Playwright tooling installs separately. Both retain their licenses.

## Yahoo Finance

The US profile uses `yfinance`, USD, the `XNYS_XNAS` calendar and split-adjusted daily prices for current-universe research.

| Acquired | Provider call | Timing |
| --- | --- | --- |
| Daily OHLCV, adjusted-close observations and corporate actions are acquired for admitted listings. | `yfinance.Ticker(...).history(...)` supplies observations. Hydration also reads adjustment history. | Acquisition occurs during first preparation and maintenance when required history is absent or stale. |
| SPY prices and adjustment history provide the market reference. | The market-reference maintainer uses the same provider. | Acquisition occurs when the required reference is not verified. |
| Current Sector labels use `sector` and `sectorKey`. | `yfinance.Ticker(...).get_info()` supplies the labels. | Acquisition occurs when the Sector reference is absent or its refresh is due. |

Yahoo's terms govern data, and the library grants no data license. Observations stay in workspace research without redistribution. Sector labels are current observations rather than historical GICS. Effective sessions preserve published values. Current membership and historical backfill do not establish a survivorship-free point-in-time universe.

## Wikipedia membership

Preparation and due maintenance capture the union of three current tables with `httpx`; readiness reads fetch nothing:

| Index | Source |
| --- | --- |
| S&P 500 | [List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) |
| NASDAQ-100 | [List of NASDAQ-100 companies](https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies) |
| DJIA | [List of Dow Jones Industrial Average companies](https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies) |

Records keep the URL, retrieval time, response hash and membership evidence. Wikipedia text is under CC BY-SA 4.0 and GFDL. Reuse credits authors with a source link. Captured membership and its URLs remain local without redistribution.

## SEC EDGAR filings

Admitted live Evidence uses `httpx` to read `https://www.sec.gov/files/company_tickers.json`, `https://data.sec.gov/submissions/CIK<10-digit-CIK>.json` and selected bodies under `https://www.sec.gov/Archives/edgar/data/`. Explicit XBRL requests use `https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit-CIK>.json`; Local Web's live policy requests filings.

`SEC_USER_AGENT` requires a trimmed contact containing `@`, without a default. Each client starts at most 5 requests per second, at least 0.2 seconds apart. Response and attempt caps, delayed retries for 429 and server errors, and allowed-path checks apply. This limit applies per client.

Live admission requires source consent, workspace network authority and SEC contact. Otherwise Evidence uses its recorded package offline. Reused local filings imply no fresh acquisition. User-supplied official documents carry declared access rights for local research; AlphaLattice grants no license.

## Cost and scope

Built-in acquisition and retrieval use no paid data feed or hosted model service. Product paths have no billing step. Access to Codex or Claude Code is separate. [Data and care](data-and-care.md) covers storage and recovery.
