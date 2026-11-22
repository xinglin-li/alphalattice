# Downloads, data sources and terms
Last checked: 2026-10-22 against 0.2.3

AlphaLattice is Apache-2.0. Software, models and data retain their terms; [NOTICE](../../NOTICE) lists included material and separately installed model licenses.

## Network and installation

The [agent guide](../../AGENTS.md) owns first-use authority and [your choices](../../AGENTS.md#what-only-a-person-decides). Acquisition starts closed; `ALPHALATTICE_NETWORK_DISABLED=1` forces it offline; calculations stay offline. Installation's `--network` consent is separate from workspace authority and that override. Local import accepts held pinned files.

Each outbound request has a timeout and a source-specific list of HTTPS hosts. When a request follows a redirect, it checks the new destination before connecting. The list covers the existing Yahoo Finance, Wikipedia, SEC EDGAR, and Hugging Face sources, including the Hub's file-delivery hosts. The product's local Host remains reachable while outbound networking is disabled.

## Local retrieval models

Evidence retrieval runs on the Host's retrieval store, which uses two packs: a static encoder and an English reranker. Setup (`--install retrieval-store`) fetches their pinned files from `https://huggingface.co/<repository>/resolve/<revision>/<file>` and verifies their hashes. Packs of the earlier recipes that a store already holds stay readable but are no longer installed.

| Repository | Pinned revision | Estimated bytes | License |
| --- | --- | ---: | --- |
| `hotchpotch/bekko-static-embedding-v1` | `c18167b8f244c1bc6de21ce9b710bac8a29c4615` | 108,305,839 | MIT |
| `cross-encoder/ettin-reranker-17m-v1` | `9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6` | 20,976,635 | Apache-2.0 |

Verified packs are reused, and interrupted installations resume their staged files. On Windows, `%LOCALAPPDATA%/AlphaLattice/models` is the default store. `ALPHALATTICE_MODEL_STORE` or `--store` changes its location, and a store packs were installed into is remembered in the user's home at `~/.alphalattice/model-store.json`, so later runs find it without the variable. The older platform-store pointer is still read when the home pointer is absent. Workspaces share linked pack files. Runtime loaders read local files and never silently install missing models. Locked Python packages install during setup, while Playwright tooling installs separately. Both retain their licenses.

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

Requests name the product by its own contact, `AlphaLattice alphalattice.project@gmail.com`, an address the project maintains; an operator who must use another sets `SEC_USER_AGENT` to a contact containing `@`. Each client starts at most 2 requests per second, at least 0.5 seconds apart, well below the SEC's fair-access limit. Response and attempt caps, delayed retries for 429 and server errors, and allowed-path checks apply. This limit applies per client.

Live admission requires source consent and workspace network authority. Otherwise Evidence uses its recorded package offline. Reused local filings imply no fresh acquisition. User-supplied official documents carry declared access rights for local research; AlphaLattice grants no license.

## Cost and scope

Built-in acquisition and retrieval use no paid data feed or hosted model service. Product paths have no billing step. Access to Claude Code or Codex is separate. [Data and care](data-and-care.md) covers storage and recovery.
