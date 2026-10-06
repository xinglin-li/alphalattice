# Delivery
Date: 2026-10-03

Use only the exact `delivery` request from a completed Portfolio readback. It binds the book Task, receipt and selected holdings date.

- Name both comparison Tasks from the actual comparison, in order, including the selected book. Incompatible inputs yield an uncomputed section, never invented metrics.
- Select existing `risk_report_hash` and, when wanted, exact `review_publication_hash`. The product validates both. Missing evidence stays visible; never substitute a latest review or imply zero risk. A Risk link is report-only and may not cover the holdings date. An expired CRO review is historical, not current clearance.
- Trace lineage by declared bindings: Alpha's `foundation_hash` is an `AlphaDevelopmentFoundationBinding`, not the Foundation readback's `ResearchFoundationBinding`; join through `foundation_admission_hash`, Factor and curation receipts, and consumed Feature and Panel bindings.
- `delivery_question` records this delivery's question, not an invented intent. Commentary keeps each specialist's final text with attribution; label a summary. Commentary changes no calculation or review.

Take the returned request and choices, then keep the resulting JSON. It holds constituent data, bounded summary, commentary, HTML and export hash; `--format html` saves HTML to a new path. Reopen through the exact offered request to revalidate the selection without new research. Later expiry changes current eligibility, not the stored review. A local export grants no permission to upload private research or data.
