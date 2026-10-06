# Unified Observation Ledger G0 case study

This fixture proves the public local observation boundary without a Provider,
model, market workspace, or second database. It covers deterministic source
ordering, idempotent append, correction, redaction-before-append, projection
checkpoint readback, tamper detection, bounded retention, truthful
availability, and protected-root failure behavior.

The Ledger records assertions and references. Task Control, domain markers,
artifact readers, and Guanyin remain the owners of the facts being observed.
