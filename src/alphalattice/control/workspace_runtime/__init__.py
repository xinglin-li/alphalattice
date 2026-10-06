"""Concrete local-workspace runtime infrastructure for the playpen.

The modules here deliberately own only process-local coordination and trusted
artifact resolution. They are not a generic workflow platform. Consumers use
explicit owner modules so artifact-only imports do not initialize checkpoint
frameworks.
"""
