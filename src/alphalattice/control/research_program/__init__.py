"""Desk-neutral orchestration of authored research programs.

This package knows how to parse an authoring document, route it to whichever
compiler was injected, and seal the result. It knows nothing about Risk, Factor,
Alpha, Portfolio, market data, or features: the Desk vocabulary it uses is the
Protocols in ``protocols/research_authoring``, and the concrete compilers and
executors arrive by injection from the product host.
"""
