"""Sector Research: the owner of Sector state, targets and forecast methods.

Holds the numerical mechanics of Sector modelling and the causal Sector state
surface. It does not own Alpha's fold-fitted training transforms, and it does not
publish current state -- the Goal/current path keeps its own writer for the
legacy compatibility method that lives here.
"""
