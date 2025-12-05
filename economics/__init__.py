"""
Economics package - Aggregate computations and utilities.
"""

from .aggregates import (
    MacroState, 
    AggregateComputer, 
    compute_gini, 
    compute_wealth_distribution
)

__all__ = [
    'MacroState',
    'AggregateComputer', 
    'compute_gini',
    'compute_wealth_distribution',
]