"""
Reasoning Module
================

Multi-dimensional reasoning capabilities for CopilotX.
"""

from .multidimensional import (
    MultiDimensionalReasoning,
    ReasoningResult,
    ReasoningStep,
    ReasoningType,
    LogicalReasoner,
    InductiveReasoner,
    AbductiveReasoner,
    CausalReasoner,
    QuantumReasoner
)

__all__ = [
    'MultiDimensionalReasoning',
    'ReasoningResult',
    'ReasoningStep', 
    'ReasoningType',
    'LogicalReasoner',
    'InductiveReasoner',
    'AbductiveReasoner',
    'CausalReasoner',
    'QuantumReasoner'
]