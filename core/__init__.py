"""
Core Module Initialization
==========================

Initializes all core components of CopilotX including quantum computing,
AI engine, and foundational systems.
"""

from .quantum_core import QuantumCore, QuantumResult, QuantumState
from .ai_engine import AIEngine, IntelligenceContext, AIResponse

__all__ = [
    'QuantumCore',
    'QuantumResult', 
    'QuantumState',
    'AIEngine',
    'IntelligenceContext',
    'AIResponse'
]

__version__ = "1.0.0"
__author__ = "CopilotX Development Team"