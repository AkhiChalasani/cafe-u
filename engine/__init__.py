"""
CAFE-u Engine — Adaptive UI Decision Engine

Architecture:
  Signal Ingestion (WS/HTTP)
      │
  Classifier (frequency → ML)
      │
  Rules Engine (YAML rules → matches)
      │
  Adaptation Builder (instruction → JSON)
      │
  Agent applies adaptation in browser

This is where the "intelligence" of CAFE-u lives.
"""

__version__ = "0.1.0"
