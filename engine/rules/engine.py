"""
CAFE-u Rules Engine — Maps signals to adaptations.

Now with ML-powered frustration classification.
Rules are loaded from YAML files in rules/definitions/.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    yaml = None

from .classifier import FrustrationClassifier
from .agent import AIAgent

logger = logging.getLogger("cafeu.engine")


class Rule:
    """A single adaptation rule."""
    
    def __init__(self, data: dict):
        self.name = data.get("name", "unnamed")
        self.description = data.get("description", "")
        self.signal = data.get("signal", "")
        self.threshold = data.get("threshold", {})
        self.action = data.get("action", "")
        self.params = data.get("params", {})
        self.priority = data.get("priority", 5)
        self.cooldown_ms = data.get("cooldown_ms", 8000)
    
    def matches(self, signal: dict) -> bool:
        """Check if this rule applies to the given signal."""
        if signal.get("type") != self.signal:
            return False
        
        threshold = self.threshold
        
        # Count-based threshold (e.g., rage click >= 3)
        if "count" in threshold:
            if signal.get("count", 0) < threshold["count"]:
                return False
        
        # Time-based threshold (e.g., hesitation > 5000ms)
        if "duration_ms" in threshold:
            if signal.get("duration_ms", 0) < threshold["duration_ms"]:
                return False
        
        # Frequency threshold (e.g., same element clicked N times)
        if "frequency" in threshold:
            if signal.get("frequency", 0) < threshold["frequency"]:
                return False
        
        return True
    
    def build_adaptation(self, signal: dict) -> dict:
        """Build an adaptation instruction from this rule + signal context."""
        adaptation = {
            "selector": signal.get("element", ""),
            "action": self.action,
        }
        
        # Merge rule params with signal context
        adaptation.update(self.params)
        
        # Add signal-specific context
        if signal.get("field_type"):
            adaptation["field_type"] = signal["field_type"]
        if signal.get("count"):
            adaptation["count"] = signal["count"]
        
        return adaptation
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "signal": self.signal,
            "action": self.action,
            "threshold": self.threshold,
            "priority": self.priority,
        }


class RulesEngine:
    """Processes signals and returns adaptations using loaded rules."""
    
    def __init__(self, rules_dir: Optional[Path] = None):
        self.rules: list[Rule] = []
        self.signal_history: list[dict] = []
        self.adaptation_log: list[dict] = []
        self.cooldowns: dict[str, float] = {}  # selector -> last adapted timestamp
        self.max_history = 1000
        self.ml_threshold = 0.7  # Frustration probability threshold for ML mode
        
        if rules_dir and rules_dir.exists():
            self.rules_dir = rules_dir
            self.load_rules()
        else:
            self.rules_dir = None
            self._load_default_rules()
        
        # Initialize ML classifier
        self.classifier = FrustrationClassifier()
        
        # Initialize AI Agent with RAG + LLM
        self.agent = AIAgent(classifier=self.classifier)
        
        logger.info(f"Loaded {len(self.rules)} adaptation rules | ML: {'trained' if self.classifier.trained else 'cold start'} | Agent: {'LLM' if self.agent.llm_client else 'rules'}")
    
    def load_rules(self):
        """Load rules from YAML files in the rules directory."""
        self.rules = []
        for yaml_file in sorted(self.rules_dir.glob("*.yaml")):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data and "rules" in data:
                        for rule_data in data["rules"]:
                            self.rules.append(Rule(rule_data))
                            logger.debug(f"  Loaded rule: {rule_data.get('name')}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
    
    def _load_default_rules(self):
        """Built-in rules as fallback."""
        self.rules = [
            Rule({
                "name": "rage-click-highlight",
                "description": "Highlight element after 3+ rapid clicks",
                "signal": "rage_click",
                "threshold": {"count": 3},
                "action": "highlight",
                "params": {},
                "priority": 8,
            }),
            Rule({
                "name": "rage-click-tooltip",
                "description": "Show floating tooltip on rage-clicked element",
                "signal": "rage_click",
                "threshold": {"count": 4},
                "action": "tooltip",
                "params": {"text": "Can we help you with this?"},
                "priority": 7,
            }),
            Rule({
                "name": "dead-click-make-interactive",
                "description": "Make dead element look clickable",
                "signal": "dead_click",
                "threshold": {},
                "action": "make-clickable",
                "params": {},
                "priority": 6,
            }),
            Rule({
                "name": "hesitation-inline-hint",
                "description": "Show inline hint for hesitated form fields",
                "signal": "hesitation",
                "threshold": {"duration_ms": 5000},
                "action": "inline-hint",
                "params": {"text": "💡 Here's a hint for this field"},
                "priority": 5,
            }),
            Rule({
                "name": "scroll-bounce-sticky-cta",
                "description": "Show sticky CTA when user bounces on scroll",
                "signal": "scroll_bounce",
                "threshold": {},
                "action": "sticky-cta",
                "params": {},
                "priority": 4,
            }),
            Rule({
                "name": "form-abandon-save",
                "description": "Save form progress on abandon",
                "signal": "form_abandon",
                "threshold": {},
                "action": "save-progress",
                "params": {},
                "priority": 3,
            }),
        ]
    
    def process(self, signals: list[dict]) -> list[dict]:
        """
        Process a batch of signals and return adaptation instructions.
        
        This is the core decision function. It:
        1. Appends signals to history
        2. Computes frequency features
        3. Matches signals against rules
        4. Checks cooldowns
        5. Returns adaptations
        """
        adaptations = []
        
        for signal in signals:
            # Add computed features
            signal = self._enrich_signal(signal)
            
            # Store in history
            self.signal_history.append(signal)
            if len(self.signal_history) > self.max_history:
                self.signal_history = self.signal_history[-self.max_history:]
            
            # Use AI Agent with RAG + LLM (or rule fallback)
            result = self.agent.decide(signal, self.signal_history)
            adaptation = result.get("adaptation")
            
            if adaptation:
                selector = adaptation.get("selector", "")
                action = adaptation.get("action", "")
                
                # Check cooldown
                cooldown_key = f"{selector}:{action}"
                last_adapted = self.cooldowns.get(cooldown_key, 0)
                now = datetime.now(timezone.utc).timestamp() * 1000
                if now - last_adapted < self._get_cooldown(action):
                    continue
                
                # Clean internal metadata before sending to client
                clean_ad = {k: v for k, v in adaptation.items() if not k.startswith("_")}
                adaptations.append(clean_ad)
                
                # Set cooldown
                self.cooldowns[cooldown_key] = now
                self.adaptation_log.append({
                    "timestamp": now,
                    "signal": signal.get("type"),
                    "selector": selector,
                    "action": action,
                    "metadata": result.get("metadata", {}),
                })
        
        return adaptations
    
    def _enrich_signal(self, signal: dict) -> dict:
        """Add computed features to a signal, including ML frustration score."""
        signal_type = signal.get("type", "")
        selector = signal.get("element", "")
        now = datetime.now(timezone.utc).timestamp() * 1000
        
        # Compute frequency: how many times has this element been clicked recently?
        if signal_type in ("rage_click", "dead_click", "click"):
            recent = [
                s for s in self.signal_history
                if s.get("element") == selector
                and s.get("type") == signal_type
                and now - s.get("timestamp", 0) < 30000
            ]
            signal["frequency"] = len(recent) + 1
        
        # ML frustration score
        try:
            prob = self.classifier.predict(signal, self.signal_history)
            signal["frustration_probability"] = round(prob, 3)
            signal["ml_triggered"] = prob >= self.ml_threshold
        except Exception as e:
            signal["frustration_probability"] = 0.0
            signal["ml_triggered"] = False
        
        return signal
    
    def list_rules(self) -> list[dict]:
        return [r.to_dict() for r in self.rules]
    
    def get_ml_stats(self) -> dict:
        """Return ML classifier + Agent statistics."""
        return {
            "classifier": self.classifier.get_stats(),
            "agent": self.agent.get_stats(),
        }
    
    def _get_cooldown(self, action: str) -> int:
        """Get cooldown period for an action type in ms."""
        cooldowns = {
            "highlight": 8000,
            "tooltip": 15000,
            "make-clickable": 10000,
            "sticky-cta": 30000,
            "inline-hint": 30000,
            "save-progress": 60000,
            "dim-section": 60000,
        }
        return cooldowns.get(action, 10000)
