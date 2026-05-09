"""
CAFE-u AI Agent — LLM-powered adaptation decision engine with RAG.

Architecture:
  1. Signal arrives from browser/mobile agent
  2. ML Classifier predicts frustration probability
  3. RAG Store retrieves similar past signals + their successful adaptations
  4. LLM Agent decides which adaptation tool to use and with what params
  5. Adaptation is returned to the agent (JS/RN)
  6. Feedback loop: was it helpful? → store in RAG for future

This replaces the static YAML rules engine with a learning system.
"""

import os
import json
import logging
from typing import Optional


from .rag_store import RAGStore
from .classifier import FrustrationClassifier

logger = logging.getLogger("cafeu.agent")

# ── LLM client with graceful fallback ──────────────────────────

try:
    from openai import OpenAI as OpenAIAPI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not available — LLM agent will use rule-based fallback")


# ── Tool Definitions ───────────────────────────────────────────

ADAPTATION_TOOLS = [
    {
        "name": "highlight",
        "description": "Add a pulsing border/glow around an element to draw attention",
        "params": {"selector": "CSS selector string"},
    },
    {
        "name": "tooltip",
        "description": "Show a floating help tooltip near an element",
        "params": {"selector": "CSS selector", "text": "tooltip message"},
    },
    {
        "name": "make_clickable",
        "description": "Make a non-interactive element look clickable (add cursor, hover effect)",
        "params": {"selector": "CSS selector", "tooltip": "optional hint text"},
    },
    {
        "name": "sticky_cta",
        "description": "Pin a call-to-action button to the bottom of the viewport",
        "params": {"selector": "CSS selector of the CTA"},
    },
    {
        "name": "inline_hint",
        "description": "Show a small hint below a form field",
        "params": {"selector": "CSS selector", "text": "hint message"},
    },
    {
        "name": "save_progress",
        "description": "Show a toast confirming progress was saved",
        "params": {"text": "confirmation message"},
    },
    {
        "name": "dim_section",
        "description": "Dim a broken/non-functional section and show fallback message",
        "params": {"selector": "CSS selector", "text": "fallback message"},
    },
]

TOOL_NAMES = [t["name"] for t in ADAPTATION_TOOLS]


class AIAgent:
    """
    AI-powered adaptation agent.
    
    Uses LLM + RAG to decide the best UI adaptation for each frustration signal.
    Falls back to rule-based heuristic when LLM is unavailable.
    """

    def __init__(self, classifier: Optional[FrustrationClassifier] = None):
        self.rag = RAGStore()
        self.classifier = classifier or FrustrationClassifier()
        self.llm_client = None
        self._init_llm()
        
        # Stats
        self.total_decisions = 0
        self.llm_decisions = 0
        self.fallback_decisions = 0

    def _init_llm(self):
        """Initialize LLM client from environment variables (no hardcoded keys)."""
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = os.environ.get("CAFEU_LLM_MODEL", "deepseek-v4-flash")
        
        if api_key and OPENAI_AVAILABLE:
            try:
                self.llm_client = OpenAIAPI(api_key=api_key, base_url=base_url)
                self.llm_model = model
                logger.info(f"LLM Agent initialized with model: {model}")
            except Exception as e:
                logger.warning(f"LLM init failed: {e}")
                self.llm_client = None
        else:
            logger.info("No LLM configured — agent will use rule-based fallback")
            logger.info("  Set DEEPSEEK_API_KEY or OPENAI_API_KEY env var to enable LLM")

    def decide(self, signal: dict, history: list[dict]) -> dict:
        """
        Main decision function.
        
        1. ML classifier → frustration probability
        2. RAG retrieval → similar past cases
        3. LLM → decide adaptation
        4. Return adaptation instruction
        
        Returns dict with 'adaptation' and 'metadata' keys.
        """
        self.total_decisions += 1
        
        # Step 1: ML classification
        prob = self.classifier.predict(signal, history)
        signal["frustration_probability"] = round(prob, 3)
        
        # Low frustration — no adaptation needed
        if prob < 0.3:
            return {"adaptation": None, "metadata": {"reason": "low_frustration", "probability": prob}}
        
        # Step 2: RAG retrieval
        similar_cases = self.rag.retrieve(signal, k=3)
        
        # Step 3: Decide adaptation
        if self.llm_client and prob >= 0.5:
            adaptation = self._llm_decide(signal, similar_cases)
            if adaptation:
                self.llm_decisions += 1
            else:
                adaptation = self._rule_decide(signal, similar_cases)
                self.fallback_decisions += 1
        else:
            adaptation = self._rule_decide(signal, similar_cases)
            self.fallback_decisions += 1
        
        if adaptation:
            # Step 4: Add ML & RAG metadata
            adaptation["_ml_probability"] = round(prob, 3)
            adaptation["_rag_matches"] = len(similar_cases)
            
            # Store in RAG for future
            self.rag.add(signal, adaptation, effective=True)
        
        return {
            "adaptation": adaptation,
            "metadata": {
                "probability": prob,
                "similar_cases": len(similar_cases),
                "decision_source": "llm" if (self.llm_client and prob >= 0.5) else "rules",
            }
        }

    def _llm_decide(self, signal: dict, similar_cases: list[dict]) -> Optional[dict]:
        """Use LLM to decide the best adaptation."""
        if not self.llm_client:
            return None
        
        system_prompt = (
            "You are a UI adaptation agent. Your job is to decide how to fix user frustration "
            "by choosing the best adaptation tool. You are given a frustration signal and "
            "similar past cases. Return ONLY a JSON object with the adaptation decision.\n\n"
            "Available tools:\n"
            + "\n".join(
                f"- {t['name']}: {t['description']} (params: {json.dumps(t['params'])})"
                for t in ADAPTATION_TOOLS
            ) + "\n\n"
            "Rules:\n"
            "- For rage clicks: use 'highlight' or 'tooltip'\n"
            "- For dead clicks: use 'make_clickable'\n"
            "- For hesitation: use 'inline_hint'\n"
            "- For scroll bounce: use 'sticky_cta'\n"
            "- For form abandon: use 'save_progress'\n"
            "- Response MUST be valid JSON with keys: tool, params, reasoning\n"
            "- If no adaptation needed, return: {\"tool\": null, \"reasoning\": \"...\"}"
        )

        user_prompt = (
            f"Current signal:\n"
            f"  Type: {signal.get('type', 'unknown')}\n"
            f"  Element: {signal.get('element', 'unknown')}\n"
            f"  Count: {signal.get('count', 1)}\n"
            f"  Duration: {signal.get('duration_ms', 0)}ms\n"
            f"  Frustration probability: {signal.get('frustration_probability', 0.5)}\n"
        )
        
        if similar_cases:
            user_prompt += "\nSimilar past cases:\n"
            for i, case in enumerate(similar_cases, 1):
                user_prompt += (
                    f"  Case {i}: {case['signal'].get('type')} on "
                    f"'{case['signal'].get('element', '?')}' → "
                    f"{case['adaptation'].get('action')} "
                    f"(effective: {case['memory'].get('effective', True)})\n"
                )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON from LLM response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            decision = json.loads(content)
            tool = decision.get("tool")
            
            if tool and tool in TOOL_NAMES:
                params = decision.get("params", {})
                # Map tool name to action name
                action_map = {
                    "highlight": "highlight",
                    "tooltip": "tooltip",
                    "make_clickable": "make-clickable", 
                    "sticky_cta": "sticky-cta",
                    "inline_hint": "inline-hint",
                    "save_progress": "save-progress",
                    "dim_section": "dim-section",
                }
                
                adaptation = {
                    "selector": signal.get("element", params.get("selector", "")),
                    "action": action_map.get(tool, tool),
                    "_llm_reasoning": decision.get("reasoning", ""),
                }
                
                # Merge additional params
                for k, v in params.items():
                    if k != "selector":
                        adaptation[k] = v
                
                logger.info(f"LLM decided: {adaptation['action']} on {adaptation['selector']}")
                return adaptation
            
            logger.debug(f"LLM returned null tool: {content[:100]}")
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM response wasn't valid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return None

    def _rule_decide(self, signal: dict, similar_cases: list[dict]) -> Optional[dict]:
        """
        Rule-based fallback decision.
        
        Uses signal type + similarity to past cases.
        Fires for moderate-to-high frustration signals (probability >= 0.5).
        """
        prob = signal.get("frustration_probability", 0.0)
        
        # Don't adapt for low probability signals
        if prob < 0.5:
            return None
        
        signal_type = signal.get("type", "")
        element = signal.get("element", "")
        
        # First, check if a similar past case suggests an adaptation
        for case in similar_cases:
            if case.get("memory", {}).get("effective", False):
                past_ad = case.get("adaptation", {})
                if past_ad:
                    adaptation = dict(past_ad)
                    adaptation["selector"] = element
                    adaptation["_from_rag"] = True
                    return adaptation
        
        # Fall back to type-based rules
        rule_map = {
            "rage_click": {"action": "highlight"},
            "dead_click": {"action": "make-clickable"},
            "hesitation": {"action": "inline-hint", "text": "💡 Need help here?"},
            "scroll_bounce": {"action": "sticky-cta"},
            "form_abandon": {"action": "save-progress", "text": "Your progress has been saved."},
            "rapid_tap": {"action": "tooltip", "text": "Having trouble?"},
            "long_press": {"action": "tooltip", "text": "Press and hold for options"},
        }
        
        base = rule_map.get(signal_type)
        if base:
            adaptation = dict(base)
            adaptation["selector"] = element
            
            # Increase intensity for repeated signals
            count = signal.get("count", 1) or 1
            frequency = signal.get("frequency", 1) or 1
            if count >= 4 or frequency >= 3:
                if adaptation["action"] == "highlight":
                    adaptation["action"] = "tooltip"
                    adaptation["text"] = "Can we help you with this?"
            
            logger.debug(f"Rule decided: {adaptation['action']} on {element}")
            return adaptation
        
        return None

    def report_feedback(self, signal: dict, adaptation: dict, was_helpful: bool):
        """Report whether an adaptation was helpful."""
        self.rag.report_feedback(signal, adaptation, was_helpful)

    def get_stats(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "llm_decisions": self.llm_decisions,
            "fallback_decisions": self.fallback_decisions,
            "llm_enabled": self.llm_client is not None,
            "rag": self.rag.stats(),
        }
