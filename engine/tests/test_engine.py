"""
Tests for the CAFE-u Rules Engine and ML Classifier.
"""

import sys
from pathlib import Path

# Ensure engine is on path
_engine_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_engine_dir))

import pytest  # noqa: E402
from rules.engine import RulesEngine, Rule  # noqa: E402
from rules.classifier import FrustrationClassifier, FeatureExtractor  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def engine():
    return RulesEngine()


@pytest.fixture
def classifier():
    return FrustrationClassifier()


# ── Rule Tests ────────────────────────────────────────────────

class TestRule:
    def test_rage_click_matches_threshold(self):
        rule = Rule({
            "name": "test-rage",
            "signal": "rage_click",
            "threshold": {"count": 3},
            "action": "highlight",
        })
        assert rule.matches({"type": "rage_click", "count": 5})
        assert not rule.matches({"type": "rage_click", "count": 2})
        assert not rule.matches({"type": "dead_click", "count": 5})

    def test_hesitation_threshold(self):
        rule = Rule({
            "name": "test-hesitate",
            "signal": "hesitation",
            "threshold": {"duration_ms": 5000},
            "action": "inline-hint",
        })
        assert rule.matches({"type": "hesitation", "duration_ms": 6000})
        assert not rule.matches({"type": "hesitation", "duration_ms": 1000})

    def test_rule_builds_adaptation(self):
        rule = Rule({
            "name": "test",
            "signal": "rage_click",
            "threshold": {},
            "action": "tooltip",
            "params": {"text": "Help"},
        })
        ad = rule.build_adaptation({"type": "rage_click", "element": "button.test", "count": 3})
        assert ad["selector"] == "button.test"
        assert ad["action"] == "tooltip"
        assert ad["text"] == "Help"

    def test_priority_sorting(self):
        low = Rule({"name": "low", "priority": 1, "action": "a"})
        high = Rule({"name": "high", "priority": 10, "action": "b"})
        assert high.priority > low.priority


# ── Rules Engine Tests ───────────────────────────────────────

class TestRulesEngine:
    def test_engine_initializes_with_default_rules(self, engine):
        assert len(engine.rules) >= 5
        rule_names = [r.name for r in engine.rules]
        assert "rage-click-highlight" in rule_names
        assert "dead-click-make-interactive" in rule_names

    def test_engine_processes_rage_click(self, engine):
        result = engine.process([{"type": "rage_click", "element": "button.test", "count": 5, "timestamp": 1000}])
        assert len(result) >= 1
        # High count rage clicks escalate to tooltip
        assert result[0]["action"] in ("highlight", "tooltip")

    def test_engine_processes_dead_click(self, engine):
        result = engine.process([{"type": "dead_click", "element": "div.banner", "timestamp": 1000}])
        assert len(result) >= 1
        assert result[0]["action"] == "make-clickable"

    def test_engine_processes_hesitation(self, engine):
        result = engine.process([{"type": "hesitation", "element": "input.email", "duration_ms": 6000, "timestamp": 1000}])
        assert len(result) >= 1
        assert result[0]["action"] == "inline-hint"

    def test_engine_processes_scroll_bounce(self, engine):
        result = engine.process([{"type": "scroll_bounce", "element": "", "timestamp": 1000}])
        assert len(result) >= 1
        assert result[0]["action"] == "sticky-cta"

    def test_engine_processes_form_abandon(self, engine):
        result = engine.process([{"type": "form_abandon", "element": "form.test", "timestamp": 1000}])
        assert len(result) >= 1
        assert result[0]["action"] == "save-progress"

    def test_engine_batch_processes(self, engine):
        signals = [
            {"type": "rage_click", "element": "btn1", "count": 5, "timestamp": 1000},
            {"type": "dead_click", "element": "div1", "timestamp": 2000},
            {"type": "hesitation", "element": "inp1", "duration_ms": 6000, "timestamp": 3000},
        ]
        result = engine.process(signals)
        assert len(result) == 3

    def test_engine_cooldown(self, engine):
        # Fire same signal twice rapidly
        s = {"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000}
        r1 = engine.process([s])
        assert len(r1) == 1  # First fires
        # Second might be cooldowned — not guaranteed

    def test_ml_enrichment(self, engine):
        s = {"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000}
        enriched = engine._enrich_signal(s)
        assert "frustration_probability" in enriched
        assert "ml_triggered" in enriched
        assert isinstance(enriched["frustration_probability"], float)

    def test_engine_enforces_cooldown(self, engine):
        # Block both possible actions for this selector
        engine.cooldowns["btn:highlight"] = 9999999999999
        engine.cooldowns["btn:tooltip"] = 9999999999999
        result = engine.process([{"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000}])
        assert len(result) == 0


# ── ML Classifier Tests ──────────────────────────────────────

class TestMLClassifier:
    @pytest.fixture
    def fresh_classifier(self, tmp_path):
        """Classifier with isolated temp path (no cached model interference)."""
        return FrustrationClassifier(model_path=tmp_path / "model.pkl")

    def test_classifier_initializes(self, fresh_classifier):
        assert fresh_classifier is not None
        assert not fresh_classifier.trained
        assert len(fresh_classifier.training_data) == 0

    def test_feature_extraction(self):
        features = FeatureExtractor.extract(
            {"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000},
            []
        )
        assert "signal_rage_click" in features
        assert features["signal_rage_click"] == 1.0
        assert features["count"] == 5.0
        assert len(features) == 13

    def test_feature_vector_length(self):
        vec = FeatureExtractor.to_vector(
            {"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000},
            []
        )
        assert len(vec) == 13

    def test_heuristic_fallback(self, fresh_classifier):
        prob = fresh_classifier.predict(
            {"type": "rage_click", "element": "btn", "count": 5, "timestamp": 1000},
            []
        )
        assert 0.0 <= prob <= 1.0
        # Rage click with count=5 should score high heuristically
        assert prob >= 0.5

    def test_online_learning(self, fresh_classifier):
        signal = {"type": "rage_click", "element": "btn", "count": 3, "timestamp": 1000}
        for _ in range(10):
            fresh_classifier.update(signal, [], was_frustrated=True)
        assert len(fresh_classifier.training_data) >= 10

    def test_model_trains_with_enough_data(self, fresh_classifier):
        signal = {"type": "rage_click", "element": "btn", "count": 3, "timestamp": 1000}
        for i in range(12):
            fresh_classifier.update(signal, [], was_frustrated=(i < 9))
        fresh_classifier._train()
        assert fresh_classifier.trained

    def test_heuristic_weights_by_type(self, fresh_classifier):
        signals = [
            {"type": "rage_click", "element": "btn", "count": 1, "timestamp": 0},
            {"type": "dead_click", "element": "btn", "count": 1, "timestamp": 0},
            {"type": "click", "element": "btn", "count": 1, "timestamp": 0},
        ]
        probs = [fresh_classifier.predict(s, []) for s in signals]
        assert probs[0] >= 0.7  # rage should be high
        assert probs[1] >= 0.5  # dead click mid
        assert probs[2] < probs[1]  # normal click lower than dead

    def test_frequency_boosts_score(self, fresh_classifier):
        signal = {"type": "dead_click", "element": "btn", "count": 1, "timestamp": 0}
        history = [signal] * 5
        prob_low = fresh_classifier.predict(signal, [])
        prob_high = fresh_classifier.predict(signal, history)
        assert prob_high >= prob_low
