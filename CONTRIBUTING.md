# Contributing to CAFE-u

First off — thank you. We want CAFE-u to be the standard for adaptive UI, and every contribution moves that forward.

## Quick Start

```bash
# Clone
git clone https://github.com/AkhiChalasani/cafe-u.git
cd cafe-u

# Agent (JS)
cd agent
npm install     # Install deps
npm run build   # Build dist/cafeu.min.js

# Engine (Python)
cd ../engine
pip install -r requirements.txt
pip install scikit-learn numpy  # For ML classifier
python -m pytest tests/ -v      # Run tests
```

## How to Contribute

### 1. Pick an issue

Check [issues](https://github.com/AkhiChalasani/cafe-u/issues) for:

- `good first issue` — small, well-scoped, mentor available
- `help wanted` — we specifically need help here
- `adaptation-rule` — propose a new frustration → UI fix mapping
- `bug` — something's broken

### 2. Create a branch

```bash
git checkout -b feat/your-feature-name
```

**Branch naming:**
- `feat/` — new features (new adaptation, new SDK)
- `fix/` — bug fixes
- `docs/` — documentation
- `rule/` — new YAML adaptation rule
- `perf/` — performance

### 3. Make your change

**New adaptation rule** (easiest way to contribute):
1. Create `engine/rules/definitions/your-rule.yaml`
2. Add the corresponding action in `agent/src/cafeu.js` → `applyAdaptation()`
3. Test with the demo page

**Agent changes:**
Edit `agent/src/cafeu.js`, then `npm run build` to minify.

**Engine changes:**
Edit files in `engine/`, then restart the server to test.

**ML classifier improvements:**
Edit `engine/rules/classifier.py`. Add new features to `FeatureExtractor`.

### 4. Commit

```bash
git commit -m "feat(agent): detect long-press on mobile elements"
```

Use [conventional commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `rule:` new adaptation rule
- `refactor:` code change with no behavior change
- `test:` adding tests

### 5. Push and PR

```bash
git push origin feat/your-feature-name
```

Open a PR against `main`. Link the issue. Describe what you changed.

## Architecture Notes

- **Agent is stateless** — all decisions come from the engine
- **Engine is stateless across sessions** — signals in, adaptations out
- **YAML rules are user-editable** — design for customization
- **ML is optional** — gracefully falls back to heuristics
- **No PII ever** — signals contain element paths, not user data

## Testing

```bash
# Engine
cd engine
pytest tests/ -v --tb=short

# Manual: start server
python -c "import sys; sys.path.insert(0,'.'); from api.server import app; import uvicorn; uvicorn.run(app, port=8080)"

# Test with demo
open ../examples/demo.html
```

## Getting Help

- Open a GitHub Discussion
- Email: chalasaniakhil010@gmail.com

---

_This project implements the CAFE-u research framework. If you're contributing based on the paper, mention it in your PR._
