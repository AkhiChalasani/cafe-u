# CAFE-u ⚡

**Open-source AI agent that detects user frustration and fixes your UI in real-time.**

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/AkhiChalasani/cafe-u?style=social)](https://github.com/AkhiChalasani/cafe-u)
[![npm](https://img.shields.io/npm/v/cafeu-agent)](https://www.npmjs.com/package/cafeu-agent)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](docker-compose.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](engine/requirements.txt)
[![DOI](https://img.shields.io/badge/Paper-CAFE--u-8A2BE2)](#research)

---

## 🔥 Demo

**[Try the live demo →](https://akhichalasani.github.io/cafe-u/)**

Rage-click a button. The UI adapts. No reload. No developer needed.

What you'll see:
1. Click a stat card rapidly 3x → it glows orange + a tooltip appears
2. Click a dead "View Report" button → it becomes clickable
3. Pause on a form field for 5s → an inline hint shows up

## What is CAFE-u?

Every web app has friction points. A button that looks clickable but isn't. A form that confuses users. A CTA they scroll past.

Normally: deploy analytics → see the problem → write a fix → deploy again. **Days.** Users churn.

CAFE-u: **detects frustration → decides the fix → adapts the UI → all in under 500ms.**

```
User rage-clicks → Agent detects signal → ML predicts frustration (0-1)
    → RAG retrieves similar past cases → LLM decides best adaptation
    → Agent applies DOM fix → User continues, frustration resolved
```

## Quick Start

### 1. Web (add to any page)

```html
<script src="https://cdn.jsdelivr.net/npm/cafeu-agent@0.1/dist/cafeu.min.js"
        data-key="my-app">
</script>
```

That's it. The agent runs locally in the browser. Without an engine server, it uses built-in rule-based adaptations.

### 2. Full AI Engine (Docker)

```bash
docker run -p 8080:8080 ghcr.io/akhichalasani/cafeu-engine
```

Then update your script tag:

```html
<script src="cafeu.min.js" data-key="my-app" data-ws="ws://localhost:8080/ws"></script>
```

With the engine, CAFE-u uses ML + RAG + LLM to make smarter adaptation decisions.

### 3. npm

```bash
npm install cafeu-agent
```

## Features

| Signal | What it detects | Default adaptation |
|--------|----------------|-------------------|
| Rage click | 3+ clicks on same element in 1.2s | Pulsing highlight + floating tooltip |
| Dead click | Click on non-interactive element | Make it look clickable + hover effect |
| Hesitation | >5s pause on a form field | Inline hint + focus ring |
| Scroll bounce | User scrolls past CTA, then back up | Sticky CTA at viewport bottom |
| Form abandon | User leaves mid-form | "Progress saved" toast |
| Rapid tap (mobile) | 3+ taps in same area in 2s | Highlight + tooltip |

See all [built-in adaptation rules](engine/rules/definitions/).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Your Web App                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  CAFE-u Agent (10KB JS) → detects signals          │  │
│  │  ← applies DOM adaptations (highlight, tooltip…)   │  │
│  └──────────────────────┬─────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────┘
                          │
              Signals     │     Adaptations
              ▼           │     ▲
┌─────────────────────────┴────────────────────────────────┐
│                  CAFE-u Engine (optional)                  │
│                                                           │
│  1. Feature Extractor → 13 numerical features              │
│  2. ML Classifier → frustration probability (0-1)          │
│  3. RAG Store → top-3 similar past cases (FAISS)          │
│  4. AI Agent (LLM) → decides tool + parameters            │
│     ↕ Fallback: rule-based heuristic                       │
│  5. Adaptation returned to browser agent                   │
│  6. Feedback loop → stores in RAG for future               │
└──────────────────────────────────────────────────────────┘
```

## One-Click Deploy

```bash
git clone https://github.com/AkhiChalasani/cafe-u.git
cd cafe-u
docker compose up
```

Open http://localhost:8080 — dashboard + demo ready.

## ML + RAG + Agent Pipeline

CAFE-u uses a three-stage decision pipeline:

**1. ML Classifier** (scikit-learn LogisticRegression)
- 13 features extracted from each signal
- Predicts frustration probability 0-1
- Trains online from user feedback

**2. RAG Store** (FAISS vector search)
- Stores past signal→adaptation pairs
- Retrieves top-3 most similar cases
- Falls back to keyword search if FAISS unavailable

**3. AI Agent** (LLM with tool calling)
- 7 adaptation tools: highlight, tooltip, make-clickable, sticky-cta, inline-hint, save-progress, dim-section
- Uses DeepSeek V4 Flash (or any OpenAI-compatible API)
- Falls back to rule-based heuristic when LLM unavailable
- No API key needed for rule-based mode

## Project Structure

```
cafe-u/
├── agent/                         # Browser SDK (10KB JS)
│   ├── src/cafeu.js              # Source
│   ├── dist/cafeu.min.js         # Minified production build
│   └── package.json              # npm package
├── engine/                        # Python backend
│   ├── api/server.py             # FastAPI server
│   ├── rules/
│   │   ├── agent.py              # AI Agent (LLM + RAG)
│   │   ├── classifier.py         # ML frustration classifier
│   │   ├── rag_store.py          # FAISS vector memory
│   │   ├── engine.py             # Rules engine
│   │   └── definitions/          # YAML adaptation rules
│   └── tests/test_engine.py      # 22 tests
├── docs/                          # GitHub Pages demo
│   └── index.html                # Live interactive demo
├── examples/                      # Demo files
├── .github/workflows/ci.yml      # CI/CD (8 jobs)
├── Dockerfile                     # Production image
└── docker-compose.yml             # One-command deploy
```

## Research

CAFE-u is based on the paper:

> **"An AI Comprehensive Framework for Enhancing Usability integrating in Human–Computer Interaction Systems"**
> — Akhil Chowdary Chalasani, M.S. Computer Science in AI/ML, Methodist University

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

**Quick ways to contribute:**
- Add a new adaptation rule (just a YAML file)
- Improve the ML classifier
- Write a framework integration (React, Vue, Angular)
- Report a bug or suggest a feature

## License

MIT — free for personal and commercial use.

---

_Because your app should get better the more people use it._

**[⭐ Star on GitHub](https://github.com/AkhiChalasani/cafe-u) · [🐛 Report Bug](https://github.com/AkhiChalasani/cafe-u/issues) · [🚀 Live Demo](https://akhichalasani.github.io/cafe-u/)**
