# CAFE-u: The HN Launch Post

## Title (the most important part — multiple options)

**Option A (technical):**
> Show HN: CAFE-u – Open-source AI agent that detects UI frustration and fixes it in real-time

**Option B (problem-first):**
> Show HN: Your users rage-click buttons. This open-source AI agent fixes the UI automatically.

**Option C (comparison):**
> Show HN: CAFE-u – Self-hosted AI that does what Hotjar shows you (fix the UI automatically)

---

## First comment (pinned — this is what people read):

"Hi HN, I built CAFE-u based on my research paper on AI-driven usability adaptation.

The problem: every web app has friction points — dead buttons, confusing forms, missed CTAs. Normally you'd use analytics to see the problem, write a fix, deploy it. That takes days. Users churn.

CAFE-u detects frustration signals (rage clicks, dead clicks, hesitation, scroll bounce) and adapts the UI in under 500ms. No human intervention. No redeploy.

**How it works:**
1. A 10KB JS agent detects frustration signals in the browser
2. An ML classifier predicts frustration probability (0-1)
3. A RAG store (FAISS) retrieves similar past cases with successful adaptations
4. An LLM agent (DeepSeek V4 Flash) decides which adaptation to apply
5. The agent applies the DOM fix — highlight, tooltip, sticky CTA, etc.
6. If the LLM is unavailable, it falls back to rule-based heuristics

**Quick start:**
```html
<script src="https://cdn.jsdelivr.net/npm/cafeu-agent@0.1/dist/cafeu.min.js"
        data-key="my-app"></script>
```

For the full ML pipeline: `docker run -p 8080:8080 ghcr.io/akhichalasani/cafeu-engine`

**What's inside:**
- ML classifier with online learning (scikit-learn)
- RAG store with FAISS vector search
- AI Agent with 7 adaptation tools (LLM + rule fallback)
- 10KB browser agent, zero dependencies
- React Native SDK
- 22 passing tests
- MIT license

**Try the live demo:** https://akhichalasani.github.io/cafe-u/
**GitHub:** https://github.com/AkhiChalasani/cafe-u

Would love your feedback — especially on:
- What adaptations would be most useful for your app?
- What signal types should we add next?
- Any edge cases we're missing?"

---

## Why this will work on HN:

1. **The demo is compelling** — people can try it right in the browser. No signup. No install.
2. **The problem is universal** — every developer has shipped a feature that users struggled with.
3. **The tech stack is HN-friendly** — FastAPI, scikit-learn, FAISS, LLM agents. All the buzzwords, but with real code behind them.
4. **It has academic credibility** — the paper gives it weight beyond "another weekend project."
5. **"Self-hosted alternative to X"** is a proven HN formula (X = Hotjar/FullStory here).

## Timing:

- **Post on Tuesday or Wednesday at ~9 AM Pacific** — peak HN traffic
- **Have the live demo ready** (GitHub Pages already deployed)
- **Reply to every comment within the first 3 hours** — engagement drives visibility
- **Update the README with feedback received** within 24 hours

## If it hits the front page:

- Expect 10,000-30,000 unique visitors
- 100-400 GitHub stars in the first day
- 5-20 issues/feature requests
- 1-3 pull requests from first-time contributors
- Server load: the Docker image will be pulled 50-200 times
