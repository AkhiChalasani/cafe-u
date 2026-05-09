# Security Policy

## Reporting a Vulnerability

CAFE-u processes user behavior signals in real-time. We take security seriously.

**Do not open a public issue for security vulnerabilities.**

Instead, email: **chalasaniakhil010@gmail.com**

We will:
1. Acknowledge receipt within 48 hours
2. Assess severity and impact
3. Release a fix and disclose responsibly

## Scope

In-scope:
- The CAFE-u agent (`cafeu.js`) — XSS, data injection, DOM clobbering
- The CAFE-u engine API — authentication bypass, RCE, data leakage
- The rules engine — YAML injection, path traversal
- The ML classifier — pickle deserialization, model poisoning

Out-of-scope:
- Applications using CAFE-u (report to their maintainers)
- Third-party dependencies (report upstream)

## Safe Harbor

We will not pursue legal action for security research conducted responsibly
and in good faith.
