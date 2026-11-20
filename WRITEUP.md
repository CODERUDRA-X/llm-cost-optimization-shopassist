# ShopAssist cost lab — write-up (Q1 + Q2)

Billing: `points = uncached_in x 3 + cached_in x 0.3 + out x 15`. Output tokens cost 5x input, and
the product only consumes `action` + `kb_article`. Every change below follows from those two facts.

## Results (public tickets; each ticket measured on a fresh session)
| | Baseline | Optimized | Target |
|---|---|---|---|
| Q1 T-1001 | ~9,800 | ~1,870 | <= 3,000 |
| Q2 T-1001 (physical fulfilment path) | ~581,000 | ~1,930 | <= 2,500 |
| Q2 T-1003 (escalation path) | ~580,000 | ~1,790 | <= 2,500 |
Holdout T-1091 (subscription) still correct in Q1 (1,788 pts). All graded fields match on every ticket.

## Q1 — one call, 308 in / 63 out
Baseline waste: (1) "think step by step / write full reasoning / quote the article" -> ~490 output
tokens (7,400 of the 9,800 points); (2) a per-run `uuid` and four few-shot examples in the system prompt;
(3) full article prose + indented order JSON + agent boilerplate in the input.
Fix: terse policy sheet (id + rule only), "one-line JSON only", no reasoning/quote/restate instructions,
no uuid (system prompt is byte-identical across tickets), minified snapshot with only the 7 decision-relevant
fields, customer messages only. A small guard maps an invalid action to ESCALATE_TIER2 (no LLM re-check).
Caching: the static prefix is ~190 tokens, below the 1,024 floor, so I do NOT claim cache hits and do not pad to
reach the floor. Rough math: padded prefix = 3,072 first call + ~307/call after, vs 570/call unpadded, so padding only
breaks even after ~11 calls sharing the prefix; for a 4-ticket run it loses, and padding is gaming, not engineering.

## Q2 — four nodes, one LLM call
Baseline: triage sent ~110k tokens, audit ~74k, the retrieve node "printed" 8 articles, and compose got the
triage twice. ~581k points per ticket. The 210k-token "help center" is 5 real articles + routing-note filler.
Fix: keep the triage -> retrieve -> compose -> audit structure but pay for an LLM only where judgment is needed.
- triage (local): build a query from the thread + structured order facts.
- retrieve (local): lexical top-3 over the real articles (never the filler). Scales to a big KB by swapping in BM25/embeddings.
- compose (LLM): ticket + order + 3 short articles -> one-line JSON.
- audit (local): validate JSON schema, enum and that `kb_article` exists; failure -> ESCALATE_TIER2.
Per-ticket cost is now independent of KB size.

## Honest caveats
- Local `LlmClient` is a simulator: its completion is fixed-shape (~60 tokens incl. `customer_reply`) whatever the
  prompt says, so ~950 of ~1,870 points are output I cannot shrink here. A real API honoring "one-line JSON" would
  emit ~20 tokens (~half the points again).
- A local audit catches malformed output, not a wrong-but-valid decision. If accuracy matters more than cost in production, add a
  sampled/low-confidence LLM audit instead of a blanket one.
- Numbers above were measured with an approximate tokenizer (sandbox couldn't download tiktoken's vocab); re-run
  `python question1/run.py`, `python question2/run.py`, `python scripts/breakdown.py q1 T-1001` locally for exact figures.
- Replaced the two `test_baseline_cost_is_high` tests (they assert the baseline is expensive) with target-cost tests.
