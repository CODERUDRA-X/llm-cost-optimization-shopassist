# ShopAssist cost optimization

Shreyansh Srivastava · CODERUDRA-X

Each public ticket stays within its individual limit. On T-1001, Q1 fell from 8,850 to 1,257 points (85.8%) and Q2 from 73,416 to 1,248 (98.3%). The original baseline numbers are from the recorded Windows run against commit d9989e2. The optimized counts below use the supplied simulator and cl100k_base tokenizer. The four-ticket averages exclude T-1091.

| Ticket | Expected action | KB article | Q1 input / output | Q1 points | Q2 input / output | Q2 points |
|---|---|---|---:|---:|---:|---:|
| T-1001 | REFUND | returns-within-30-days | 189 / 46 | 1,257 | 186 / 46 | 1,248 |
| T-1002 | SEND_ARTICLE | digital-license-no-refund | 193 / 39 | 1,164 | 190 / 39 | 1,155 |
| T-1003 | ESCALATE_TIER2 | custom-items-no-replace | 181 / 39 | 1,128 | 171 / 39 | 1,098 |
| T-1004 | REPLACE | damage-replacement-7-days | 183 / 36 | 1,089 | 180 / 36 | 1,080 |
| **Public total** | | | | **4,638** | | **4,581** |
| **Public mean** | | | | **1,159.50** | | **1,145.25** |
| T-1091, supplied holdout | SEND_ARTICLE | subscription-cooling-off | 187 / 38 | 1,131 | 184 / 38 | 1,122 |

## Where the savings came from

Q1 makes one model call. Its static prompt carries five short policies with their article IDs. The user message contains customer messages and a minified snapshot of decision-relevant fields. The baseline carried a random run ID, four examples, the full KB and instructions that triggered a verbose response. The simulator still includes a customer reply in its shorter output. For T-1001, 189 input tokens at 3 points and 46 output tokens at 15 give 1,257 points. Cached input is zero throughout. The prompts are below the 1,024-token prefix minimum.

Q2 performs local triage, local retrieval, one compose call and local audit. Triage produces query terms using the customer message and structured order facts. Retrieval ranks the small KB by lexical overlap and sends at most two article bodies. For known standard physical merchandise, the search skips known incompatible policy families. Other articles remain eligible. Eligibility filtering uses product metadata. It does not assign the final action or compare delivery age against the 7-day threshold.

For T-1004, compose costs 180 × 3 + 36 × 15 = 1,080. For the T-1003 escalation, it costs 171 × 3 + 39 × 15 = 1,098. Triage, retrieve and audit each use zero model calls. The earlier top-three version averaged 1,242.75 in Q2. Naively shrinking the shortlist let unrelated text displace the returns article in a constructed damaged stock-item case. The same opened stock item now retrieves the necessary evidence on both day 7 (replace) and day 8 (escalate under the simulator rules).

Audit checks the parsed action and membership of the cited ID in the retrieved set. A parse failure, missing field, unknown action or unsupported citation routes to manual review without a second model call. On this path the returned article is the first retrieved article for reviewer context. It is not a verified reason to approve a refund or replacement. The supplied KB is nonempty.

## Evaluation and limits

The public tests check each ticket against the budget, rather than only checking the mean. The review tests check retrieval coverage independently of the final decision, exactly one compose call for each supplied case, malformed or unsupported drafts, and the 7/8-day boundary. Five of five supplied cases retrieve the expected article within the top two. One extra parameterized scenario probes the damage deadline.

The simulator derives its decision from order facts, so these passes do not establish hosted-model accuracy or semantic grounding. A wrong action paired with an allowed citation can pass local validation. Q1's manually maintained policy sheet can drift and its parser can raise on malformed output. The simple lexical search scans its in-memory articles for each query. Neither large-KB latency nor real-model accuracy was benchmarked. A deployment evaluation should measure retrieval recall, action-and-article exact match, unsupported citations, cost and latency on independent labeled tickets.

## Reproduce from the repository root

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python question1/run.py
python question2/run.py
python scripts/breakdown.py q1 T-1001 T-1091
python scripts/breakdown.py q2 T-1004 T-1003
```
