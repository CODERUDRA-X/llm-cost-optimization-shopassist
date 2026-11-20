"""Per-call points breakdown.  usage: python scripts/breakdown.py q1|q2 T-1001 [T-1003 ...]"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.billing import CACHED_INPUT_PRICE, INPUT_PRICE, OUTPUT_PRICE, CostSession  # noqa: E402


def main(track: str, ids: list[str]) -> None:
    if track == "q1":
        from question1.agent import SupportCopilot as Agent
    else:
        from question2.agent import SupportPipeline as Agent
    for tid in ids:
        s = CostSession()
        res = Agent(s).handle_ticket(tid)
        print(f"\n{tid} -> {res.action} / {res.kb_article}")
        print(f"{'call':<14}{'in_tok':>7}{'cached':>8}{'out_tok':>9}{'points':>8}")
        for c in s.calls:
            cached = min(c.cached_prefix_tokens, c.input_tokens)
            print(f"{c.label:<14}{c.input_tokens:>7}{cached:>8}{c.output_tokens:>9}{c.points():>8}"
                  f"   = {c.input_tokens-cached}x{INPUT_PRICE} + {cached}x{CACHED_INPUT_PRICE} + {c.output_tokens}x{OUTPUT_PRICE}")
        print(f"{'TOTAL':<14}{'':>24}{s.total_points():>8}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
