from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.llm import ChatMessage, LlmClient
from shared.tools import get_order, get_ticket

ACTIONS = ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2")


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# Cost design (see README.md in this folder for the full write-up):
#  * ONE call per ticket, no retries, no second "audit" pass.
#  * Static, compact policy sheet (article id + the rule, nothing else): ~190 tokens
#    instead of full article prose + 4 few-shot examples.
#  * No per-run uuid / trace ids -> the system prompt is byte-identical across tickets
#    (cache-friendly if a provider caches it; it is below the 1024-token floor, so we
#    do not pretend it is cached and we do not pad it to reach the floor).
#  * No "think step by step / quote the article / restate" instructions: output is
#    billed at 5x input, and the product only consumes `action` + `kb_article`.
#  * Order snapshot sent as minified JSON with only decision-relevant fields.
_SYSTEM = (
    "ShopAssist Tier-1 router. Reply with ONLY one-line JSON "
    '{"action":..,"kb_article":..}. No prose.\n'
    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.\n"
    "Articles:\n"
    "returns-within-30-days: unopened physical, <=30d from delivery -> REFUND; else ESCALATE_TIER2\n"
    "damage-replacement-7-days: damaged catalog (not customized) item, <=7d -> REPLACE\n"
    "custom-items-no-replace: customized + damaged -> ESCALATE_TIER2\n"
    "digital-license-no-refund: license revealed -> SEND_ARTICLE\n"
    "subscription-cooling-off: subscription <=30d -> REFUND; >30d -> SEND_ARTICLE"
)

# Only the fields the policy above actually depends on.
_ORDER_FIELDS = (
    "product_type",
    "delivered_days_ago",
    "subscription_age_days",
    "opened",
    "license_revealed",
    "customized",
    "damage_reported",
)


def compact_snapshot(order: dict) -> str:
    slim = {k: order[k] for k in _ORDER_FIELDS if k in order}
    return json.dumps(slim, separators=(",", ":"))


def compact_thread(ticket: dict) -> str:
    # Customer messages only: agent boilerplate ("checking your order now") carries no signal.
    return " ".join(m["text"] for m in ticket["messages"] if m["from"] == "customer")


class SupportCopilot:
    """ShopAssist Tier-1 copilot (cost-optimized, single call)."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        user = (
            f"{compact_thread(ticket)}\n"
            f"ORDER_SNAPSHOT:{compact_snapshot(order)}END_SNAPSHOT"
        )
        return [ChatMessage("system", _SYSTEM), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        completion = self.llm.chat("handle_ticket", self.build_messages(ticket_id))
        res = parse_resolution(completion)
        if res.action not in ACTIONS:  # cheap guard instead of an LLM audit pass
            res.action = "ESCALATE_TIER2"
        return res
