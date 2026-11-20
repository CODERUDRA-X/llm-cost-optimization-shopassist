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
    # Model kabhi kabhi JSON ke aage-peeche kuch text bhi laga deta hai,
    # isliye pure text ko parse nahi karte, pehle { ... } wala hissa nikaalte hain.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        # customer_reply ab ek-line JSON mein aata hi nahi, isliye .get() with default
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# How I kept this cheap (my notes, so I don't forget why it looks like this):
#
# - Output tokens cost 5x of input, so the biggest win is making the model say less.
#   The product only reads `action` and `kb_article`. No essays, no "think step by step".
# - One call per ticket. No retries, no second "audit" call.
# - Removed the per-run uuid from the system prompt. It changed every run, so the
#   prompt was never the same twice, and caching could never kick in anyway.
# - Prompt is only ~190 tokens, below the 1024-token minimum for caching. So I'm not
#   claiming any cache hit, and I'm not padding it just to reach 1024 (that costs more).
# - Policy sheet = one line per article, instead of full article text + few-shot examples.
#
# Heads up: this sheet is my hand-written summary of the KB articles. If the KB rules
# change, this needs to be updated too, otherwise they'll go out of sync.
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

# Sirf wahi fields jo upar ke rules mein use hote hain. Baaki (sku, title, etc.)
# bhejne ka matlab sirf extra tokens, decision pe koi farak nahi.
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
    # separators se spaces hat jaate hain, har space/newline bhi token hi hota hai
    return json.dumps(slim, separators=(",", ":"))


def compact_thread(ticket: dict) -> str:
    # Sirf customer ke messages. Agent wala "checking your order now" type text
    # decision mein kuch add nahi karta, bas tokens khata hai.
    return " ".join(m["text"] for m in ticket["messages"] if m["from"] == "customer")


class SupportCopilot:
    """ShopAssist Tier-1 copilot. One LLM call per ticket."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        # ORDER_SNAPSHOT...END_SNAPSHOT markers rakhe hain kyunki local simulator
        # inhi se order facts nikaalta hai. Hata diye to simulator kuch padh nahi payega.
        user = (
            f"{compact_thread(ticket)}\n"
            f"ORDER_SNAPSHOT:{compact_snapshot(order)}END_SNAPSHOT"
        )
        return [ChatMessage("system", _SYSTEM), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        completion = self.llm.chat("handle_ticket", self.build_messages(ticket_id))
        res = parse_resolution(completion)
        # Agar model ulta-seedha action bol de, to safe side pe insaan (Tier 2) ko bhej do.
        # Iske liye alag LLM call lagane ki zaroorat nahi, ek simple check kaafi hai.
        if res.action not in ACTIONS:
            res.action = "ESCALATE_TIER2"
        return res
