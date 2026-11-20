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


# Keep policy thresholds explicit when compressing the five source articles.
_SYSTEM = """Return JSON {"action":...,"kb_article":...}. Apply the matching policy:
returns-within-30-days: unopened physical item within 30 days of delivery: REFUND, otherwise ESCALATE_TIER2.
damage-replacement-7-days: damaged standard item within 7 days of delivery: REPLACE.
custom-items-no-replace: damaged customized item: ESCALATE_TIER2.
digital-license-no-refund: revealed digital license: SEND_ARTICLE.
subscription-cooling-off: annual subscription cancelled within 30 days: REFUND, otherwise SEND_ARTICLE."""

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
