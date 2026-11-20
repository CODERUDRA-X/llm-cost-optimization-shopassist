from __future__ import annotations

import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import get_order, get_ticket

from question1.agent import ACTIONS, compact_snapshot, compact_thread, parse_resolution

TOP_K = 3


@dataclass
class TicketResolution:
    action: str
    kb_article: str


_WORD = re.compile(r"[a-z0-9]+")

# Structured order facts -> article hints. Pure retrieval signal (which articles are
# worth showing the model); the model still makes the decision from article text.
_FACT_HINTS = {
    "license_revealed": "digital license",
    "customized": "custom personalized",
    "damage_reported": "damaged damage replacement",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class SupportPipeline:
    """
    triage -> retrieve -> compose -> audit, with only `compose` paying for an LLM.

      triage   (local)  build a compact query from thread + order facts
      retrieve (local)  lexical top-k over the REAL articles; the 200k-token "expanded"
                        help center is routing-note filler and is never sent to a model
      compose  (LLM)    one short call: ticket + order + top-k articles -> JSON
      audit    (local)  schema/enum/article-id validation; safe fallback = ESCALATE_TIER2
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.ids = article_ids()
        self.docs = {aid: load_article(aid) for aid in self.ids}

    # -- node 1: triage (no LLM) ------------------------------------------------
    def _triage(self, thread: str, order: dict) -> set[str]:
        q = _tokens(thread) | _tokens(str(order.get("product_type", "")))
        if order.get("product_type") == "subscription":
            q |= _tokens("subscription cancel cooling")
        for flag, hint in _FACT_HINTS.items():
            if order.get(flag):
                q |= _tokens(hint)
        if order.get("product_type") == "physical":
            q |= _tokens("physical returns refund")
        return q

    # -- node 2: retrieve (no LLM) ----------------------------------------------
    def _retrieve(self, query: set[str]) -> list[str]:
        def score(aid: str) -> int:
            return len(query & (_tokens(aid.replace("-", " ")) | _tokens(self.docs[aid])))

        ranked = sorted(self.ids, key=lambda a: (-score(a), a))
        return ranked[:TOP_K]

    # -- node 3: compose (the only LLM call) ------------------------------------
    def _compose(self, ticket: dict, order: dict, article_ids_: list[str]) -> str:
        articles = "\n".join(f"[{a}] {self._body(a)}" for a in article_ids_)
        system = (
            "ShopAssist router. Reply ONLY one-line JSON "
            '{"action":..,"kb_article":..}. action in REFUND|REPLACE|SEND_ARTICLE|ESCALATE_TIER2; '
            "kb_article must be one of the ids below.\n" + articles
        )
        user = (
            f"{compact_thread(ticket)}\n"
            f"ORDER_SNAPSHOT:{compact_snapshot(order)}END_SNAPSHOT"
        )
        return self.llm.chat("compose", [ChatMessage("system", system), ChatMessage("user", user)])

    def _body(self, aid: str) -> str:
        return self.docs[aid].split("\n", 2)[-1].strip()  # drop the "# id" header line

    # -- node 4: audit (no LLM) -------------------------------------------------
    def _audit(self, completion: str, allowed: list[str]) -> TicketResolution:
        parsed = parse_resolution(completion)
        action, article = parsed.action, parsed.kb_article
        if action not in ACTIONS or article not in self.ids:
            return TicketResolution("ESCALATE_TIER2", "returns-within-30-days")
        return TicketResolution(action, article)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        query = self._triage(compact_thread(ticket), order)
        top = self._retrieve(query)
        draft = self._compose(ticket, order, top)
        return self._audit(draft, top)
