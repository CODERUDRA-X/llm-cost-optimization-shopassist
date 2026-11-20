from __future__ import annotations

import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import get_order, get_ticket

from question1.agent import ACTIONS, compact_snapshot, compact_thread, parse_resolution

# LLM ko hamesha sirf itne articles dikhate hain, KB chahe 5 ka ho ya 5 lakh ka.
TOP_K = 3


@dataclass
class TicketResolution:
    action: str
    kb_article: str


_WORD = re.compile(r"[a-z0-9]+")

# Order ke structured facts se articles ke liye "hint" words.
# Ye sirf retrieval ke liye hai (kaun se articles dikhane hain). Final decision
# phir bhi LLM leta hai, article ka text padhke.
_FACT_HINTS = {
    "license_revealed": "digital license",
    "customized": "custom personalized",
    "damage_reported": "damaged damage replacement",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class SupportPipeline:
    """
    Pipeline ke chaaron nodes hain (triage -> retrieve -> compose -> audit),
    par paise sirf `compose` pe lagte hain, wahi ek LLM call hai.

      triage   (local)  ticket ke words + order facts se ek chhota query bana lo
      retrieve (local)  asli articles mein keyword match karke top-k nikaalo.
                        Wo 200k-token wala "expanded help center" mostly filler hai,
                        use model ko kabhi bhejte hi nahi.
      compose  (LLM)    ticket + order + top-k articles -> ek line ka JSON
      audit    (local)  JSON valid hai? action allowed hai? article real hai?
                        Kuch bhi galat ho to ESCALATE_TIER2 (insaan dekh lega).

    Note: audit sirf format check karta hai. Valid par galat decision ye nahi pakad sakta,
    uske liye real system mein sampled LLM audit chahiye hoga.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.ids = article_ids()
        # Articles ek baar load kar lo, har ticket pe dobara disk nahi padhna
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
        # Score = query ke kitne words article mein bhi milte hain. Bahut basic hai
        # (BM25 ka sabse simple version, bina weighting ke). Bade KB ke liye BM25/embeddings lagana padega.
        def score(aid: str) -> int:
            return len(query & (_tokens(aid.replace("-", " ")) | _tokens(self.docs[aid])))

        # Tie ho to article id ke alphabet se, taaki result har baar same aaye
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
        # Pehli line "# article-id" header hoti hai, wo already [id] ke roop mein bhej rahe hain
        return self.docs[aid].split("\n", 2)[-1].strip()

    # -- node 4: audit (no LLM) -------------------------------------------------
    def _audit(self, completion: str, allowed: list[str]) -> TicketResolution:
        # (`allowed` abhi use nahi ho raha, bas retrieved ids ka reference hai.
        #  Chaho to yahan check laga sakte ho ki article top-k ke andar hi ho.)
        parsed = parse_resolution(completion)
        action, article = parsed.action, parsed.kb_article
        if action not in ACTIONS or article not in self.ids:
            # Kuch bhi ulta mila to safe fallback: insaan ko bhej do
            return TicketResolution("ESCALATE_TIER2", "returns-within-30-days")
        return TicketResolution(action, article)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        query = self._triage(compact_thread(ticket), order)
        top = self._retrieve(query)
        draft = self._compose(ticket, order, top)
        return self._audit(draft, top)
