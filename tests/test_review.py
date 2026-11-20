import pytest

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import compact_thread
from question2.agent import SupportPipeline
from shared.tools import get_order, get_ticket


@pytest.mark.parametrize("ticket_id", list(TICKET_EXPECTATIONS))
def test_policy_is_retrieved_and_compose_is_only_call(ticket_id):
    pipe = SupportPipeline()
    ticket = get_ticket(ticket_id)
    order = get_order(ticket["order_id"])
    candidates = pipe._retrieve(pipe._triage(compact_thread(ticket), order), order)
    expected = TICKET_EXPECTATIONS[ticket_id]
    assert expected.kb_article in candidates
    got = pipe.handle_ticket(ticket_id)
    assert (got.action, got.kb_article) == (expected.action, expected.kb_article)
    assert [call.label for call in pipe.session.calls] == ["compose"]


@pytest.mark.parametrize("draft", [
    "broken JSON",
    '{"action":"REFUND"}',
    '{"action":"PAY","kb_article":"damage-replacement-7-days"}',
    '{"action":"REFUND","kb_article":"returns-within-30-days"}',
    '{"action":"REFUND","kb_article":"unknown"}',
])
def test_bad_draft_routes_to_manual_review(draft):
    pipe = SupportPipeline()
    candidates = ["damage-replacement-7-days", "custom-items-no-replace"]
    got = pipe._audit(draft, candidates)
    assert (got.action, got.kb_article) == ("ESCALATE_TIER2", candidates[0])
    assert not pipe.session.calls


@pytest.mark.parametrize("days,action,article", [
    (7, "REPLACE", "damage-replacement-7-days"),
    (8, "ESCALATE_TIER2", "returns-within-30-days"),
])
def test_stock_damage_window(monkeypatch, days, action, article):
    order = {"product_type": "physical", "customized": False, "opened": True,
             "damage_reported": True, "delivered_days_ago": days}
    ticket = {"order_id": "boundary", "messages": [
        {"from": "customer", "text": "My warehouse mug cracked, not custom. Replacement?"}
    ]}
    monkeypatch.setattr("question2.agent.get_ticket", lambda _: ticket)
    monkeypatch.setattr("question2.agent.get_order", lambda _: order)
    pipe = SupportPipeline()
    candidates = pipe._retrieve(pipe._triage(compact_thread(ticket), order), order)
    assert article in candidates
    got = pipe.handle_ticket("boundary")
    assert (got.action, got.kb_article) == (action, article)
