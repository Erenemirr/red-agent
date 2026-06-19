"""Orchestrator node — döngünün beyni.

Her turdan sonra üç karardan birini verir (CLAUDE.md):
  - `report`    : `current_round >= max_rounds` (kesin terminal koşul).
  - `interrupt` : yeterli yeni kritik açık birikti → human-in-the-loop, "devam
                  edeyim mi?" diye sorulur. Aynı eşikte tekrar tekrar sormamak
                  için her +threshold kritikte yalnızca bir kez tetiklenir.
  - `continue`  : yeni bir tur için Attacker'a dön.

Karar `next_action` alanına yazılır; LangGraph conditional edge bunu okuyup
ilgili node'a yönlendirir (`route` yardımcı fonksiyonu).

Node saf bir fonksiyondur — LangGraph'a bağımlı değildir, izole test edilebilir.
"""

from __future__ import annotations

import logging

from config.settings import settings
from agents.state import RedTeamState

logger = logging.getLogger(__name__)


def decide_next_action(state: RedTeamState) -> dict:
    """Döngü kararını hesapla ve state güncellemesi döndür."""
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", settings.max_rounds)
    critical = state.get("critical_count", 0)
    threshold = settings.critical_findings_for_interrupt
    last_interrupt_at = state.get("last_interrupt_at_count", 0)

    # 1) Kesin terminal: tur limiti aşıldıysa rapora git.
    if current_round >= max_rounds:
        logger.info(
            "orchestrator: max_rounds (%d) aşıldı -> report", max_rounds
        )
        return {"next_action": "report", "human_interrupt": False}

    # 2) Human-in-the-loop: son interrupt'tan bu yana +threshold kritik biriktiyse sor.
    if critical - last_interrupt_at >= threshold:
        logger.info(
            "orchestrator: %d kritik açık (eşik %d) -> human interrupt",
            critical, threshold,
        )
        return {
            "next_action": "interrupt",
            "human_interrupt": True,
            "last_interrupt_at_count": critical,
        }

    # 3) Aksi halde döngüye devam.
    logger.info(
        "orchestrator: tur %d/%d, %d kritik -> continue",
        current_round, max_rounds, critical,
    )
    return {"next_action": "continue", "human_interrupt": False}


def orchestrator_node(state: RedTeamState) -> dict:
    """LangGraph node imzası — kararı hesaplayıp state'e yazar."""
    return decide_next_action(state)


def route(state: RedTeamState) -> str:
    """Conditional edge yönlendirmesi: next_action -> hedef node adı.

    LangGraph `add_conditional_edges` ile kullanılır. interrupt durumunda
    graph akışı human onayı için duraklatılır (main.py'de checkpointer +
    interrupt mekanizması ile ele alınır); döndürülen anahtar oraya yönlendirir.
    """
    action = state.get("next_action", "continue")
    return {
        "continue": "attacker",
        "report": "reporter",
        "interrupt": "human",
    }.get(action, "attacker")
