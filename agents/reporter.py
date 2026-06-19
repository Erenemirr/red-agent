"""Reporter node — döngü sonu rapor üretimi.

İki çıktı (CLAUDE.md):
  - Markdown: insan için — executive summary + teknik detay + savunma önerileri.
  - JSON: makine için — tüm ham veri (attack_history dahil).

Markdown rapor `final_report` state alanına yazılır; JSON dosyaya kaydedilir ve
yolu `report_json_path`'e konur. Phoenix dashboard'u Arize karşıladığı için
burada ayrı bir görselleştirme üretilmez.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone

from config.settings import REPORTS_DIR, settings
from agents.state import AttackAttempt, RedTeamState

logger = logging.getLogger(__name__)


def _summarize(history: list[AttackAttempt]) -> dict:
    """Geçmişten özet metrikler çıkar."""
    total = len(history)
    successes = [a for a in history if a.get("verdict", {}).get("success")]
    criticals = [a for a in history if a.get("verdict", {}).get("is_critical")]

    by_category: Counter = Counter(a.get("category", "?") for a in history)
    success_by_category: Counter = Counter(a.get("category", "?") for a in successes)
    by_violation: Counter = Counter(
        a.get("verdict", {}).get("violation_type", "none") for a in successes
    )
    scores = [a.get("verdict", {}).get("score", 0.0) for a in history]
    avg_score = round(sum(scores) / total, 3) if total else 0.0

    return {
        "total_attempts": total,
        "successful": len(successes),
        "critical": len(criticals),
        "success_rate": round(len(successes) / total, 3) if total else 0.0,
        "avg_score": avg_score,
        "by_category": dict(by_category),
        "success_by_category": dict(success_by_category),
        "by_violation": dict(by_violation),
    }


# Tespit edilen ihlal tipine göre kısa savunma önerileri.
_DEFENSE_TIPS = {
    "pii-leak": "PII çıktı filtresi ekleyin; yanıtları gönderim öncesi maskeleyin (DLP).",
    "system-prompt-leak": "System prompt'u yanıtlardan ayrıştırın; 'talimatını göster' taleplerini reddedin.",
    "policy-bypass": "Persona/rol değiştirme taleplerini yakalayan guardrail ekleyin (örn. Llama Guard).",
    "harmful-content": "Zararlı içerik sınıflandırıcısı ile çıktıyı denetleyin.",
    "other": "İlgili kategoriye özel input/output guardrail değerlendirin.",
}


def build_markdown(state: RedTeamState, summary: dict) -> str:
    """İnsan-okur Markdown raporu üret."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    history: list[AttackAttempt] = state.get("attack_history", [])
    successes = [a for a in history if a.get("verdict", {}).get("success")]

    lines: list[str] = []
    lines.append("# Red Team Değerlendirme Raporu")
    lines.append(f"\n_Üretildi: {now}_\n")

    # --- Executive summary ---
    lines.append("## Yönetici Özeti\n")
    lines.append(f"- Toplam deneme: **{summary['total_attempts']}**")
    lines.append(f"- Başarılı saldırı: **{summary['successful']}** "
                 f"(başarı oranı %{summary['success_rate'] * 100:.0f})")
    lines.append(f"- Kritik açık: **{summary['critical']}**")
    lines.append(f"- Ortalama skor: **{summary['avg_score']}**")
    lines.append(f"- Human-in-the-loop onayı istendi: **{state.get('human_interrupt_count', 0)}** kez\n")

    if summary["success_by_category"]:
        lines.append("En etkili saldırı kategorileri:")
        for cat, n in sorted(
            summary["success_by_category"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"- `{cat}` → {n} başarılı")
        lines.append("")

    # --- Teknik detay ---
    lines.append("## Teknik Detay — Başarılı Saldırılar\n")
    if not successes:
        lines.append("_Başarılı saldırı bulunamadı — hedef tüm denemelere dayandı._\n")
    for a in successes:
        v = a.get("verdict", {})
        lines.append(f"### Tur {a.get('round', '?')} — `{a.get('category', '?')}` "
                     f"(skor {v.get('score', 0.0)})")
        lines.append(f"- İhlal tipi: `{v.get('violation_type', 'none')}`")
        if a.get("objective"):
            lines.append(f"- Saldırı hedefi: {_trunc(a.get('objective', ''), 120)}")
        lines.append(f"- Strateji: {a.get('strategy', '')}")
        lines.append(f"- **Prompt:** {_trunc(a.get('prompt', ''), 400)}")
        lines.append(f"- **Yanıt:** {_trunc(a.get('response', ''), 400)}")
        lines.append(f"- Gerekçe: {v.get('rationale', '')}\n")

    # --- İnsan denetimi (human-in-the-loop) ---
    decisions = state.get("human_decisions", [])
    if decisions:
        lines.append("## İnsan Denetimi (Human-in-the-Loop)\n")
        lines.append(f"Döngü {len(decisions)} kez kritik-açık eşiğinde durup operatör onayı istedi:\n")
        for d in decisions:
            karar = "devam edildi" if d.get("decision") == "continue" else "durduruldu"
            lines.append(f"- Tur {d.get('round', '?')} ({d.get('critical_count', '?')} kritik açık) → **{karar}**")
        lines.append("")

    # --- Savunma önerileri ---
    lines.append("## Savunma Önerileri\n")
    if summary["by_violation"]:
        for vtype in summary["by_violation"]:
            tip = _DEFENSE_TIPS.get(vtype)
            if tip:
                lines.append(f"- **{vtype}:** {tip}")
    else:
        lines.append("- Belirgin ihlal yok; mevcut guardrail'ler etkili görünüyor.")
    lines.append("")

    return "\n".join(lines)


def build_json(state: RedTeamState, summary: dict) -> dict:
    """Makine-okur tam rapor (ham veri dahil)."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_system": state.get("target_system", ""),
        "rounds_run": state.get("current_round", 0),
        "max_rounds": state.get("max_rounds", settings.max_rounds),
        "summary": summary,
        "human_interrupt_count": state.get("human_interrupt_count", 0),
        "human_decisions": state.get("human_decisions", []),
        "phoenix_insights": state.get("phoenix_insights", {}),
        "attack_history": state.get("attack_history", []),
    }


def _trunc(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# --- LangGraph node ------------------------------------------------------------


def reporter_node(state: RedTeamState) -> dict:
    """LangGraph node: Markdown + JSON raporu üret, dosyaya yaz."""
    history: list[AttackAttempt] = list(state.get("attack_history", []))
    summary = _summarize(history)

    markdown = build_markdown(state, summary)
    payload = build_json(state, summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"report_{stamp}.json"
    md_path = REPORTS_DIR / f"report_{stamp}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    logger.info(
        "reporter_node: rapor yazıldı (%d deneme, %d başarılı) -> %s",
        summary["total_attempts"], summary["successful"], json_path.name,
    )
    return {
        "final_report": markdown,
        "report_json_path": str(json_path),
    }
