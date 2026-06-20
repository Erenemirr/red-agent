"""Analyzer node — self-improvement loop'unun beyni.

Geçmiş saldırı verisini analiz edip `phoenix_insights`'ı günceller. Attacker bir
sonraki turda bu içgörüleri okuyup stratejisini buna göre seçer (exploit/explore).

İki katman:
  1. Yerel (her zaman): `attack_history`'den kategori başarı oranı, başarısız
     pattern'ler ve skor trendi hesaplanır — gerçek, deterministik, API'siz.
  2. Phoenix (opsiyonel): bağlıysa trace-seviyesi metrikle zenginleştirilir
     (`tools.phoenix_mcp`). Bağlantı yoksa sessizce atlanır.

Çıkardığı içgörüler (CLAUDE.md):
  - En yüksek skorlu / en çok işe yarayan kategoriler (top_categories)
  - Sürekli başarısız pattern'ler (failing_patterns)
  - Ortalama skor trendi (avg_score_trend)
  - Attacker'a somut tavsiyeler (recommendations)
"""

from __future__ import annotations

import logging

from agents.state import AttackAttempt, PhoenixInsights, RedTeamState
from tools.phoenix_mcp import PhoenixInsightSource

logger = logging.getLogger(__name__)


def _category_stats(history: list[AttackAttempt]) -> dict[str, dict]:
    """Kategori bazında deneme/başarı/skor toplamlarını çıkar."""
    stats: dict[str, dict] = {}
    for attempt in history:
        cat = attempt.get("category", "?")
        verdict = attempt.get("verdict", {})
        s = stats.setdefault(cat, {"attempts": 0, "successes": 0, "score_sum": 0.0})
        s["attempts"] += 1
        s["score_sum"] += float(verdict.get("score", 0.0))
        if verdict.get("success"):
            s["successes"] += 1
    return stats


def _cumulative_trend(history: list[AttackAttempt]) -> list[float]:
    """Tur tur kümülatif ortalama skor (öğrenme trendini gösterir)."""
    trend: list[float] = []
    running = 0.0
    for i, attempt in enumerate(history, start=1):
        running += float(attempt.get("verdict", {}).get("score", 0.0))
        trend.append(round(running / i, 3))
    return trend


def compute_insights(
    history: list[AttackAttempt],
    current_round: int,
    phoenix_stats: dict | None = None,
) -> PhoenixInsights:
    """İçgörü üret. `phoenix_stats` verilirse (run'lar arası) yerelle birleştirilir."""
    local_stats = _category_stats(history)
    # Yerel (bu run) + Phoenix (geçmiş run'lar) istatistiğini birleştir.
    stats = _merge_stats(local_stats, phoenix_stats) if phoenix_stats else local_stats

    trend = _cumulative_trend(history)  # trend bu run'a özgü kalır

    if not stats:
        return PhoenixInsights(
            top_categories=[],
            failing_patterns=[],
            avg_score_trend=trend,
            recommendations=["Henüz veri yok — repertuarı keşfetmeye devam et."],
            last_updated_round=current_round,
        )

    # Kategorileri başarı oranına, eşitlikte ortalama skora göre sırala.
    # Laplace smoothing: ham successes/attempts yerine (successes+1)/(attempts+2).
    # Böylece tek şanslı başarı (1/1=1.0) tüm sıralamayı domine etmez; bir kategori
    # ancak TUTARLI başarı gösterirse üste çıkar. "Az kanıttan aşırı güven"i önler.
    #   1/1 → 0.67 (1.0 değil)   |   4/5 → 0.71 (tutarlı, daha üstte)   |   0/3 → 0.20
    def rank_key(item):
        _cat, s = item
        smoothed_rate = (s["successes"] + 1) / (s["attempts"] + 2)
        avg = s["score_sum"] / s["attempts"]
        return (smoothed_rate, avg)

    ranked = sorted(stats.items(), key=rank_key, reverse=True)
    top_categories = [cat for cat, s in ranked if s["successes"] > 0]
    failing_patterns = [
        cat for cat, s in stats.items()
        if s["attempts"] >= 1 and s["successes"] == 0
    ]

    # --- Tavsiyeler ---
    recommendations: list[str] = []
    if top_categories:
        best = top_categories[0]
        bs = stats[best]
        recommendations.append(
            f"`{best}` en etkili (ort. skor {bs['score_sum'] / bs['attempts']:.2f}, "
            f"{bs['successes']}/{bs['attempts']}) — varyasyonlarını derinleştir."
        )
    if failing_patterns:
        recommendations.append(
            f"`{', '.join(failing_patterns)}` işe yaramıyor — yaklaşımı değiştir veya bırak."
        )
    if phoenix_stats:
        hist_total = sum(s["attempts"] for s in phoenix_stats.values())
        recommendations.append(
            f"Phoenix'ten {hist_total} geçmiş deneme okundu — sıralama run'lar arası veriyle güçlendirildi."
        )
    if len(trend) >= 2:
        delta = trend[-1] - trend[0]
        yon = "yükseliyor" if delta > 0.05 else ("düşüyor" if delta < -0.05 else "sabit")
        recommendations.append(f"Ortalama skor trendi {yon} ({trend[0]:.2f} → {trend[-1]:.2f}).")

    return PhoenixInsights(
        top_categories=top_categories,
        failing_patterns=failing_patterns,
        avg_score_trend=trend,
        recommendations=recommendations,
        last_updated_round=current_round,
    )


def _merge_stats(a: dict, b: dict) -> dict:
    """İki kategori-istatistik sözlüğünü topla (attempts/successes/score_sum)."""
    out: dict[str, dict] = {}
    for src in (a, b):
        for cat, s in src.items():
            o = out.setdefault(cat, {"attempts": 0, "successes": 0, "score_sum": 0.0})
            o["attempts"] += s["attempts"]
            o["successes"] += s["successes"]
            o["score_sum"] += s["score_sum"]
    return out


def _tech_obj_stats(history: list[AttackAttempt]) -> dict:
    """Yerel teknik×hedef matrisi: {hedef_kategorisi: {teknik: {attempts, successes, score_sum}}}."""
    matrix: dict[str, dict] = {}
    for a in history:
        tech = a.get("category")
        if not tech:
            continue
        objc = a.get("objective_category") or "?"
        v = a.get("verdict", {})
        techs = matrix.setdefault(objc, {})
        s = techs.setdefault(tech, {"attempts": 0, "successes": 0, "score_sum": 0.0})
        s["attempts"] += 1
        s["score_sum"] += float(v.get("score", 0.0))
        if v.get("success"):
            s["successes"] += 1
    return matrix


def _merge_tech_obj(a: dict, b: dict) -> dict:
    """İki teknik×hedef matrisini topla."""
    out: dict[str, dict] = {}
    for src in (a, b):
        for objc, techs in src.items():
            o = out.setdefault(objc, {})
            for tech, s in techs.items():
                t = o.setdefault(tech, {"attempts": 0, "successes": 0, "score_sum": 0.0})
                t["attempts"] += s["attempts"]
                t["successes"] += s["successes"]
                t["score_sum"] += s["score_sum"]
    return out


def analyzer_node(state: RedTeamState) -> dict:
    """LangGraph node: yerel + Phoenix (run'lar arası) içgörüyü birleştir."""
    history = list(state.get("attack_history", []))
    current_round = state.get("current_round", 0)

    # Phoenix bağlıysa geçmiş run'ların istatistiğini çek (cross-run öğrenme).
    phoenix_stats = None
    phoenix_tech_obj = None
    source = PhoenixInsightSource()
    if source.available():
        phoenix_stats = source.get_category_stats()
        phoenix_tech_obj = source.get_technique_objective_stats()

    insights = compute_insights(history, current_round, phoenix_stats=phoenix_stats)

    # Teknik×hedef matrisi (yerel + Phoenix) → Attacker eşleştirerek öğrensin.
    tech_obj = _tech_obj_stats(history)
    if phoenix_tech_obj:
        tech_obj = _merge_tech_obj(tech_obj, phoenix_tech_obj)
    insights = {**insights, "technique_objective": tech_obj}

    logger.info(
        "analyzer_node: tur %d | top=%s | failing=%s | phoenix=%s",
        current_round,
        insights.get("top_categories"),
        insights.get("failing_patterns"),
        f"{sum(s['attempts'] for s in phoenix_stats.values())} geçmiş deneme" if phoenix_stats else "yok",
    )
    return {"phoenix_insights": insights}
