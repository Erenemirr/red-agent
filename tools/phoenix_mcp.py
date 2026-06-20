"""Phoenix okuma katmanı — run'lar arası (kalıcı) öğrenme.

`attack_history` her run'da sıfırlanır; Phoenix ise TÜM geçmiş run'ların
trace'lerini saklar. Bu modül Phoenix'e geri sorup, Judge'ın yazdığı
`attack_evaluation` span'lerindeki `attack.category` / `attack.score` /
`attack.success` attribute'larını okur ve kategori bazında run'lar arası
istatistik çıkarır. Analyzer bunu yerel (bu run) istatistiğiyle birleştirir →
agent önceki oturumlardan öğrenir.

Anahtar/SDK yoksa veya henüz işaretli span yoksa `None` döner; Analyzer yerele düşer.
"""

from __future__ import annotations

import logging
import time

from config.settings import settings

logger = logging.getLogger(__name__)

# Run içinde her turda Phoenix'i tekrar tekrar sorgulamamak için kısa süreli cache.
_CACHE_TTL_SEC = 60
_cache: dict = {"ts": 0.0, "attacks": None}


class PhoenixInsightSource:
    """Phoenix'ten run'lar arası kategori istatistiği çeken okuma kaynağı."""

    @staticmethod
    def available() -> bool:
        """Phoenix anahtarı ve okuma client'ı mevcut mu?"""
        if not (settings.enable_tracing and settings.phoenix_api_key):
            return False
        try:
            from phoenix.client import Client  # noqa: F401
        except ImportError:
            return False
        return True

    def _client(self):
        from phoenix.client import Client

        return Client(
            base_url=settings.phoenix_collector_endpoint,
            api_key=settings.phoenix_api_key,
        )

    def _marker_attacks(self, limit: int = 5000) -> list[dict] | None:
        """Phoenix'teki `attack_evaluation` span'lerinin attack dict'lerini döndür.

        Tek sorgu; sonuç cache'lenir (aynı run'da her tur tekrar sorgulamamak için).
        Hem kategori hem teknik×hedef istatistiği bundan türetilir.
        """
        if not self.available():
            return None

        now = time.time()
        if _cache["attacks"] is not None and (now - _cache["ts"]) < _CACHE_TTL_SEC:
            return _cache["attacks"]

        try:
            df = self._client().spans.get_spans_dataframe(
                project_identifier=settings.phoenix_project_name,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Phoenix sorgusu atlandı: %s", exc)
            return None

        # Phoenix iç içe attribute'ları tek bir `attributes.attack` DICT kolonunda toplar.
        col = "attributes.attack"
        if df is None or len(df) == 0 or col not in df.columns:
            _cache.update(ts=now, attacks=None)
            return None

        attacks = [row for row in df[col].tolist() if isinstance(row, dict)]
        _cache.update(ts=now, attacks=attacks or None)
        if attacks:
            logger.info("Phoenix: %d run'lar-arası işaretli span okundu.", len(attacks))
        return attacks or None

    def get_category_stats(self, limit: int = 5000) -> dict | None:
        """Run'lar arası kategori istatistiği: {kategori: {attempts, successes, score_sum}}."""
        attacks = self._marker_attacks(limit)
        if not attacks:
            return None
        stats: dict[str, dict] = {}
        for a in attacks:
            cat = str(a.get("category", "")).strip()
            if not cat:
                continue
            s = stats.setdefault(cat, {"attempts": 0, "successes": 0, "score_sum": 0.0})
            s["attempts"] += 1
            s["score_sum"] += _to_float(a.get("score"))
            if _truthy(a.get("success")):
                s["successes"] += 1
        return stats or None

    def get_technique_objective_stats(self, limit: int = 5000) -> dict | None:
        """Run'lar arası teknik×hedef istatistiği:
        {hedef_kategorisi: {teknik: {attempts, successes, score_sum}}}."""
        attacks = self._marker_attacks(limit)
        if not attacks:
            return None
        matrix: dict[str, dict] = {}
        for a in attacks:
            tech = str(a.get("category", "")).strip()
            objc = str(a.get("objective_category", "")).strip() or "?"
            if not tech:
                continue
            techs = matrix.setdefault(objc, {})
            s = techs.setdefault(tech, {"attempts": 0, "successes": 0, "score_sum": 0.0})
            s["attempts"] += 1
            s["score_sum"] += _to_float(a.get("score"))
            if _truthy(a.get("success")):
                s["successes"] += 1
        return matrix or None


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0"}
