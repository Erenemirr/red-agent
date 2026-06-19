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
_cache: dict = {"ts": 0.0, "stats": None}


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

    def get_category_stats(self, limit: int = 5000) -> dict | None:
        """Run'lar arası kategori istatistiği döndür.

        { kategori: {"attempts": n, "successes": k, "score_sum": float} }
        Bağlantı yoksa / işaretli span yoksa None.
        """
        if not self.available():
            return None

        # Kısa süreli cache — aynı run'da her tur yeniden sorgulamayı önler.
        now = time.time()
        if _cache["stats"] is not None and (now - _cache["ts"]) < _CACHE_TTL_SEC:
            return _cache["stats"]

        try:
            df = self._client().spans.get_spans_dataframe(
                project_identifier=settings.phoenix_project_name,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Phoenix sorgusu atlandı: %s", exc)
            return None

        # Phoenix iç içe attribute'ları (attack.category, attack.score...) tek bir
        # `attributes.attack` kolonunda DICT olarak toplar.
        col = "attributes.attack"
        if df is None or len(df) == 0 or col not in df.columns:
            _cache.update(ts=now, stats=None)
            return None

        marker = df[df[col].notna()]
        if len(marker) == 0:
            _cache.update(ts=now, stats=None)
            return None

        stats: dict[str, dict] = {}
        for _, row in marker.iterrows():
            attack = row.get(col)
            if not isinstance(attack, dict):
                continue
            cat = str(attack.get("category", "")).strip()
            if not cat:
                continue
            score = _to_float(attack.get("score"))
            success = _truthy(attack.get("success"))
            s = stats.setdefault(cat, {"attempts": 0, "successes": 0, "score_sum": 0.0})
            s["attempts"] += 1
            s["score_sum"] += score
            if success:
                s["successes"] += 1

        _cache.update(ts=now, stats=stats)
        logger.info("Phoenix: %d run'lar-arası işaretli span okundu, %d kategori.",
                    len(marker), len(stats))
        return stats


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0"}
