"""API güvenliği — API-key auth + basit in-memory rate limiting.

Portfolyo/öğrenme amaçlı, HARİCİ BAĞIMLILIK OLMADAN (Redis/slowapi yok):

- **Auth:** `X-API-Key` header'ı `settings.api_key` ile karşılaştırılır. Anahtar
  tanımlı DEĞİLSE auth kapalıdır — mock/dev/test kurulumları anahtar zorunluluğu
  olmadan çalışsın diye bilinçli bir tercih.
- **Rate limit:** istemci (IP) başına KAYAN PENCERE (sliding window) sayacı;
  son 60 saniyede `settings.rate_limit_per_minute` isteği aşan istemciye 429
  `Too Many Requests` + `Retry-After` döner. Sayaç süreç-içi ve thread-safe'tir.

İkisi de FastAPI `Depends(...)` bağımlılığı olarak endpoint'lere takılır.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from config.settings import settings


# --- API-key auth --------------------------------------------------------------


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """API-key doğrulama bağımlılığı.

    `settings.api_key` boşsa auth devre dışıdır (açık geçer). Tanımlıysa,
    eksik/yanlış anahtar 401 ile reddedilir.
    """
    expected = settings.api_key
    if not expected:
        return  # auth kapalı (dev/mock)
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz veya eksik API anahtarı (X-API-Key başlığı).",
        )


# --- Rate limiting (kayan pencere) ---------------------------------------------


class SlidingWindowRateLimiter:
    """İstemci başına kayan-pencere rate limiter (thread-safe, in-memory).

    Her istemci için son `window` saniyedeki istek zaman damgalarını tutar;
    yeni istek geldiğinde pencere dışı damgalar atılır ve limit kontrol edilir.
    """

    def __init__(self, max_per_minute: int, window: float = 60.0) -> None:
        self.max = max_per_minute
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str) -> None:
        """Bir isteği kaydet; limit aşıldıysa HTTPException(429) fırlat."""
        if self.max <= 0:
            return  # limit kapalı
        now = time.monotonic()
        with self._lock:
            dq = self._hits[client_id]
            # Pencere dışına düşen eski vuruşları temizle.
            while dq and dq[0] <= now - self.window:
                dq.popleft()
            if len(dq) >= self.max:
                retry_after = self.window - (now - dq[0])
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit aşıldı ({self.max}/dk). "
                           f"{retry_after:.0f}s sonra tekrar deneyin.",
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )
            dq.append(now)

    def reset(self) -> None:
        """Tüm sayaçları temizle (testler arası)."""
        with self._lock:
            self._hits.clear()


# Süreç-geneli tek limiter (istemciler arası paylaşılır).
_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)


def rate_limit(request: Request) -> None:
    """Rate-limit bağımlılığı — istemciyi IP adresiyle ayırır."""
    client = request.client.host if request.client else "anon"
    _limiter.check(client)


def reset_rate_limiter() -> None:
    """Testlerin sayaç durumunu sıfırlaması için yardımcı."""
    _limiter.reset()
