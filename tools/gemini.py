"""Merkezi Gemini istemcisi — modern `google-genai` SDK.

Üç çağırıcı (target, attacker, judge) tek bir istemci + `generate()` yardımcısı
üzerinden gider. Böylece istemci kurulumu, token sınırı, usage kaydı ve model
seçimi tek yerde yönetilir. `openinference-instrumentation-google-genai` bu SDK'yı
otomatik instrument ettiği için tüm çağrılar Phoenix'e trace olarak akar.

Eski `google-generativeai` SDK'sından (deprecated) buraya geçildi.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time

from config.settings import settings
from tools import usage

logger = logging.getLogger(__name__)

_client = None  # tembel başlatılan singleton istemci


class QuotaExceededError(RuntimeError):
    """429 / RESOURCE_EXHAUSTED — free-tier kota/RPM aşımı, retry'lar tükendi.

    Çağıran taraf bunu "hedefin savunması" değil "altyapı hatası" olarak ele
    almalı (öğrenme verisine yazılmamalı).
    """


# --- 429 tespiti ve backoff ----------------------------------------------------

# Sunucu "30s" / "retryDelay: 12" gibi bir bekleme önerirse onu yakala.
_RETRY_DELAY_RE = re.compile(r"retry.?delay['\":\s]*['\"]?(\d+(?:\.\d+)?)\s*s?", re.I)


def _is_rate_limit_error(exc: Exception) -> bool:
    """İstisna bir 429 / kota aşımı mı? (SDK sürümünden bağımsız, dayanıklı tespit.)"""
    if isinstance(exc, QuotaExceededError):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text


def _server_retry_delay(exc: Exception) -> float | None:
    """Sunucunun önerdiği bekleme süresini (sn) çıkar — yoksa None."""
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _backoff_seconds(attempt: int, exc: Exception) -> float:
    """attempt. (0-tabanlı) yeniden deneme için beklenecek süre.

    Öncelik sunucunun retryDelay'i; yoksa üstel backoff (base * 2**attempt).
    Küçük bir jitter eklenir (aynı anda birden çok çağrının senkron çarpmasını
    önler) ve `gemini_retry_max_delay` ile kırpılır.
    """
    server = _server_retry_delay(exc)
    if server is not None:
        base = server
    else:
        base = settings.gemini_retry_base_delay * (2 ** attempt)
    base = min(base, settings.gemini_retry_max_delay)
    return base + random.uniform(0, 0.5)


# --- Proaktif hız sınırı (RPM) -------------------------------------------------

_rate_lock = threading.Lock()
_last_call_ts = 0.0


def _throttle() -> None:
    """Free-tier RPM'i aşmamak için çağrılar arası minimum aralığı uygula.

    `gemini_rpm <= 0` ise no-op (varsayılan). Aksi halde ardışık çağrılar
    60/RPM saniye arayla sıraya alınır — 429 hiç oluşmaz.
    """
    rpm = settings.gemini_rpm
    if rpm <= 0:
        return
    min_interval = 60.0 / rpm
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _last_call_ts + min_interval - now
        if wait > 0:
            logger.info("Rate limit (RPM=%d): %.1fs bekleniyor.", rpm, wait)
            time.sleep(wait)
            now = time.monotonic()
        _last_call_ts = now


def _execute(role: str, call):
    """`call` (0-argümanlı) Gemini çağrısını throttle + retry ile çalıştır.

    429 gelince backoff'la yeniden dener; denemeler tükenince
    `QuotaExceededError` fırlatır. 429 dışındaki hataları olduğu gibi yükseltir.
    Usage yalnızca gerçekten API'ye gidilen her denemede kaydedilir.
    """
    max_retries = settings.gemini_max_retries
    attempt = 0
    while True:
        _throttle()
        usage.record(role)
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit_error(exc):
                raise
            if attempt >= max_retries:
                logger.warning("Gemini 429: %d deneme sonrası pes edildi.", max_retries)
                raise QuotaExceededError(str(exc)) from exc
            wait = _backoff_seconds(attempt, exc)
            logger.info("Gemini 429 (deneme %d/%d) — %.1fs sonra yeniden.",
                        attempt + 1, max_retries, wait)
            time.sleep(wait)
            attempt += 1


def available() -> bool:
    """API anahtarı ve SDK mevcut mu?"""
    if not settings.gemini_api_key:
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def get_client():
    """google-genai istemcisini (tek sefer) oluştur."""
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def generate(
    role: str,
    contents: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    max_output_tokens: int | None = None,
):
    """Tek bir Gemini çağrısı yap, ham response nesnesini döndür.

    role: usage sayacı etiketi ('attacker' | 'target' | 'judge').
    json_mode: True ise structured JSON çıktısı ister (judge için).
    Ham response döner; çağıran `.text` ve hata/blok durumunu kendi yorumlar.
    """
    from google.genai import types

    client = get_client()
    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens or settings.gemini_max_output_tokens,
        # gemini-2.5-flash bir "thinking" modeli; düşünme de output token harcayıp
        # asıl yanıtı kırpıyordu. Bu görevler (saldırı üret / skor ver) derin
        # muhakeme gerektirmediğinden thinking'i kapatıyoruz: kırpılma biter,
        # token tüketimi düşer (free-tier dostu), yanıt hızlanır.
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    # throttle + 429 retry `_execute` içinde; kota tükenirse QuotaExceededError.
    return _execute(role, lambda: client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    ))


def generate_chat(
    role: str,
    messages: list[dict],
    *,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int | None = None,
):
    """Çok-turlu (durumlu) Gemini çağrısı — tüm konuşmayı gönderir.

    messages: [{"role": "user"|"model", "content": str}, ...]
    Hedef, önceki turları hatırlar (crescendo/incremental-escalation için şart).
    """
    from google.genai import types

    client = get_client()
    contents = [
        {"role": m.get("role", "user"), "parts": [{"text": m.get("content", "")}]}
        for m in messages
    ]
    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens or settings.gemini_max_output_tokens,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    return _execute(role, lambda: client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    ))


def safe_text(response) -> str:
    """response.text'i güvenli oku (bloklanmışsa raise edebilir)."""
    try:
        return (response.text or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def block_reason(response) -> str:
    """Yanıt boşsa bloklama/finish sebebini güvenli çıkar."""
    try:
        pf = getattr(response, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            return f"block_reason={pf.block_reason}"
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            return f"finish_reason={getattr(candidates[0], 'finish_reason', '?')}"
    except Exception:  # noqa: BLE001
        pass
    return "bilinmeyen"
