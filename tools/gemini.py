"""Merkezi Gemini istemcisi — modern `google-genai` SDK.

Üç çağırıcı (target, attacker, judge) tek bir istemci + `generate()` yardımcısı
üzerinden gider. Böylece istemci kurulumu, token sınırı, usage kaydı ve model
seçimi tek yerde yönetilir. `openinference-instrumentation-google-genai` bu SDK'yı
otomatik instrument ettiği için tüm çağrılar Phoenix'e trace olarak akar.

Eski `google-generativeai` SDK'sından (deprecated) buraya geçildi.
"""

from __future__ import annotations

import logging

from config.settings import settings
from tools import usage

logger = logging.getLogger(__name__)

_client = None  # tembel başlatılan singleton istemci


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

    usage.record(role)
    return client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )


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
