"""Test fixtures — tüm testleri hermetik (API'siz) çalıştır.

`_hermetic` autouse fixture'ı her testten önce:
  - Gemini'yi "kullanılamaz" gösterir → Judge rule-based'e, Target Echo'ya,
    Attacker şablon fallback'ine düşer (gerçek API çağrısı OLMAZ, kota yanmaz).
  - Phoenix okumasını kapatır → Analyzer ağa çıkmaz.
  - random'ı sabit seed'ler → deterministik sonuç.
"""

from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr("tools.gemini.available", lambda: False)

    from tools.phoenix_mcp import PhoenixInsightSource
    
    monkeypatch.setattr(PhoenixInsightSource, "available", staticmethod(lambda: False))
    random.seed(1337)
    yield 

