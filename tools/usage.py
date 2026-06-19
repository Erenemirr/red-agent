"""Basit Gemini API kullanım sayacı — free-tier farkındalığı için.

Her Gemini çağrısı yapan modül (target, attacker, llm_judge) başarılı/denenmiş
çağrıyı burada kaydeder. main.py run sonunda toplamı yazdırır; böylece bir
çalışmanın kaç API çağrısı tükettiğini görürsün.

Process-global, tek iş parçacıklı CLI kullanımı için yeterli. Studio gibi uzun
ömürlü süreçlerde `reset()` ile sıfırlanabilir.
"""

from __future__ import annotations

from collections import Counter

_counter: Counter = Counter()


def record(role: str, n: int = 1) -> None:
    """Bir çağrıyı kaydet (role: 'attacker' | 'target' | 'judge')."""
    _counter[role] += n
    _counter["total"] += n


def total() -> int:
    """Toplam Gemini çağrısı sayısı."""
    return _counter["total"]


def breakdown() -> dict[str, int]:
    """Role bazında dağılım (total hariç)."""
    return {k: v for k, v in _counter.items() if k != "total"}


def reset() -> None:
    """Sayaçları sıfırla."""
    _counter.clear()
