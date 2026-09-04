"""Hedef sistem interface'i — test edilen kurumsal chatbot.

Hedef, bir `system_prompt` ile tanımlanan ve saldırı prompt'larına yanıt üreten
herhangi bir LLM olabilir. `TargetSystem` soyut interface'i sayesinde arkadaki
model değiştirilebilir (Gemini, OpenAI, lokal model, mock...). Red team döngüsü
yalnızca `generate()` sözleşmesine bağımlıdır.

Varsayılan implementasyon `GeminiTarget` — Google AI Studio ücretsiz tier.
API anahtarı yokken hızlı geliştirme/test için `EchoTarget` mock'u da vardır.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.settings import settings
from agents.state import AttackAttempt, RedTeamState
from tools import gemini

logger = logging.getLogger(__name__)


@dataclass
class TargetResponse:
    """Hedefin tek bir saldırıya verdiği yanıt + meta veri."""

    text: str                    # modelin ürettiği yanıt metni
    model: str                   # yanıtı üreten model adı
    error: str | None = None     # çağrı başarısızsa hata mesajı
    infra_error: bool = False    # hata altyapı/kota kaynaklı mı (hedef savunması değil)?

    @property
    def ok(self) -> bool:
        return self.error is None


class TargetSystem(ABC):
    """Test edilen hedef chatbot için soyut sözleşme.

    Yeni bir backend eklemek için bu sınıfı genişlet ve `generate`'i uygula.
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, attack_prompt: str) -> TargetResponse:
        """Saldırı prompt'una hedefin yanıtını üret (tek-tur)."""
        raise NotImplementedError

    def generate_conversation(self, messages: list[dict]) -> TargetResponse:
        """Çok-turlu: tüm konuşmayı işleyip yanıt üret.

        Varsayılan (durumsuz) davranış: yalnızca son mesajı işler. Durumlu
        backend'ler (GeminiTarget) bunu override edip tüm geçmişi gönderir.
        """
        last = messages[-1].get("content", "") if messages else ""
        return self.generate(last)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Bu hedefin arkasındaki model adı (raporlama/trace için)."""
        raise NotImplementedError


class GeminiTarget(TargetSystem):
    """Gemini API ile çalışan hedef (varsayılan).

    `google-genai` paketi ve `GEMINI_API_KEY` gerektirir. Bu ikisinden biri
    eksikse `available()` False döner; çağıran taraf EchoTarget'a düşebilir.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(system_prompt)
        self._model_name = model or settings.gemini_model
        self._temperature = (
            temperature if temperature is not None else settings.target_temperature
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def available() -> bool:
        """API anahtarı ve SDK mevcut mu?"""
        return gemini.available()

    def generate(self, attack_prompt: str) -> TargetResponse:
        try:
            # Hedefin kişiliği/kısıtlamaları system_instruction ile tanımlanır —
            # saldırılar tam da bu talimatı aşmaya çalışır.
            result = gemini.generate(
                "target",
                attack_prompt,
                system_instruction=self.system_prompt,
                temperature=self._temperature,
            )
            text = gemini.safe_text(result)
            if not text:
                # Gemini güvenlik filtresi yanıtı bloklamış olabilir — bu da bir
                # veri noktası; ham geri bildirimi hata alanına koy.
                return TargetResponse(
                    text="",
                    model=self._model_name,
                    error=f"boş yanıt ({gemini.block_reason(result)})",
                )
            return TargetResponse(text=text, model=self._model_name)
        except Exception as exc:  # noqa: BLE001 - dış API hatasını yumuşat
            logger.warning("GeminiTarget çağrısı başarısız: %s", exc)
            return TargetResponse(
                text="", model=self._model_name, error=str(exc),
                infra_error=gemini._is_rate_limit_error(exc),
            )

    def generate_conversation(self, messages: list[dict]) -> TargetResponse:
        """Durumlu: tüm konuşmayı Gemini'ye gönder (hedef önceki turları hatırlar)."""
        try:
            result = gemini.generate_chat(
                "target",
                messages,
                system_instruction=self.system_prompt,
                temperature=self._temperature,
            )
            text = gemini.safe_text(result)
            if not text:
                return TargetResponse(
                    text="", model=self._model_name,
                    error=f"boş yanıt ({gemini.block_reason(result)})",
                )
            return TargetResponse(text=text, model=self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GeminiTarget konuşma çağrısı başarısız: %s", exc)
            return TargetResponse(
                text="", model=self._model_name, error=str(exc),
                infra_error=gemini._is_rate_limit_error(exc),
            )


class EchoTarget(TargetSystem):
    """API'siz çalışan mock hedef — geliştirme/test için.

    Gerçek bir LLM gibi davranmaz; deterministik, basit bir yanıt döndürür.
    Saldırının system prompt'a doğru iletilip iletilmediğini görmeye yarar.
    """

    @property
    def model_name(self) -> str:
        return "echo-mock"

    def generate(self, attack_prompt: str) -> TargetResponse:
        text = (
            "[EchoTarget mock yanıtı] "
            f"System prompt: {self.system_prompt[:80]!r} | "
            f"Saldırı: {attack_prompt[:120]!r}"
        )
        return TargetResponse(text=text, model=self.model_name)


def build_target(
    system_prompt: str,
    *,
    prefer_mock: bool = False,
) -> TargetSystem:
    """Ortama göre uygun hedef implementasyonunu seç.

    - `prefer_mock=True` ya da Gemini kullanılamıyorsa → EchoTarget
    - Aksi halde → GeminiTarget
    """
    if prefer_mock or not GeminiTarget.available():
        if not prefer_mock:
            logger.info("Gemini kullanılamıyor (API key/SDK eksik) — EchoTarget'a düşülüyor.")
        return EchoTarget(system_prompt)
    return GeminiTarget(system_prompt)


# --- LangGraph node ------------------------------------------------------------


def target_node(state: RedTeamState) -> dict:
    """LangGraph node: bekleyen saldırıyı hedefe gönderip yanıtı ekle.

    Attacker `pending_attempt`'i prompt ile doldurmuş olmalı. Bu node hedefi
    çağırır ve aynı pending kaydı `response` ile günceller (Judge tamamlayacak).
    """
    pending: AttackAttempt | None = state.get("pending_attempt")
    if not pending or not pending.get("prompt"):
        logger.warning("target_node: gönderilecek prompt yok.")
        return {}

    # Çok-turlu: attacker mesajını konuşmaya ekle, hedefe TÜM konuşmayı gönder.
    conversation = list(state.get("conversation", []))
    conversation.append({"role": "user", "content": pending["prompt"]})

    target = build_target(state.get("target_system", ""))
    result = target.generate_conversation(conversation)

    # Kota/altyapı hatası (429): bu bir hedef yanıtı DEĞİL. Konuşmaya sahte bir
    # "model" turu ekleme (kampanyayı kirletir) ve pending'i infra_error diye
    # işaretle — Judge bunu öğrenmeye yazmadan atlar.
    if result.infra_error:
        updated: AttackAttempt = {
            **pending,
            "response": f"[ALTYAPI HATASI] {result.error}",
            "infra_error": True,
        }
        logger.warning("target_node: kota/altyapı hatası — tur atlanacak (%s).", result.error)
        return {"pending_attempt": updated}  # conversation'a dokunma

    text = result.text if result.ok else f"[HATA] {result.error}"
    conversation.append({"role": "model", "content": text})

    # Hedef yanıtını pending kayda işle.
    updated = {**pending, "response": text}
    logger.info(
        "target_node: model=%s ok=%s tur=%d yanıt_len=%d",
        result.model, result.ok, len(conversation) // 2, len(result.text),
    )
    return {"pending_attempt": updated, "conversation": conversation}
