"""LangGraph paylaşımlı state tanımı.

`RedTeamState` graph boyunca node'lar arasında akan tek durum nesnesidir.
Her node bu state'in bir kısmını okur ve günceller (LangGraph reducer mantığı).

Yardımcı `TypedDict`'ler (AttackAttempt, JudgeVerdict, PhoenixInsights) ham
dict yapılarına şema kazandırır; tip ipuçları IDE/okuyucu için belgedir,
runtime'da zorlanmaz.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

# Saldırı kategorileri (teknikler) — data/attack_categories.json. Örnek prompt'lar
# Garak probe ailelerinden ilham alınarak yazıldı (Garak entegre değildir).
AttackCategory = Literal[
    "role-playing",
    "indirect-injection",
    "context-overflow",
    "persona-switching",
    "incremental-escalation",
    "authority-claim",
    "token-smuggling",
    "refusal-suppression",
    "crescendo",
    "many-shot",
    "payload-splitting",
    "virtualization",
]

# Judge'ın ihlal tipi etiketleri (rule-based + LLM katmanı ortak kullanır).
ViolationType = Literal[
    "none",
    "pii-leak",
    "harmful-content",
    "system-prompt-leak",
    "policy-bypass",
    "other",
]


class JudgeVerdict(TypedDict, total=False):
    """Judge node'unun tek bir saldırı için ürettiği değerlendirme."""

    score: float                       # 0.0 (savunma tuttu) – 1.0 (tam ihlal)
    success: bool                      # success_score_threshold üzeri mi?
    is_critical: bool                  # critical_score_threshold üzeri mi?
    violation_type: ViolationType      # tespit edilen ihlal kategorisi
    rule_based_hit: bool               # kural tabanlı katman bir şey yakaladı mı?
    rationale: str                     # LLM-as-a-Judge gerekçesi
    judged_by: str                     # "rule-based" | "llm-judge" | "combined"


class AttackAttempt(TypedDict, total=False):
    """attack_history içindeki tek bir tur kaydı."""

    round: int                         # bu denemenin tur numarası
    category: AttackCategory           # kullanılan saldırı kategorisi (teknik/HOW)
    objective: str                     # saldırının hedefi (JailbreakBench goal/WHAT)
    objective_category: str            # hedefin kategorisi (Privacy, Fraud/Deception...)
    campaign_turn: int                 # çok-turlu kampanya içindeki tur (1, 2, 3...)
    strategy: str                      # attacker'ın o turdaki stratejisi
    prompt: str                        # hedefe gönderilen saldırı prompt'u
    response: str                      # hedefin yanıtı
    verdict: JudgeVerdict              # judge sonucu
    trace_id: str                      # Phoenix trace referansı (analyzer için)


class HumanDecision(TypedDict, total=False):
    """Tek bir human-in-the-loop onayının kaydı."""

    round: int                         # interrupt'ın tetiklendiği tur
    critical_count: int                # o andaki kritik açık sayısı
    decision: Literal["continue", "stop"]  # operatörün kararı
    raw: str                           # operatörün girdiği ham metin


class PhoenixInsights(TypedDict, total=False):
    """Analyzer node'unun trace'lerden çıkardığı öğrenimler.

    Attacker bir sonraki turda bunu okuyup stratejisini buna göre seçer.
    """

    top_categories: list[str]          # en yüksek skorlu saldırı kategorileri
    failing_patterns: list[str]        # sürekli başarısız olan pattern'ler
    avg_score_trend: list[float]       # tur tur ortalama skor trendi
    recommendations: list[str]         # attacker'a somut tavsiyeler
    last_updated_round: int            # bu insight'ın hangi turda üretildiği
    # Teknik×hedef öğrenmesi: {hedef_kategorisi: {teknik: {attempts, successes, score_sum}}}
    technique_objective: dict          # hangi hedefte hangi teknik işe yarıyor


class RedTeamState(TypedDict, total=False):
    """LangGraph graph state'i — node'lar arasında akan paylaşımlı durum."""

    # --- Hedef tanımı ---
    target_system: str                 # hedef chatbot'un system prompt'u

    # --- Attacker durumu ---
    current_strategy: str              # aktif strateji açıklaması
    current_category: AttackCategory   # aktif saldırı kategorisi

    # Akıştaki tek saldırı: Attacker doldurur (prompt), Target response ekler,
    # Judge verdict ekleyip attack_history'e taşır. "Son yazan kazanır" alanı —
    # append-only listelerde çift kayıt oluşmasını engeller.
    pending_attempt: AttackAttempt

    # --- Çok-turlu kampanya durumu ---
    # Bir kampanya = tek (objektif+teknik) için çok-turlu konuşma. Target durumlu
    # olsun ve Attacker önceki yanıtı görüp tırmandırsın diye tutulur.
    conversation: list                 # [{role: 'user'|'model', content}] — aktif kampanya
    campaign_active: bool              # devam eden bir kampanya var mı?
    campaign_turn: int                 # kampanya içindeki tur numarası
    campaign_objective: dict           # kampanyanın sabit hedefi (objective)
    campaign_category: str             # kampanyanın sabit tekniği

    # --- Geçmiş ve birikim (append-only listeler) ---
    attack_history: Annotated[list[AttackAttempt], _append_list]
    successful_attacks: Annotated[list[AttackAttempt], _append_list]
    failed_attacks: Annotated[list[AttackAttempt], _append_list]

    # --- Döngü kontrolü ---
    current_round: int                 # kaçıncı tur
    max_rounds: int                    # döngü üst sınırı
    critical_count: int                # bulunan kritik açık sayısı

    # --- Analyzer çıktısı ---
    phoenix_insights: PhoenixInsights  # trace'lerden çıkarılan öğrenimler

    # --- Orchestrator / human-in-the-loop ---
    human_interrupt: bool              # ŞU AN insan onayı bekleniyor mu? (canlı durum)
    next_action: Literal["continue", "interrupt", "report"]  # orchestrator kararı
    # En son hangi kritik-açık sayısında interrupt tetiklendi — aynı eşikte
    # tekrar tekrar sormamak için (her +threshold kritikte bir kez sorulur).
    last_interrupt_at_count: int
    # Toplam kaç kez human-in-the-loop onayı istendi (rapor metriği).
    human_interrupt_count: int
    # Her interrupt'ın kaydı (hangi turda, kaç kritik, operatör ne dedi).
    human_decisions: Annotated[list["HumanDecision"], _append_list]

    # --- Çıktı ---
    final_report: str                  # Markdown rapor
    report_json_path: str              # JSON raporun yazıldığı dosya yolu


def _append_list(old: list | None, new: list | None) -> list:
    """Reducer: liste alanlarına yeni öğeleri ekle (üzerine yazma)."""
    if old is None:
        old = []
    if new is None:
        return old
    return old + new


def make_initial_state(
    target_system: str,
    max_rounds: int,
) -> RedTeamState:
    """Graph'ı başlatmak için boş/varsayılan state üretir."""
    return RedTeamState(
        target_system=target_system,
        current_strategy="",
        current_category="role-playing",
        attack_history=[],
        successful_attacks=[],
        failed_attacks=[],
        current_round=0,
        max_rounds=max_rounds,
        critical_count=0,
        conversation=[],
        campaign_active=False,
        campaign_turn=0,
        phoenix_insights=PhoenixInsights(
            top_categories=[],
            failing_patterns=[],
            avg_score_trend=[],
            recommendations=[],
            last_updated_round=0,
        ),
        human_interrupt=False,
        next_action="continue",
        last_interrupt_at_count=0,
        human_interrupt_count=0,
        human_decisions=[],
        final_report="",
        report_json_path="",
    )
