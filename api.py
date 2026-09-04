"""FastAPI wrapper — Red Team Agent'ı REST servisi olarak sun.

Endpoint'ler:
  GET  /health  → servis durumu
  POST /scan    → tarama başlat. İnterrupt'a takılırsa "interrupted" döner
                  (auto_continue=true ise otomatik devam eder).
  POST /resume  → duraklatılmış taramayı operatör kararıyla (continue/stop) sürdür.

İnteraktif human-in-the-loop: graph 3 kritik açıkta `interrupt()` ile durur,
durumu çağırana döner; operatör açıkları inceleyip /resume ile karar verir.
Paylaşılan graph (MemorySaver checkpointer) durumu thread_id'ye göre hatırlar.

Çalıştırma:  uvicorn api:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from config.settings import settings
from main import get_app_graph, setup_tracing
from agents.state import make_initial_state
from security import rate_limit, require_api_key
from tools import usage

logger = logging.getLogger("red_team.api")

_tracer_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    global _tracer_provider
    _tracer_provider = setup_tracing()
    logger.info("Red Team Agent API hazır.")
    yield
    if _tracer_provider is not None:
        _tracer_provider.force_flush()


app = FastAPI(
    title="Red Team Agent API",
    description="Kurumsal LLM sistemlerinin güvenlik açıklarını otomatik tespit eden multi-agent sistemi.",
    version="1.1.0",
    lifespan=lifespan,
)


# --- Şemalar ------------------------------------------------------------------


class ScanRequest(BaseModel):
    target_system: str = Field(..., description="Hedef chatbot'un system prompt'u.", min_length=10)
    max_rounds: int = Field(3, ge=1, le=50, description="Maksimum saldırı turu (~3 Gemini çağrısı/tur).")
    auto_continue: bool = Field(
        False,
        description="True ise interrupt'larda otomatik devam (insan onayı atlanır).",
    )


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="/scan'in döndürdüğü oturum kimliği.")
    decision: str = Field("continue", description="Operatör kararı: 'continue' veya 'stop'.")


class ScanSummary(BaseModel):
    total_attempts: int
    successful: int
    critical: int
    success_rate: float


class ScanResult(BaseModel):
    status: str = Field(..., description="'interrupted' (onay bekliyor) | 'completed' (rapor hazır).")
    thread_id: str
    # status == "interrupted" alanları:
    message: str | None = None
    critical_count: int | None = None
    round: int | None = None
    # status == "completed" alanları:
    summary: ScanSummary | None = None
    rounds_run: int | None = None
    human_interrupts: int | None = None
    gemini_calls: int | None = None
    report_markdown: str | None = None
    report_json_path: str | None = None


# --- Yardımcılar --------------------------------------------------------------


def _config(thread_id: str, max_rounds: int) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_rounds * 6 + 25,
    }


def _interrupt_value(result: dict):
    """invoke sonucunda bekleyen interrupt varsa payload'ını döndür, yoksa None."""
    intr = result.get("__interrupt__")
    return intr[0].value if intr else None


def _to_result(result: dict, thread_id: str) -> ScanResult:
    """Graph invoke sonucunu (interrupt ya da tamamlanmış) API yanıtına çevir."""
    payload = _interrupt_value(result)
    if payload:
        return ScanResult(
            status="interrupted",
            thread_id=thread_id,
            message=payload.get("message"),
            critical_count=payload.get("critical_count"),
            round=payload.get("round"),
        )

    history = result.get("attack_history", [])
    successes = [a for a in history if a.get("verdict", {}).get("success")]
    total = len(history)
    return ScanResult(
        status="completed",
        thread_id=thread_id,
        summary=ScanSummary(
            total_attempts=total,
            successful=len(successes),
            critical=result.get("critical_count", 0),
            success_rate=round(len(successes) / total, 3) if total else 0.0,
        ),
        rounds_run=result.get("current_round", 0),
        human_interrupts=result.get("human_interrupt_count", 0),
        gemini_calls=usage.total(),
        report_markdown=result.get("final_report", ""),
        report_json_path=result.get("report_json_path", ""),
    )


# --- Endpoint'ler -------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
        "tracing_enabled": settings.enable_tracing and bool(settings.phoenix_api_key),
        "auth_enabled": bool(settings.api_key),
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "model": settings.gemini_model,
    }


@app.post(
    "/scan",
    response_model=ScanResult,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def scan(req: ScanRequest):
    """Tarama başlat. İnterrupt'a takılırsa 'interrupted' döner (resume bekler)."""
    usage.reset()
    thread_id = str(uuid.uuid4())
    graph = get_app_graph()
    config = _config(thread_id, req.max_rounds)
    state = make_initial_state(req.target_system, req.max_rounds)

    try:
        result = graph.invoke(state, config)
        # auto_continue: interrupt'ları otomatik geç (insan onayı atlanır).
        while req.auto_continue and _interrupt_value(result):
            result = graph.invoke(Command(resume="continue"), config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tarama başarısız")
        raise HTTPException(status_code=500, detail=f"Tarama başarısız: {exc}") from exc

    return _to_result(result, thread_id)


@app.post(
    "/resume",
    response_model=ScanResult,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def resume(req: ResumeRequest):
    """Duraklatılmış taramayı operatör kararıyla sürdür (continue/stop)."""
    graph = get_app_graph()
    config = {"configurable": {"thread_id": req.thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="thread_id bulunamadı.")
    if not snapshot.next:
        # Graph zaten tamamlanmış — mevcut durumu döndür.
        return _to_result(snapshot.values, req.thread_id)

    # recursion_limit'i checkpoint'teki max_rounds'tan türet.
    config["recursion_limit"] = snapshot.values.get("max_rounds", settings.max_rounds) * 6 + 25
    try:
        result = graph.invoke(Command(resume=req.decision), config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resume başarısız")
        raise HTTPException(status_code=500, detail=f"Resume başarısız: {exc}") from exc

    return _to_result(result, req.thread_id)
