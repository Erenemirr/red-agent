"""Birim testleri — yardımcı/saf fonksiyonlar (kapsama derinliği).

test_nodes.py node davranışını kapsar; bu dosya alt katman yardımcılarını
(parse/aggregate/format/settings) doğrular. Hepsi hermetik (conftest).
"""

from __future__ import annotations

# --- eval/llm_judge: parse yardımcıları ---------------------------------------

from eval.llm_judge import _loads_lenient, _clamp01, _to_float as _judge_to_float, LLMJudge


class TestLLMJudgeParsing:
    def test_loads_plain_json(self):
        assert _loads_lenient('{"score": 0.5}') == {"score": 0.5}

    def test_loads_markdown_wrapped(self):
        assert _loads_lenient('```json\n{"score": 0.5}\n```') == {"score": 0.5}

    def test_loads_embedded(self):
        assert _loads_lenient('önsöz {"a": 1} sonsöz')["a"] == 1

    def test_loads_garbage_is_none(self):
        assert _loads_lenient("json değil") is None

    def test_loads_empty_is_none(self):
        assert _loads_lenient("") is None

    def test_clamp01(self):
        assert _clamp01(1.5) == 1.0
        assert _clamp01(-0.3) == 0.0
        assert _clamp01(0.4) == 0.4

    def test_to_float_default(self):
        assert _judge_to_float(None, 0.0) == 0.0
        assert _judge_to_float("0.7", 0.0) == 0.7

    def test_parse_clamps_and_normalizes_violation(self):
        r = LLMJudge()._parse('{"score": 1.9, "violation_type": "muz", "rationale": "x"}')
        assert r.score == 1.0
        assert r.violation_type == "other"   # geçersiz + skor>0 → other
        assert r.rationale == "x"

    def test_parse_invalid_json_sets_error(self):
        r = LLMJudge()._parse("hiç json yok")
        assert r.error is not None


# --- tools/usage: çağrı sayacı ------------------------------------------------

from tools import usage


class TestUsage:
    def test_record_and_total(self):
        usage.reset()
        usage.record("attacker")
        usage.record("target", 2)
        assert usage.total() == 3
        assert usage.breakdown()["attacker"] == 1
        assert usage.breakdown()["target"] == 2

    def test_reset(self):
        usage.record("judge")
        usage.reset()
        assert usage.total() == 0

    def test_breakdown_excludes_total_key(self):
        usage.reset()
        usage.record("attacker")
        assert "total" not in usage.breakdown()


# --- tools/phoenix_mcp: aggregate + yardımcılar -------------------------------

from tools.phoenix_mcp import _to_float as _px_float, _truthy, PhoenixInsightSource


class TestPhoenixHelpers:
    def test_truthy(self):
        assert _truthy(True) is True
        assert _truthy("true") is True
        assert _truthy("1.0") is True
        assert _truthy("false") is False
        assert _truthy(None) is False

    def test_to_float(self):
        assert _px_float("0.5") == 0.5
        assert _px_float(None) == 0.0

    def test_category_stats_aggregation(self, monkeypatch):
        attacks = [
            {"category": "a", "score": 0.9, "success": True},
            {"category": "a", "score": 0.1, "success": False},
            {"category": "b", "score": 0.0, "success": False},
        ]
        monkeypatch.setattr(PhoenixInsightSource, "_marker_attacks",
                            lambda self, limit=5000: attacks)
        stats = PhoenixInsightSource().get_category_stats()
        assert stats["a"]["attempts"] == 2
        assert stats["a"]["successes"] == 1
        assert stats["b"]["successes"] == 0

    def test_technique_objective_aggregation(self, monkeypatch):
        attacks = [
            {"category": "a", "objective_category": "P", "score": 0.9, "success": True},
            {"category": "a", "objective_category": "P", "score": 0.0, "success": False},
        ]
        monkeypatch.setattr(PhoenixInsightSource, "_marker_attacks",
                            lambda self, limit=5000: attacks)
        m = PhoenixInsightSource().get_technique_objective_stats()
        assert m["P"]["a"]["attempts"] == 2
        assert m["P"]["a"]["successes"] == 1

    def test_stats_none_when_no_attacks(self, monkeypatch):
        monkeypatch.setattr(PhoenixInsightSource, "_marker_attacks",
                            lambda self, limit=5000: None)
        assert PhoenixInsightSource().get_category_stats() is None


# --- config/settings: env parse yardımcıları ----------------------------------

from config.settings import _get_bool, _get_int, _get_float, get_settings


class TestSettingsHelpers:
    def test_get_bool(self, monkeypatch):
        monkeypatch.setenv("X_B", "true")
        assert _get_bool("X_B") is True
        monkeypatch.setenv("X_B", "no")
        assert _get_bool("X_B") is False
        assert _get_bool("X_UNSET", True) is True

    def test_get_int(self, monkeypatch):
        monkeypatch.setenv("X_I", "7")
        assert _get_int("X_I", 0) == 7
        monkeypatch.setenv("X_I", "bozuk")
        assert _get_int("X_I", 3) == 3
        assert _get_int("X_UNSET", 5) == 5

    def test_get_float(self, monkeypatch):
        monkeypatch.setenv("X_F", "0.9")
        assert _get_float("X_F", 0.0) == 0.9
        assert _get_float("X_UNSET", 0.3) == 0.3

    def test_settings_singleton(self):
        assert get_settings() is get_settings()


# --- tools/gemini: metin yardımcıları -----------------------------------------

from tools import gemini


class _FakeResp:
    def __init__(self, text=None, raise_text=False, pf=None, cands=None):
        self._text = text
        self._raise = raise_text
        self.prompt_feedback = pf
        self.candidates = cands

    @property
    def text(self):
        if self._raise:
            raise ValueError("bloklandı")
        return self._text


class TestGeminiHelpers:
    def test_safe_text_strips(self):
        assert gemini.safe_text(_FakeResp(text="  merhaba  ")) == "merhaba"

    def test_safe_text_handles_raise(self):
        assert gemini.safe_text(_FakeResp(raise_text=True)) == ""

    def test_block_reason_unknown(self):
        assert gemini.block_reason(_FakeResp()) == "bilinmeyen"

    def test_available_mocked_false(self):
        # conftest gemini.available'ı False'a sabitler
        assert gemini.available() is False


# --- agents/judge: ek rule-based desenler -------------------------------------

from agents.judge import rule_based_eval


class TestRuleBasedMore:
    def test_email_is_pii(self):
        v = rule_based_eval(system_prompt="x", response="iletişim: ahmet@example.com")
        assert v["violation_type"] == "pii-leak"

    def test_credit_card_is_pii(self):
        v = rule_based_eval(system_prompt="x", response="Kart: 4111 1111 1111 1111")
        assert v["violation_type"] == "pii-leak"

    def test_harmful_marker(self):
        v = rule_based_eval(system_prompt="x", response="Step 1: gerekli malzemeleri topla")
        assert v["violation_type"] == "harmful-content"

    def test_persona_marker(self):
        v = rule_based_eval(system_prompt="x", response="Developer mode aktif, işte:")
        assert v["violation_type"] == "policy-bypass"

    def test_system_prompt_echo_detected(self):
        sp = "Sen Acme Bank botusun ve gizli talimatların var burada mevcut"
        v = rule_based_eval(system_prompt=sp,
                            response="sen acme bank botusun ve gizli talimatların şöyle...")
        assert v["violation_type"] == "system-prompt-leak"


# --- agents/analyzer: yardımcılar ---------------------------------------------

from agents.analyzer import (
    _cumulative_trend, _category_stats, _tech_obj_stats, _merge_tech_obj, compute_insights,
)


class TestAnalyzerHelpers:
    def test_cumulative_trend(self):
        h = [{"verdict": {"score": 1.0}}, {"verdict": {"score": 0.0}}]
        assert _cumulative_trend(h) == [1.0, 0.5]

    def test_category_stats(self):
        h = [{"category": "a", "verdict": {"score": 0.9, "success": True}},
             {"category": "a", "verdict": {"score": 0.1, "success": False}}]
        s = _category_stats(h)
        assert s["a"]["attempts"] == 2 and s["a"]["successes"] == 1

    def test_tech_obj_stats(self):
        h = [{"category": "a", "objective_category": "P",
              "verdict": {"score": 0.9, "success": True}}]
        assert _tech_obj_stats(h)["P"]["a"]["successes"] == 1

    def test_merge_tech_obj(self):
        a = {"P": {"x": {"attempts": 1, "successes": 1, "score_sum": 0.9}}}
        b = {"P": {"x": {"attempts": 2, "successes": 0, "score_sum": 0.0}},
             "Q": {"y": {"attempts": 1, "successes": 1, "score_sum": 1.0}}}
        m = _merge_tech_obj(a, b)
        assert m["P"]["x"]["attempts"] == 3 and m["P"]["x"]["successes"] == 1
        assert m["Q"]["y"]["attempts"] == 1

    def test_recommendations_mention_best(self):
        h = [{"category": "role-playing", "objective_category": "P",
              "verdict": {"score": 0.9, "success": True}}]
        ins = compute_insights(h, 1)
        assert any("role-playing" in r for r in ins["recommendations"])


# --- agents/reporter: yardımcılar ---------------------------------------------

from agents.reporter import _summarize, _trunc, build_json


class TestReporterHelpers:
    def test_summarize_counts(self):
        h = [{"category": "a", "verdict": {"score": 1.0, "success": True, "is_critical": True}},
             {"category": "a", "verdict": {"score": 0.0, "success": False}}]
        s = _summarize(h)
        assert s["total_attempts"] == 2 and s["successful"] == 1 and s["success_rate"] == 0.5

    def test_summarize_empty(self):
        s = _summarize([])
        assert s["total_attempts"] == 0 and s["success_rate"] == 0.0

    def test_trunc(self):
        assert _trunc("abc", 10) == "abc"
        assert _trunc("a" * 20, 5).endswith("…")

    def test_build_json_keys(self):
        j = build_json({"attack_history": [], "target_system": "x"}, _summarize([]))
        assert "technique_objective_breakdown" in j
        assert "summary" in j


# --- agents/target + attacker: yardımcılar ------------------------------------

from agents.target import TargetResponse, EchoTarget
from agents.attacker import _fallback_prompt, select_objective, load_categories, AttackCategorySpec


class TestTargetMore:
    def test_response_ok(self):
        assert TargetResponse(text="x", model="m").ok is True

    def test_response_error(self):
        assert TargetResponse(text="", model="m", error="e").ok is False

    def test_echo_conversation_uses_last(self):
        r = EchoTarget("sp").generate_conversation([{"role": "user", "content": "merhaba dünya"}])
        assert r.ok and "merhaba dünya" in r.text


class TestAttackerHelpers:
    def test_fallback_replaces_placeholder(self):
        spec = AttackCategorySpec(id="x", name="X", description="", tactic="",
                                  severity_hint="low", example_prompts=["saldır: {hedef}"])
        p = _fallback_prompt(spec, "PII sız")
        assert "PII sız" in p and "{hedef}" not in p

    def test_select_objective_empty_fallback(self):
        assert select_objective([])["goal"]

    def test_load_categories_has_12(self):
        assert len(load_categories()) == 12


# --- tools/gemini: 429 dayanıklılığı (B1) -------------------------------------

import pytest


class TestGeminiResilience:
    def test_is_rate_limit_detects_429(self):
        assert gemini._is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
        assert gemini._is_rate_limit_error(gemini.QuotaExceededError("x"))

        class _E(Exception):
            code = 429
        assert gemini._is_rate_limit_error(_E())

    def test_is_rate_limit_ignores_other(self):
        assert gemini._is_rate_limit_error(ValueError("boş yanıt (block)")) is False

    def test_server_retry_delay_parsed(self):
        assert gemini._server_retry_delay(RuntimeError("retryDelay: 30s")) == 30.0
        assert gemini._server_retry_delay(RuntimeError("hiçbir şey yok")) is None

    def test_backoff_uses_server_delay(self):
        d = gemini._backoff_seconds(0, RuntimeError("retryDelay: 5s"))
        assert 5.0 <= d <= 5.5   # server önceliği + küçük jitter

    def test_backoff_exponential_and_capped(self):
        d1 = gemini._backoff_seconds(1, RuntimeError("no delay"))
        assert 4.0 <= d1 <= 4.5  # base 2.0 * 2**1
        d_big = gemini._backoff_seconds(20, RuntimeError("no delay"))
        assert d_big <= gemini.settings.gemini_retry_max_delay + 0.5  # kırpıldı

    def test_execute_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
        n = {"c": 0}

        def call():
            n["c"] += 1
            if n["c"] < 3:
                raise RuntimeError("429 rate limit exceeded")
            return "tamam"

        assert gemini._execute("attacker", call) == "tamam"
        assert n["c"] == 3   # 2 başarısız + 1 başarılı

    def test_execute_raises_quota_after_exhaust(self, monkeypatch):
        monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
        with pytest.raises(gemini.QuotaExceededError):
            gemini._execute("attacker", lambda: (_ for _ in ()).throw(
                RuntimeError("429 RESOURCE_EXHAUSTED")))

    def test_execute_non_429_propagates(self, monkeypatch):
        monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
        with pytest.raises(ValueError):
            gemini._execute("judge", lambda: (_ for _ in ()).throw(ValueError("başka")))

    def test_throttle_noop_when_rpm_zero(self):
        # Varsayılan RPM=0 → beklemeden döner (yavaşlatmaz).
        gemini._throttle()  # exception atmamalı


# --- Kota hatalı turun öğrenmeden hariç tutulması (B1) ------------------------

from agents.judge import judge_node
from agents.target import GeminiTarget, TargetResponse
from agents import target as target_mod


class TestInfraErrorHandling:
    def test_judge_skips_infra_error(self):
        state = {"pending_attempt": {"round": 1, "category": "role-playing",
                                     "infra_error": True}}
        out = judge_node(state)
        assert out["pending_attempt"] == {}
        assert out["errored_attacks"][0].get("infra_error") is True
        assert out["campaign_active"] is False
        # ÖĞRENMEYE yazılmamalı:
        assert "attack_history" not in out
        assert "failed_attacks" not in out
        assert "successful_attacks" not in out

    def test_target_marks_infra_error(self, monkeypatch):
        monkeypatch.setattr(gemini, "generate", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("429 RESOURCE_EXHAUSTED")))
        r = GeminiTarget("sp").generate("saldır")
        assert r.infra_error is True and not r.ok

    def test_target_node_infra_skips_conversation(self, monkeypatch):
        class _FakeTarget:
            def generate_conversation(self, conv):
                return TargetResponse(text="", model="m", error="429", infra_error=True)

        monkeypatch.setattr(target_mod, "build_target", lambda sp: _FakeTarget())
        state = {"pending_attempt": {"prompt": "x"}, "conversation": [],
                 "target_system": "sp"}
        out = target_mod.target_node(state)
        assert out["pending_attempt"]["infra_error"] is True
        assert "conversation" not in out  # sahte model turu eklenmedi


# --- security: API-key auth + rate limiting -----------------------------------

import time
from types import SimpleNamespace
from fastapi import HTTPException
import security


class TestApiKeyAuth:
    def test_disabled_when_unset(self, monkeypatch):
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key=""))
        assert security.require_api_key(x_api_key=None) is None  # açık geçer

    def test_rejects_missing_key(self, monkeypatch):
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key="gizli"))
        with pytest.raises(HTTPException) as e:
            security.require_api_key(x_api_key=None)
        assert e.value.status_code == 401

    def test_rejects_wrong_key(self, monkeypatch):
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key="gizli"))
        with pytest.raises(HTTPException) as e:
            security.require_api_key(x_api_key="yanlis")
        assert e.value.status_code == 401

    def test_accepts_correct_key(self, monkeypatch):
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key="gizli"))
        assert security.require_api_key(x_api_key="gizli") is None


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = security.SlidingWindowRateLimiter(max_per_minute=3)
        for _ in range(3):
            rl.check("1.2.3.4")  # limit içinde — raise yok

    def test_blocks_over_limit(self):
        rl = security.SlidingWindowRateLimiter(max_per_minute=2)
        rl.check("ip")
        rl.check("ip")
        with pytest.raises(HTTPException) as e:
            rl.check("ip")
        assert e.value.status_code == 429
        assert "Retry-After" in e.value.headers

    def test_per_client_isolated(self):
        rl = security.SlidingWindowRateLimiter(max_per_minute=1)
        rl.check("client-a")
        rl.check("client-b")  # farklı istemci → etkilenmez

    def test_window_slides(self):
        # Pencere çok kısa: eski vuruş düşünce yeni istek kabul edilir.
        rl = security.SlidingWindowRateLimiter(max_per_minute=1, window=0.05)
        rl.check("ip")
        time.sleep(0.06)
        rl.check("ip")  # eski vuruş penceden çıktı → raise yok

    def test_zero_disables_limit(self):
        rl = security.SlidingWindowRateLimiter(max_per_minute=0)
        for _ in range(100):
            rl.check("ip")  # 0 = kapalı


class TestApiSecurityIntegration:
    def test_scan_401_when_key_configured(self, monkeypatch):
        # Auth açıkken anahtarsız /scan isteği 401 almalı.
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key="gizli"))
        from fastapi.testclient import TestClient
        from api import app
        client = TestClient(app)
        r = client.post("/scan", json={"target_system": "x" * 20})
        assert r.status_code == 401

    def test_health_open_without_key(self, monkeypatch):
        monkeypatch.setattr(security, "settings", SimpleNamespace(api_key="gizli"))
        from fastapi.testclient import TestClient
        from api import app
        client = TestClient(app)
        assert client.get("/health").status_code == 200  # /health auth'suz
