"""
Тесты для claude_client.py.

Запуск:
  python test_claude_client.py              # только unit-тесты (без реального API)
  python test_claude_client.py --live       # реальный вызов Claude API
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

from analyzer import AnalysisResult, BenchmarkDelta, MetricTrend, Quartile
from benchmarks_client import BenchmarkData, Percentiles
from claude_client import (
    ClaudeCoach,
    _confidence,
    _flags,
    build_context,
    MIN_GAMES_HIGH_CONF,
)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
_failures: list[str] = []


def ok(cond: bool, label: str) -> None:
    if cond:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}")
        _failures.append(label)


def eq(a, b, label: str) -> None:
    ok(a == b, f"{label}  (got {a!r}, expected {b!r})" if a != b else label)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    games_used: int = 15,
    games_analyzed: int = 16,
    rank_dir: str | None = None,
    patch_changed: bool = False,
) -> AnalysisResult:
    def trend(direction="stable") -> MetricTrend:
        return MetricTrend(rolling_10=7.0, rolling_20=7.0, direction=direction)

    def delta(metric, pv, quartile=Quartile.BELOW) -> BenchmarkDelta:
        return BenchmarkDelta(
            metric=metric,
            player_value=pv,
            benchmark_p25=5.5,
            benchmark_p50=7.0,
            benchmark_p75=8.5,
            quartile=quartile,
            delta_vs_median=pv - 7.0,
        )

    return AnalysisResult(
        summoner="Faker#KR1",
        region="asia",
        role="BOTTOM",
        tier="GOLD",
        patch="15.10",
        games_analyzed=games_analyzed,
        games_used=games_used,
        outlier_games=games_analyzed - games_used,
        game_stats=[],
        trends={
            "cs_per_min":         trend("improving"),
            "vision_per_min":     trend("declining"),
            "deaths":             trend("stable"),
            "kill_participation": trend("stable"),
            "damage_share":       trend("stable"),
        },
        benchmark_deltas={
            "cs_per_min":         delta("cs_per_min", 6.0, Quartile.BELOW),
            "vision_per_min":     delta("vision_per_min", 0.5, Quartile.BOTTOM),
            "deaths":             delta("deaths", 5.0, Quartile.BOTTOM),
            "kill_participation": delta("kill_participation", 60.0, Quartile.ABOVE),
        },
        rank_changed=rank_dir is not None,
        rank_direction=rank_dir,
        patch_changed=patch_changed,
        summary={
            "cs_per_min": 6.0,
            "vision_per_min": 0.5,
            "deaths_per_game": 5.0,
            "kill_participation": 60.0,
            "damage_share": 22.0,
            "winrate": 48.5,
            "solo_deaths_early_avg": 1.2,
        },
    )


def _make_benchmark(source: str = "opgg", stale: bool = False) -> BenchmarkData:
    return BenchmarkData(
        champion="Jinx",
        role="BOTTOM",
        tier="GOLD",
        patch="15.10",
        source=source,
        sample_size=120_000,
        winrate=50.5,
        cs_per_min=Percentiles(5.5, 7.0, 8.5),
        vision_score_per_min=Percentiles(0.65, 0.95, 1.30),
        deaths_per_game=Percentiles(2.4, 4.1, 6.0),
        kill_participation=Percentiles(42.0, 54.0, 66.0),
        scraped_at=time.time(),
        stale=stale,
    )


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

def test_context_contains_player_header() -> None:
    ctx = build_context(_make_result(), _make_benchmark(), [])
    ok("Faker#KR1" in ctx,     "context: summoner name present")
    ok("BOTTOM" in ctx,        "context: role present")
    ok("GOLD" in ctx,          "context: tier present")
    ok("15.10" in ctx,         "context: patch present")


def test_context_contains_stats() -> None:
    ctx = build_context(_make_result(), _make_benchmark(), [])
    ok("CS/min" in ctx,        "context: CS/min stat present")
    ok("Vision/min" in ctx,    "context: Vision/min stat present")
    ok("Deaths" in ctx,        "context: Deaths stat present")
    ok("KP%" in ctx,           "context: KP% stat present")


def test_context_contains_trends() -> None:
    ctx = build_context(_make_result(), _make_benchmark(), [])
    ok("TRENDS" in ctx,        "context: TRENDS section present")
    ok("improving" in ctx,     "context: trend direction 'improving' present")
    ok("declining" in ctx,     "context: trend direction 'declining' present")


def test_context_contains_mistakes() -> None:
    mistakes = [
        {"metric": "cs_per_min", "description": "CS низкий",
         "severity": "moderate", "sessions_present": 3, "sessions_absent": 0},
    ]
    ctx = build_context(_make_result(), _make_benchmark(), mistakes)
    ok("OPEN MISTAKES" in ctx, "context: mistakes section present")
    ok("cs_per_min" in ctx,    "context: mistake metric present")
    ok("CS низкий" in ctx,     "context: mistake description present")


def test_context_no_mistakes_label() -> None:
    ctx = build_context(_make_result(), _make_benchmark(), [])
    ok("OPEN MISTAKES: none" in ctx, "context: 'none' label when no mistakes")


def test_context_rank_up_flag() -> None:
    ctx = build_context(_make_result(rank_dir="up"), _make_benchmark(), [])
    ok("RANK_UP" in ctx, "context: RANK_UP tag when rank_direction=up")


def test_context_patch_changed_flag() -> None:
    ctx = build_context(_make_result(patch_changed=True), _make_benchmark(), [])
    ok("PATCH_CHANGED" in ctx, "context: PATCH_CHANGED tag when patch_changed=True")


def test_context_benchmark_source() -> None:
    ctx = build_context(_make_result(), _make_benchmark(source="static"), [])
    ok("BENCHMARK_SOURCE: static" in ctx, "context: benchmark source shown")


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------

def test_confidence_full_data() -> None:
    conf = _confidence(_make_result(games_used=20), _make_benchmark())
    eq(conf, 1.0, "confidence: 20 clean games, live source -> 1.0")


def test_confidence_few_games() -> None:
    conf = _confidence(_make_result(games_used=3), _make_benchmark())
    ok(conf < 0.6, f"confidence: 3 games -> low (<0.6, got {conf})")


def test_confidence_static_benchmark() -> None:
    conf = _confidence(_make_result(games_used=20), _make_benchmark(source="static"))
    ok(conf < 1.0, f"confidence: static benchmark -> <1.0 (got {conf})")
    ok(conf <= 0.75, f"confidence: static benchmark -> <=0.75 (got {conf})")


def test_confidence_stale_benchmark() -> None:
    conf = _confidence(_make_result(), _make_benchmark(stale=True))
    ok(conf < 1.0, f"confidence: stale benchmark -> <1.0 (got {conf})")


def test_confidence_many_outliers() -> None:
    # 40 % outliers
    conf = _confidence(
        _make_result(games_used=6, games_analyzed=10),
        _make_benchmark(),
    )
    ok(conf < 1.0, f"confidence: 40% outliers -> <1.0 (got {conf})")


def test_confidence_clamped_to_zero() -> None:
    conf = _confidence(
        _make_result(games_used=2, games_analyzed=10),
        _make_benchmark(source="static", stale=True),
    )
    ok(0.0 <= conf <= 1.0, f"confidence: always in [0,1] (got {conf})")


# ---------------------------------------------------------------------------
# _flags
# ---------------------------------------------------------------------------

def test_flags_rank_up() -> None:
    fl = _flags(_make_result(rank_dir="up"), _make_benchmark())
    ok("rank_up" in fl, "flags: rank_up when promoted")


def test_flags_rank_down() -> None:
    fl = _flags(_make_result(rank_dir="down"), _make_benchmark())
    ok("rank_down" in fl, "flags: rank_down when demoted")


def test_flags_patch_changed() -> None:
    fl = _flags(_make_result(patch_changed=True), _make_benchmark())
    ok("patch_changed" in fl, "flags: patch_changed when patch changed")


def test_flags_stale_data() -> None:
    fl = _flags(_make_result(), _make_benchmark(stale=True))
    ok("stale_data" in fl, "flags: stale_data when benchmark stale")


def test_flags_static_benchmark() -> None:
    fl = _flags(_make_result(), _make_benchmark(source="static"))
    ok("static_benchmark" in fl, "flags: static_benchmark when source=static")


def test_flags_empty_when_clean() -> None:
    fl = _flags(_make_result(), _make_benchmark())
    eq(fl, [], "flags: empty list when everything is clean")


# ---------------------------------------------------------------------------
# ClaudeCoach.coach (mocked API)
# ---------------------------------------------------------------------------

def _mock_claude_response(payload: dict) -> MagicMock:
    """Создаёт MagicMock, имитирующий ответ Anthropic SDK."""
    content_block = MagicMock()
    content_block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [content_block]
    return response


def test_coach_returns_structured_dict() -> None:
    """ClaudeCoach.coach должен возвращать dict с нужными ключами."""
    fake_payload = {
        "primary_focus": "Improve wave control",
        "summary": "Your CS is improving but vision is weak.",
        "flags": [],
        "coaching_points": [
            {
                "metric": "vision_per_min",
                "quartile": "bottom",
                "trend": "declining",
                "suggestion": "Buy control wards every back.",
            }
        ],
        "confidence": 0.9,
    }

    with patch("claude_client.anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.return_value = _mock_claude_response(fake_payload)

        coach = ClaudeCoach(api_key="test-key")
        result = coach.coach(_make_result(), _make_benchmark(), [])

    ok("primary_focus"   in result, "coach: primary_focus key present")
    ok("summary"         in result, "coach: summary key present")
    ok("coaching_points" in result, "coach: coaching_points key present")
    ok("confidence"      in result, "coach: confidence key present")
    eq(result["primary_focus"], "Improve wave control", "coach: primary_focus value")
    eq(result["confidence"], 0.9, "coach: confidence value")


def test_coach_strips_markdown_wrapper() -> None:
    """Если Claude оборачивает ответ в ```json ... ```, клиент должен снять обёртку."""
    fake_payload = {"primary_focus": "x", "summary": "y",
                    "flags": [], "coaching_points": [], "confidence": 0.5}
    raw_with_md = f"```json\n{json.dumps(fake_payload)}\n```"

    content_block = MagicMock()
    content_block.text = raw_with_md
    response = MagicMock()
    response.content = [content_block]

    with patch("claude_client.anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.return_value = response

        coach = ClaudeCoach(api_key="test-key")
        result = coach.coach(_make_result(), _make_benchmark(), [])

    eq(result["primary_focus"], "x", "coach: markdown wrapper stripped")


def test_coach_raises_on_invalid_json() -> None:
    """Невалидный JSON от Claude → ValueError."""
    content_block = MagicMock()
    content_block.text = "This is not JSON at all."
    response = MagicMock()
    response.content = [content_block]

    with patch("claude_client.anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.return_value = response

        coach = ClaudeCoach(api_key="test-key")
        raised = False
        try:
            coach.coach(_make_result(), _make_benchmark(), [])
        except ValueError:
            raised = True
    ok(raised, "coach: ValueError on invalid JSON response")


def test_coach_passes_model_and_max_tokens() -> None:
    """Убеждаемся, что клиент передаёт нужную модель и max_tokens в API."""
    from claude_client import MODEL, MAX_TOKENS

    fake_payload = {"primary_focus": "x", "summary": "y",
                    "flags": [], "coaching_points": [], "confidence": 0.5}

    with patch("claude_client.anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        instance.messages.create.return_value = _mock_claude_response(fake_payload)

        coach = ClaudeCoach(api_key="test-key")
        coach.coach(_make_result(), _make_benchmark(), [])

        call_kwargs = instance.messages.create.call_args.kwargs
        eq(call_kwargs["model"],      MODEL,      "coach: correct model used")
        eq(call_kwargs["max_tokens"], MAX_TOKENS, "coach: correct max_tokens")


# ---------------------------------------------------------------------------
# Live probe (--live flag)
# ---------------------------------------------------------------------------

async def run_live() -> None:
    """Реальный вызов Claude API. Требует ANTHROPIC_API_KEY в .env."""
    import asyncio
    from analyzer import GameStats, MetricTrend, BenchmarkDelta, Quartile

    print("\n--- Live Claude API probe ---")
    result  = _make_result()
    bench   = _make_benchmark()
    mistakes = [
        {"metric": "vision_per_min", "description": "Vision score очень низкий",
         "severity": "major", "sessions_present": 5, "sessions_absent": 0},
    ]

    coach = ClaudeCoach()
    try:
        fb = coach.coach(result, bench, mistakes)
        print("\nClaude response:")
        print(json.dumps(fb, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    if "--live" in sys.argv:
        import asyncio
        asyncio.run(run_live())
        return

    print("\n=== claude_client.py - unit tests ===\n")

    # build_context
    test_context_contains_player_header()
    test_context_contains_stats()
    test_context_contains_trends()
    test_context_contains_mistakes()
    test_context_no_mistakes_label()
    test_context_rank_up_flag()
    test_context_patch_changed_flag()
    test_context_benchmark_source()

    # _confidence
    test_confidence_full_data()
    test_confidence_few_games()
    test_confidence_static_benchmark()
    test_confidence_stale_benchmark()
    test_confidence_many_outliers()
    test_confidence_clamped_to_zero()

    # _flags
    test_flags_rank_up()
    test_flags_rank_down()
    test_flags_patch_changed()
    test_flags_stale_data()
    test_flags_static_benchmark()
    test_flags_empty_when_clean()

    # ClaudeCoach
    test_coach_returns_structured_dict()
    test_coach_strips_markdown_wrapper()
    test_coach_raises_on_invalid_json()
    test_coach_passes_model_and_max_tokens()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll tests passed.\033[0m")


if __name__ == "__main__":
    main()
