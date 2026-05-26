"""
Claude AI coaching client.

Строит контекстный промпт из AnalysisResult + активных ошибок,
вызывает Claude API и возвращает структурированный JSON с советами.

Выходной формат
---------------
{
  "primary_focus":   str,          # главное, что нужно улучшить
  "summary":         str,          # 2–3 предложения об общем состоянии игры
  "flags":           list[str],    # ["rank_up"|"rank_down"|"patch_changed"|"stale_data"]
  "coaching_points": [             # до 4 конкретных совета
    {
      "metric":      str,          # "cs_per_min" | "vision_per_min" | ...
      "quartile":    str,          # "top"|"above"|"below"|"bottom"
      "trend":       str,          # "improving"|"declining"|"stable"|"insufficient_data"
      "suggestion":  str           # конкретный, действенный совет (1–2 предложения)
    }
  ],
  "confidence":      float         # 0.0–1.0, на основе объёма данных
}

Модель: claude-sonnet-4-20250514
Лимит токенов ответа: 1 200
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import anthropic
from dotenv import load_dotenv

from analyzer import AnalysisResult
from benchmarks_client import BenchmarkData

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 1_200
MIN_GAMES_HIGH_CONF = 15    # games_used >= N → confidence высокий

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _fmt_pct(label: str, delta, player_val: float) -> str:
    """Форматирует одну строку статистики с перцентилями."""
    return (
        f"  {label}: {player_val:.2f}"
        f"  [p25={delta.benchmark_p25}, p50={delta.benchmark_p50}, p75={delta.benchmark_p75}]"
        f"  → {delta.quartile.value.upper()}"
    )


def _fmt_trend(label: str, trend) -> str:
    r10 = f"{trend.rolling_10:.2f}" if trend.rolling_10 is not None else "n/a"
    r20 = f"{trend.rolling_20:.2f}" if trend.rolling_20 is not None else "n/a"
    return f"  {label}: {trend.direction}  (r10={r10}, r20={r20})"


def build_prev_coaching_block(
    prev_log: dict,
    new_games_count: int,
    current_summary: dict,
) -> str:
    """
    Формирует блок с предыдущим советом Claude для follow-up контекста.
    Вызывается только если new_games_count > 0.
    """
    import datetime
    advice      = prev_log["advice_json"]
    prev_stats  = prev_log["stats_json"]
    dt          = datetime.datetime.fromtimestamp(prev_log["created_at"]).strftime("%d.%m.%Y")

    lines = [
        f"ПРЕДЫДУЩИЙ СОВЕТ (от {dt}, с тех пор сыграно {new_games_count} новых игр):",
        f"  Главный фокус: {advice.get('primary_focus', '—')}",
        "",
        "  Статистика НА МОМЕНТ совета vs СЕЙЧАС:",
    ]

    metrics = [
        ("cs_per_min",         "CS/min"),
        ("vision_per_min",     "Vision/min"),
        ("deaths_per_game",    "Deaths"),
        ("kill_participation", "KP%"),
    ]
    for key, label in metrics:
        prev_v = prev_stats.get(key)
        curr_v = current_summary.get(key)
        if prev_v is not None and curr_v is not None:
            delta = curr_v - prev_v
            sign  = "+" if delta >= 0 else ""
            lines.append(f"    {label:14s}: было {prev_v:.2f} → сейчас {curr_v:.2f}  ({sign}{delta:.2f})")

    pts = advice.get("coaching_points", [])
    if pts:
        lines.append("")
        lines.append("  Рекомендации из прошлой сессии:")
        for pt in pts[:3]:
            lines.append(f"    [{pt['metric']}] {pt['suggestion'][:120]}")

    lines.append("")
    lines.append("Оцени выполнение рекомендаций в поле \"follow_up\".")
    return "\n".join(lines)


def build_context(
    result: AnalysisResult,
    benchmark: BenchmarkData,
    active_mistakes: list[dict],
    prev_coaching: Optional[dict] = None,
    new_games_since_prev: int = 0,
) -> str:
    """
    Формирует компактный текстовый контекст для промпта Claude.
    Возвращает строку, умещающуюся в ≈600 токенов.
    """
    bd = result.benchmark_deltas
    tr = result.trends

    # --- Заголовок ---
    lines = [
        f"PLAYER: {result.summoner} | {result.region} | {result.role} | {result.tier}",
        f"PATCH: {result.patch}"
        + (" | PATCH_CHANGED: YES" if result.patch_changed else "")
        + (" | RANK_UP" if result.rank_direction == "up" else "")
        + (" | RANK_DOWN" if result.rank_direction == "down" else ""),
        f"GAMES: {result.games_analyzed} total | {result.games_used} clean"
        + (f" | {result.outlier_games} outliers excluded" if result.outlier_games else ""),
        f"BENCHMARK_SOURCE: {benchmark.source}"
        + (f" | SAMPLE: {benchmark.sample_size:,}" if benchmark.sample_size else ""),
        "",
    ]

    # --- Статистика vs бенчмарк ---
    lines.append("STATS (player rolling-10 vs tier benchmark percentiles):")
    if "cs_per_min" in bd:
        lines.append(_fmt_pct("CS/min", bd["cs_per_min"], result.summary.get("cs_per_min", 0)))
    if "vision_per_min" in bd:
        lines.append(_fmt_pct("Vision/min", bd["vision_per_min"], result.summary.get("vision_per_min", 0)))
    if "deaths" in bd:
        lines.append(_fmt_pct("Deaths/game", bd["deaths"], result.summary.get("deaths_per_game", 0)))
    if "kill_participation" in bd:
        lines.append(_fmt_pct("KP%", bd["kill_participation"], result.summary.get("kill_participation", 0)))

    wr    = result.summary.get("winrate", 0)
    sde   = result.summary.get("solo_deaths_early_avg", 0)
    ds    = result.summary.get("damage_share", 0)
    lines.append(f"  Winrate: {wr:.1f}%  |  Solo-deaths<10min avg: {sde:.2f}  |  Damage share: {ds:.1f}%")
    lines.append("")

    # --- Тренды ---
    lines.append("TRENDS (direction of change):")
    if "cs_per_min" in tr:
        lines.append(_fmt_trend("CS/min", tr["cs_per_min"]))
    if "vision_per_min" in tr:
        lines.append(_fmt_trend("Vision/min", tr["vision_per_min"]))
    if "deaths" in tr:
        lines.append(_fmt_trend("Deaths (inverted, improving=less)", tr["deaths"]))
    if "kill_participation" in tr:
        lines.append(_fmt_trend("KP%", tr["kill_participation"]))
    lines.append("")

    # --- Активные ошибки ---
    if active_mistakes:
        lines.append(f"OPEN MISTAKES ({len(active_mistakes)}):")
        for m in active_mistakes[:5]:
            sev     = m.get("severity", "minor")
            present = m.get("sessions_present", 1)
            absent  = m.get("sessions_absent", 0)
            lines.append(
                f"  [{sev.upper()} | {present} sessions present | {absent} absent]"
                f" {m['metric']}: {m['description']}"
            )
    else:
        lines.append("OPEN MISTAKES: none")

    # Блок предыдущего совета (только если были новые игры)
    if prev_coaching is not None and new_games_since_prev > 0:
        lines.append("")
        lines.append("=" * 60)
        lines.append(build_prev_coaching_block(
            prev_coaching, new_games_since_prev, result.summary
        ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _confidence(result: AnalysisResult, benchmark: BenchmarkData) -> float:
    """
    Оценка достоверности (0.0–1.0).

    Факторы снижения:
      • мало чистых игр (< MIN_GAMES_HIGH_CONF)
      • источник бенчмарка — статика (source="static")
      • данные бенчмарка устарели (stale=True)
      • много выбросов (> 30 % матчей)
    """
    score = 1.0

    # Объём данных
    if result.games_used < 5:
        score -= 0.5
    elif result.games_used < MIN_GAMES_HIGH_CONF:
        score -= 0.2

    # Источник бенчмарка
    if benchmark.source == "static":
        score -= 0.25
    elif benchmark.stale:
        score -= 0.10

    # Много выбросов
    if result.games_analyzed > 0:
        outlier_ratio = result.outlier_games / result.games_analyzed
        if outlier_ratio > 0.3:
            score -= 0.15

    return round(max(0.0, min(1.0, score)), 2)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _flags(result: AnalysisResult, benchmark: BenchmarkData) -> list[str]:
    f: list[str] = []
    if result.rank_direction == "up":
        f.append("rank_up")
    elif result.rank_direction == "down":
        f.append("rank_down")
    if result.patch_changed:
        f.append("patch_changed")
    if benchmark.stale:
        f.append("stale_data")
    if benchmark.source == "static":
        f.append("static_benchmark")
    return f


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
Ты эксперт-тренер по League of Legends.
Получаешь структурированные данные о производительности игрока и даёшь конкретные советы в формате JSON.
Общайся с игроком на русском языке.

Правила:
- Отвечай ТОЛЬКО валидным JSON — без markdown, без преамбул, без лишнего текста.
- coaching_points: до 4 пунктов, начиная с самого проблемного показателя.
- Советы должны быть конкретными и механическими (не «фармить лучше», а «закупать контрол-варды каждую базу»).
- Если метрика в квартиле TOP — не включай её в coaching_points.
- primary_focus: одна самая важная вещь для улучшения прямо сейчас.
- summary: 2–3 предложения, честно и конструктивно.
- Если confidence < 0.5 — упомяни недостаточность данных в summary.
- follow_up: если в контексте есть раздел ПРЕДЫДУЩИЙ СОВЕТ — оцени прогресс по рекомендациям.
  Скажи что улучшилось, что нет, и насколько хорошо игрок выполнил советы.
  Если предыдущего совета нет — верни null.

Схема ответа (строгая):
{
  "primary_focus":   "<короткая фраза>",
  "summary":         "<2-3 предложения>",
  "flags":           ["<флаг>", ...],
  "follow_up":       "<оценка выполнения прошлых рекомендаций или null>",
  "coaching_points": [
    {
      "metric":     "<название метрики>",
      "quartile":   "<top|above|below|bottom>",
      "trend":      "<improving|declining|stable|insufficient_data>",
      "suggestion": "<конкретный совет, 1-2 предложения>"
    }
  ],
  "confidence": <0.0-1.0>
}
"""

# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class ClaudeCoach:
    """
    Вызывает Claude API для генерации персонализированного коучинга.

    Usage:
        coach = ClaudeCoach()
        feedback = await coach.coach(result, benchmark, active_mistakes)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def coach(
        self,
        result: AnalysisResult,
        benchmark: BenchmarkData,
        active_mistakes: list[dict],
        prev_coaching: Optional[dict] = None,
        new_games_since_prev: int = 0,
    ) -> dict:
        """
        Синхронный вызов Claude API.
        Возвращает структурированный dict с коучинговыми советами.
        Выбрасывает исключение при ошибке API или невалидном JSON.
        """
        context  = build_context(result, benchmark, active_mistakes,
                                 prev_coaching, new_games_since_prev)
        conf     = _confidence(result, benchmark)
        fl       = _flags(result, benchmark)

        human_msg = (
            f"{context}\n\n"
            f"PRECOMPUTED_FLAGS: {json.dumps(fl)}\n"
            f"PRECOMPUTED_CONFIDENCE: {conf}\n\n"
            "Дай коучинговый совет в формате JSON."
        )

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": human_msg}],
        )

        raw = response.content[0].text.strip()

        # Защита от markdown-обёртки: ```json ... ```
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned non-JSON response: {raw[:200]}") from exc

    async def coach_async(
        self,
        result: AnalysisResult,
        benchmark: BenchmarkData,
        active_mistakes: list[dict],
        prev_coaching: Optional[dict] = None,
        new_games_since_prev: int = 0,
    ) -> dict:
        """
        Асинхронная обёртка для FastAPI-обработчиков.
        Использует AsyncAnthropic под капотом.
        """
        import asyncio
        from anthropic import AsyncAnthropic

        context  = build_context(result, benchmark, active_mistakes,
                                 prev_coaching, new_games_since_prev)
        conf     = _confidence(result, benchmark)
        fl       = _flags(result, benchmark)

        human_msg = (
            f"{context}\n\n"
            f"PRECOMPUTED_FLAGS: {json.dumps(fl)}\n"
            f"PRECOMPUTED_CONFIDENCE: {conf}\n\n"
            "Дай коучинговый совет в формате JSON."
        )

        async_client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        response = await async_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": human_msg}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned non-JSON response: {raw[:200]}") from exc
