"""
FastAPI backend для LoL AI Coaching Service.

Эндпоинты
----------
  GET  /analyze           — полный анализ игрока + коучинг Claude
  GET  /mistakes          — активные ошибки игрока
  GET  /benchmarks        — тировые бенчмарки для чемпиона/роли
  POST /mistakes/resolve  — ручной резолв ошибки

Особенности
-----------
  • role=ALL       → анализ всех ролей, сравнение результативности
  • Кэш            → если не появилось новых игр — возвращает кэш без API-вызовов
  • Follow-up      → если игрок сыграл игры после прошлого совета — Claude оценивает прогресс
  • Logging        → каждый совет сохраняется в coaching_log для следующего follow-up
"""

from __future__ import annotations

import json
import os
import statistics
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(override=True)

from analyzer import analyze, _extract_game_stats
from benchmarks_client import BenchmarksClient, StaticBenchmarkProvider
from claude_client import ClaudeCoach
from db import (
    get_active_mistakes,
    get_cached_analysis,
    get_last_coaching_log,
    get_player_by_summoner,
    get_player_rank,
    init_db,
    process_analysis_mistakes,
    save_analysis_cache,
    save_coaching_log,
    upsert_player,
)
from riot_client import RiotClient

DB_PATH = os.getenv("DB_PATH", "lol_coaching.db")

ROLE_LABELS = {
    "TOP": "Топ", "JUNGLE": "Джунгли", "MIDDLE": "Мид",
    "BOTTOM": "Бот", "UTILITY": "Сапорт",
}

# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

class _State:
    db     = None
    riot   = None
    bench  = None
    claude = None

state = _State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.db     = init_db(DB_PATH)
    state.riot   = RiotClient()
    state.bench  = BenchmarksClient(db_path=DB_PATH)
    state.claude = ClaudeCoach()
    yield
    await state.bench.close()
    await state.riot.close()
    state.db.close()


app = FastAPI(title="LoL AI Coaching API", version="0.2.0", lifespan=lifespan)

_cors_extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://smokebellow.github.io",
        *_cors_extra,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_dict(p) -> dict:
    return {"p25": p.p25, "p50": p.p50, "p75": p.p75}


def _benchmark_to_dict(bd) -> dict:
    return {
        "champion":             bd.champion,
        "role":                 bd.role,
        "tier":                 bd.tier,
        "patch":                bd.patch,
        "source":               bd.source,
        "sample_size":          bd.sample_size,
        "winrate":              bd.winrate,
        "stale":                bd.stale,
        "cs_per_min":           _pct_dict(bd.cs_per_min),
        "vision_score_per_min": _pct_dict(bd.vision_score_per_min),
        "deaths_per_game":      _pct_dict(bd.deaths_per_game),
        "kill_participation":   _pct_dict(bd.kill_participation),
    }


def _delta_to_dict(d) -> dict:
    return {
        "metric":          d.metric,
        "player_value":    d.player_value,
        "benchmark_p25":   d.benchmark_p25,
        "benchmark_p50":   d.benchmark_p50,
        "benchmark_p75":   d.benchmark_p75,
        "quartile":        d.quartile.value,
        "delta_vs_median": d.delta_vs_median,
    }


def _trend_to_dict(t) -> dict:
    return {"rolling_10": t.rolling_10, "rolling_20": t.rolling_20, "direction": t.direction}


def _detect_flagged_mistakes(result, benchmark) -> list[dict]:
    """Определяет ошибки по квартильной позиции."""
    from analyzer import Quartile
    labels = {
        "cs_per_min":         "CS ниже медианы уровня",
        "vision_per_min":     "Vision score ниже медианы уровня",
        "deaths":             "Смертей больше медианы уровня",
        "kill_participation": "Kill participation ниже медианы уровня",
    }
    flagged = []
    for metric, delta in result.benchmark_deltas.items():
        if delta.quartile in (Quartile.BOTTOM, Quartile.BELOW):
            severity = "major" if delta.quartile == Quartile.BOTTOM else "minor"
            flagged.append({
                "metric":      metric,
                "description": labels.get(metric, f"{metric} ниже медианы"),
                "severity":    severity,
            })
    return flagged


def _quartile_score(q: str) -> int:
    """Числовой балл квартиля для сравнения ролей: top=3, above=2, below=1, bottom=0."""
    return {"top": 3, "above": 2, "below": 1, "bottom": 0}.get(q, 1)


def _build_analyze_response(result, benchmark, coaching, active_mistakes, rank, cached: bool) -> dict:
    rank_dict = None
    if rank:
        rank_dict = {"tier": rank.tier, "division": rank.division, "lp": rank.lp}

    return {
        "summoner":         result.summoner,
        "region":           result.region,
        "role":             result.role,
        "tier":             result.tier,
        "patch":            result.patch,
        "rank":             rank_dict,
        "games_analyzed":   result.games_analyzed,
        "games_used":       result.games_used,
        "outlier_games":    result.outlier_games,
        "rank_changed":     result.rank_changed,
        "rank_direction":   result.rank_direction,
        "patch_changed":    result.patch_changed,
        "summary":          result.summary,
        "trends":           {k: _trend_to_dict(v) for k, v in result.trends.items()},
        "benchmark_deltas": {k: _delta_to_dict(v) for k, v in result.benchmark_deltas.items()},
        "benchmark":        _benchmark_to_dict(benchmark),
        "coaching":         coaching,
        "active_mistakes":  active_mistakes,
        "from_cache":       cached,
    }


# ---------------------------------------------------------------------------
# All-roles analysis
# ---------------------------------------------------------------------------

async def _analyze_all_roles(summoner: str, region: str, count: int) -> dict:
    """
    Загружает count игр, группирует по роли, считает статистику для каждой.
    Без Claude — только цифры и сравнение.
    """
    try:
        player = await state.riot.fetch_player_data(summoner, region, count=max(count, 30))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    role_games: dict[str, list] = {}
    for match in player.matches:
        gs = _extract_game_stats(match, player.puuid)
        if gs and gs.role:
            role_games.setdefault(gs.role, []).append(gs)

    role_summary = {}
    for role, games in role_games.items():
        if len(games) < 2:
            continue
        wins    = sum(1 for g in games if g.win)
        cs_vals = [g.cs_per_min for g in games]
        vs_vals = [g.vision_per_min for g in games]
        kp_vals = [g.kill_participation for g in games]
        dt_vals = [float(g.deaths) for g in games]

        role_summary[role] = {
            "role_label":      ROLE_LABELS.get(role, role),
            "games":           len(games),
            "winrate":         round(wins / len(games) * 100, 1),
            "cs_per_min":      round(statistics.mean(cs_vals), 2),
            "vision_per_min":  round(statistics.mean(vs_vals), 2),
            "deaths_per_game": round(statistics.mean(dt_vals), 2),
            "kill_participation": round(statistics.mean(kp_vals), 1),
        }

    # Определяем «лучшую» роль по winrate (при ≥5 играх), иначе по KP
    ranked = sorted(
        [(r, s) for r, s in role_summary.items() if s["games"] >= 3],
        key=lambda x: (x[1]["winrate"], x[1]["kill_participation"]),
        reverse=True,
    )
    best_role = ranked[0][0] if ranked else None

    rank_dict = None
    if player.rank:
        rank_dict = {"tier": player.rank.tier, "division": player.rank.division, "lp": player.rank.lp}

    return {
        "mode":         "all_roles",
        "summoner":     player.summoner_name,
        "region":       player.region,
        "rank":         rank_dict,
        "games_total":  sum(s["games"] for s in role_summary.values()),
        "role_summary": role_summary,
        "best_role":    best_role,
    }


# ---------------------------------------------------------------------------
# Single-role full analysis
# ---------------------------------------------------------------------------

async def _analyze_single_role(
    summoner: str, region: str, role: str, tier: Optional[str], count: int
) -> dict:
    # 1. Резолвим puuid + получаем список match_id (быстро, без детальной загрузки)
    try:
        platform      = state.riot.resolve_platform(region)
        routing       = state.riot._http   # используем тот же http-клиент
        from riot_client import PLATFORM_TO_REGION
        routing_region = PLATFORM_TO_REGION[platform]

        # account-v1 → puuid (если Riot ID)
        if "#" in summoner:
            gn, tl = summoner.split("#", 1)
            account = await state.riot.get_account_by_riot_id(gn, tl, routing_region)
            puuid   = account["puuid"]
        else:
            summ_obj = await state.riot.get_summoner_by_name(summoner, platform)
            puuid    = summ_obj["puuid"]

        match_ids = await state.riot.get_match_ids(puuid, routing_region, count)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    newest_match = match_ids[0] if match_ids else ""
    cache_role   = role.upper()

    # 2. Проверка кэша
    cached_result = get_cached_analysis(state.db, puuid, cache_role, newest_match)
    if cached_result is not None:
        cached_result["from_cache"] = True
        return cached_result

    # 3. Полный пайплайн Riot API
    try:
        player = await state.riot.fetch_player_data(summoner, region, count)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    effective_tier = (
        tier.upper() if tier
        else (player.rank.tier if player.rank else "GOLD")
    )

    # Определяем чемпиона из последнего матча роли
    champion = "Jinx"
    for match in player.matches:
        info = match.get("info", {})
        for p in info.get("participants", []):
            if p.get("puuid") == player.puuid and p.get("teamPosition") == role.upper():
                champion = p.get("championName", "Jinx")
                break
        else:
            continue
        break

    # 4. Бенчмарки
    try:
        benchmark = await state.bench.get(champion, role.upper(), effective_tier)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Benchmark: {exc}")

    # 5. Анализ
    prev_rank = get_player_rank(state.db, puuid)
    result    = analyze(player, benchmark, previous_rank=prev_rank)

    # 6. Загружаем предыдущий coaching_log
    prev_log   = get_last_coaching_log(state.db, puuid, role.upper())
    new_games  = 0
    if prev_log and match_ids:
        prev_set  = set(prev_log["match_ids"])
        curr_set  = set(match_ids)
        new_games = len(curr_set - prev_set)

    # 7. Claude
    active_mistakes = get_active_mistakes(state.db, puuid, role.upper())
    try:
        coaching = await state.claude.coach_async(
            result, benchmark, active_mistakes,
            prev_coaching=prev_log if new_games > 0 else None,
            new_games_since_prev=new_games,
        )
    except Exception as exc:
        coaching = {
            "primary_focus": "Анализ данных",
            "summary": f"Claude временно недоступен: {exc}",
            "flags": [], "follow_up": None,
            "coaching_points": [], "confidence": 0.0,
        }

    # 8. Сохраняем в БД
    upsert_player(state.db, puuid, player.summoner_name, player.platform, player.rank)
    flagged = _detect_flagged_mistakes(result, benchmark)
    process_analysis_mistakes(state.db, puuid, role.upper(), flagged)

    # Сохраняем coaching_log (только если Claude ответил нормально)
    if coaching.get("confidence", 0) > 0:
        save_coaching_log(
            state.db, puuid, role.upper(), result.patch,
            match_ids=list(match_ids),
            advice=coaching,
            stats=result.summary,
            games_count=result.games_used,
        )

    # 9. Формируем ответ
    fresh_mistakes = get_active_mistakes(state.db, puuid, role.upper())
    response = _build_analyze_response(
        result, benchmark, coaching, fresh_mistakes, player.rank, cached=False
    )
    response["new_games_since_prev"] = new_games

    # 10. Кэшируем только если Claude ответил корректно (confidence > 0)
    if coaching.get("confidence", 0) > 0:
        save_analysis_cache(state.db, puuid, cache_role, newest_match, response)

    return response


# ---------------------------------------------------------------------------
# POST body models
# ---------------------------------------------------------------------------

class ResolveBody(BaseModel):
    mistake_id: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/analyze")
async def analyze_endpoint(
    summoner: str = Query(..., description="Riot ID (Name#TAG) или ник"),
    region:   str = Query(..., description="Регион: na, euw, kr, ..."),
    role:     str = Query("ALL", description="Роль: TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY/ALL"),
    tier:     Optional[str] = Query(None,  description="Тир (авто из ранга если не задан)"),
    count:    int = Query(20, ge=5, le=50, description="Количество матчей"),
):
    """
    Анализ игрока.

    role=ALL    → статистика по всем ролям без Claude.
    role=BOTTOM → полный анализ одной роли + Claude + кэш + follow-up.
    """
    if role.upper() == "ALL":
        return await _analyze_all_roles(summoner, region, count)
    return await _analyze_single_role(summoner, region, role, tier, count)


@app.get("/mistakes")
async def get_mistakes_endpoint(
    summoner: str = Query(..., description="Riot ID (Name#TAG)"),
    role:     Optional[str] = Query(None, description="Роль (опционально)"),
):
    """Активные ошибки игрока из локальной БД (требует предварительного /analyze)."""
    player = get_player_by_summoner(state.db, summoner)
    if player is None:
        raise HTTPException(
            status_code=404,
            detail="Игрок не найден. Сначала выполни /analyze.",
        )
    mistakes = get_active_mistakes(state.db, player["puuid"], role.upper() if role else None)
    return {"mistakes": mistakes}


@app.get("/benchmarks")
async def get_benchmark_endpoint(
    champion: str = Query(..., description="Имя чемпиона"),
    role:     str = Query(..., description="Роль"),
    tier:     str = Query("GOLD", description="Тир"),
):
    """Тировые бенчмарки для указанного чемпиона/роли (TTL 48 ч)."""
    try:
        bd = await state.bench.get(champion, role.upper(), tier.upper())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Benchmark: {exc}")
    return _benchmark_to_dict(bd)


@app.post("/mistakes/resolve")
async def resolve_mistake_endpoint(body: ResolveBody):
    """Вручную помечает ошибку как решённую."""
    from db import resolve_mistake, get_mistake
    row = get_mistake(state.db, body.mistake_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Ошибка не найдена")
    if row["resolved"]:
        raise HTTPException(status_code=409, detail="Уже решена")
    resolve_mistake(state.db, body.mistake_id)
    return {"ok": True, "mistake_id": body.mistake_id}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
