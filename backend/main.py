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

import asyncio
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
    log_request,
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

# Каскадный сбор матчей
_TARGET_ROLE_GAMES = 10   # минимум игр на роли для полноценного анализа
_BATCH_SIZE        = 20   # игр за один запрос
_MAX_TOTAL         = 100  # максимум всего матчей для поиска

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
        "champion_stats":   result.champion_stats,
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
# Cascade match fetcher
# ---------------------------------------------------------------------------

async def _cascade_fetch(
    puuid: str, routing_region: str, role: str
) -> tuple[list[str], list[dict], int, int]:
    """
    Fetch matches in batches of _BATCH_SIZE until _TARGET_ROLE_GAMES games
    on `role` are found, or _MAX_TOTAL total matches are searched.

    Returns (all_match_ids, all_matches, role_count, total_fetched).
    """
    all_match_ids: list[str] = []
    all_matches:   list[dict] = []
    role_count  = 0
    start       = 0

    while role_count < _TARGET_ROLE_GAMES and start < _MAX_TOTAL:
        this_batch = min(_BATCH_SIZE, _MAX_TOTAL - start)
        batch_ids  = await state.riot.get_match_ids(
            puuid, routing_region, this_batch, start=start
        )
        if not batch_ids:
            break   # история закончилась

        all_match_ids.extend(batch_ids)

        batch_matches = list(await asyncio.gather(
            *[state.riot.get_match(mid, routing_region) for mid in batch_ids]
        ))
        all_matches.extend(batch_matches)

        for match in batch_matches:
            for p in match.get("info", {}).get("participants", []):
                if p.get("puuid") == puuid and p.get("teamPosition", "").upper() == role:
                    role_count += 1
                    break

        start += len(batch_ids)
        if len(batch_ids) < this_batch:
            break   # Riot вернул меньше — конец истории

    return all_match_ids, all_matches, role_count, start


# ---------------------------------------------------------------------------
# Single-role full analysis
# ---------------------------------------------------------------------------

async def _analyze_single_role(
    summoner: str, region: str, role: str, tier: Optional[str], count: int
) -> dict:
    from riot_client import PLATFORM_TO_REGION, PlayerData, SummonerRank

    # 1. Резолвим puuid + display name
    try:
        platform       = state.riot.resolve_platform(region)
        routing_region = PLATFORM_TO_REGION[platform]

        if "#" in summoner:
            gn, tl   = summoner.split("#", 1)
            account  = await state.riot.get_account_by_riot_id(gn, tl, routing_region)
            puuid    = account["puuid"]
            summoner_name = f"{account['gameName']}#{account['tagLine']}"
        else:
            summ_obj = await state.riot.get_summoner_by_name(summoner, platform)
            puuid    = summ_obj["puuid"]
            summoner_name = summ_obj.get("name", summoner)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    # 2. Проверяем кэш по первому (новейшему) match_id
    try:
        first_ids    = await state.riot.get_match_ids(puuid, routing_region, 1)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    newest_match = first_ids[0] if first_ids else ""
    cache_role   = role.upper()

    cached_result = get_cached_analysis(state.db, puuid, cache_role, newest_match)
    if cached_result is not None:
        cached_result["from_cache"] = True
        return cached_result

    # 3. Ранг
    try:
        rank_entries = await state.riot.get_rank(puuid, platform)
        solo_raw     = next(
            (r for r in rank_entries if r.get("queueType") == "RANKED_SOLO_5x5"), None
        )
        rank = (
            SummonerRank(
                tier=solo_raw["tier"], division=solo_raw["rank"],
                lp=solo_raw["leaguePoints"], queue_type="RANKED_SOLO_5x5",
            ) if solo_raw else None
        )
    except Exception:
        rank = None

    # 4. Каскадный сбор матчей — ищем _TARGET_ROLE_GAMES игр на роли
    try:
        all_match_ids, all_matches, role_count, total_fetched = await _cascade_fetch(
            puuid, routing_region, role.upper()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Riot API: {exc}")

    # 4b. Фоллбэк: нет ни одной игры на роли
    if role_count == 0:
        role_label = ROLE_LABELS.get(role.upper(), role)
        return {
            "summoner":          summoner_name,
            "region":            region,
            "role":              role.upper(),
            "insufficient_data": True,
            "games_searched":    total_fetched,
            "role_games_found":  0,
            "message": (
                f"За последние {total_fetched} ранговых игр не найдено ни одной "
                f"игры на роли «{role_label}». Сыграй хотя бы несколько матчей на этой роли!"
            ),
        }

    # 5. Таймлайны (solo deaths до 10 мин)
    # Запрашиваем только для матчей НА ЦЕЛЕВОЙ РОЛИ со смертями — не для всех 100.
    # Это снижает число API-вызовов с ~100 до ~10 и ускоряет ответ в разы.
    try:
        needs_timeline: list[str] = []
        for match in all_matches:
            for p in match["info"]["participants"]:
                if (p.get("puuid") == puuid
                        and p.get("teamPosition", "").upper() == role.upper()
                        and p.get("deaths", 0) > 0):
                    needs_timeline.append(match["metadata"]["matchId"])
                    break

        tl_results = await asyncio.gather(
            *[state.riot.get_timeline(mid, routing_region) for mid in needs_timeline],
            return_exceptions=True,
        )
        timelines = {
            mid: res for mid, res in zip(needs_timeline, tl_results)
            if not isinstance(res, Exception)
        }
        for match in all_matches:
            mid          = match["metadata"]["matchId"]
            participants = match["info"]["participants"]
            player_p     = next((p for p in participants if p.get("puuid") == puuid), None)
            if player_p:
                player_p["solo_deaths_before_10"] = (
                    state.riot._extract_solo_deaths_before_10(timelines[mid], puuid, participants)
                    if mid in timelines else 0
                )
    except Exception:
        pass   # таймлайны некритичны

    # 6. Собираем PlayerData из накопленных матчей
    player = PlayerData(
        summoner_name=summoner_name,
        summoner_id=puuid,
        puuid=puuid,
        account_id="",
        platform=platform,
        region=routing_region,
        rank=rank,
        matches=all_matches,
    )

    effective_tier = tier.upper() if tier else (rank.tier if rank else "GOLD")

    # 7. Определяем чемпиона из последнего матча на роли
    champion = "Jinx"
    for match in reversed(all_matches):
        for p in match.get("info", {}).get("participants", []):
            if p.get("puuid") == puuid and p.get("teamPosition") == role.upper():
                champion = p.get("championName", "Jinx")
                break
        else:
            continue
        break

    # 8. Бенчмарки
    try:
        benchmark = await state.bench.get(champion, role.upper(), effective_tier)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Benchmark: {exc}")

    # 9. Анализ
    prev_rank = get_player_rank(state.db, puuid)
    result    = analyze(player, benchmark, previous_rank=prev_rank)

    # 10. Follow-up: сколько новых игр с прошлого коучинга
    prev_log  = get_last_coaching_log(state.db, puuid, role.upper())
    new_games = 0
    if prev_log and all_match_ids:
        new_games = len(set(all_match_ids) - set(prev_log["match_ids"]))

    # 11. Claude
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
            "summary":       f"Claude временно недоступен: {exc}",
            "flags": [], "follow_up": None,
            "coaching_points": [], "confidence": 0.0,
        }

    # 12. БД
    upsert_player(state.db, puuid, summoner_name, platform, rank)
    flagged = _detect_flagged_mistakes(result, benchmark)
    process_analysis_mistakes(state.db, puuid, role.upper(), flagged)

    if coaching.get("confidence", 0) > 0:
        save_coaching_log(
            state.db, puuid, role.upper(), result.patch,
            match_ids=list(all_match_ids),
            advice=coaching,
            stats=result.summary,
            games_count=result.games_used,
        )

    # 13. Ответ
    fresh_mistakes = get_active_mistakes(state.db, puuid, role.upper())
    response = _build_analyze_response(
        result, benchmark, coaching, fresh_mistakes, rank, cached=False
    )
    response["new_games_since_prev"] = new_games
    response["games_searched"]       = total_fetched
    response["role_games_found"]     = role_count

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
    import time as _time

    if role.upper() == "ALL":
        return await _analyze_all_roles(summoner, region, count)

    t0 = _time.monotonic()
    error_msg: Optional[str] = None
    result: Optional[dict]   = None
    try:
        result = await _analyze_single_role(summoner, region, role, tier, count)
        return result
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        try:
            log_request(
                state.db,
                summoner=summoner,
                region=region,
                role=role if role.upper() != "ALL" else "ALL",
                response_ms      = elapsed_ms,
                games_searched   = result.get("games_searched")   if result else None,
                role_games_found = result.get("role_games_found") if result else None,
                from_cache       = bool(result.get("from_cache")) if result else False,
                error            = error_msg,
            )
        except Exception:
            pass  # логирование некритично


@app.get("/stats/sla")
async def sla_stats_endpoint(limit: int = Query(200, ge=10, le=1000)):
    """SLA статистика запросов из request_log."""
    rows = state.db.execute(
        """
        SELECT role, from_cache, error,
               response_ms, games_searched, role_games_found,
               datetime(created_at, 'unixepoch') as ts
        FROM request_log
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    records = [dict(r) for r in rows]

    # Агрегаты по успешным некэшированным запросам
    ok = [r for r in records if r["error"] is None and not r["from_cache"] and r["response_ms"]]
    cached = [r for r in records if r["from_cache"]]
    errors = [r for r in records if r["error"]]

    def pct(lst, p):
        if not lst: return None
        s = sorted(lst)
        return s[int(len(s) * p)]

    ms_vals = [r["response_ms"] for r in ok]

    return {
        "total_requests":  len(records),
        "ok":              len(ok),
        "from_cache":      len(cached),
        "errors":          len(errors),
        "response_ms": {
            "p50":  pct(ms_vals, 0.50),
            "p75":  pct(ms_vals, 0.75),
            "p90":  pct(ms_vals, 0.90),
            "p95":  pct(ms_vals, 0.95),
            "max":  max(ms_vals) if ms_vals else None,
        },
        "recent": records[:20],
    }


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
