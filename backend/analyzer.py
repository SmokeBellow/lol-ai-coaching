"""
Analyzer: извлекаем признаки из матчей, фильтруем выбросы,
считаем rolling-тренды и сравниваем с тировыми бенчмарками.

Входные данные
--------------
  PlayerData   — результат RiotClient.fetch_player_data()
  BenchmarkData — результат BenchmarksClient.get()

Выходные данные
---------------
  AnalysisResult  — всё, что нужно Claude и фронтенду:
    • per-game stats
    • outlier-флаги
    • rolling-10 / rolling-20 тренды
    • квартильная позиция относительно бенчмарка
    • ранговый сдвиг и смена патча
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from riot_client import PlayerData, SummonerRank
from benchmarks_client import BenchmarkData, Percentiles

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Quartile(str, Enum):
    TOP    = "top"      # выше p75
    ABOVE  = "above"    # p50–p75
    BELOW  = "below"    # p25–p50
    BOTTOM = "bottom"   # ниже p25


@dataclass
class GameStats:
    """Признаки одного матча для конкретного игрока."""
    match_id: str
    champion: str
    role: str
    win: bool
    cs_per_min: float
    vision_per_min: float
    deaths: int
    kills: int
    assists: int
    kill_participation: float    # 0–100
    damage_share: float          # % от урона команды
    solo_deaths_early: int       # смерти без союзных ассистов до 10 мин
    duration_minutes: float
    patch: str
    outlier: bool = False        # True — игра исключена из трендов


@dataclass
class MetricTrend:
    """Rolling-средние + направление тренда для одного показателя."""
    rolling_10: Optional[float]
    rolling_20: Optional[float]
    direction: str   # "improving" | "declining" | "stable" | "insufficient_data"


@dataclass
class BenchmarkDelta:
    """Позиция игрока относительно тировых перцентилей."""
    metric: str
    player_value: float
    benchmark_p25: float
    benchmark_p50: float
    benchmark_p75: float
    quartile: Quartile
    delta_vs_median: float    # player_value − p50 (отрицательное = ниже медианы)


@dataclass
class AnalysisResult:
    summoner: str
    region: str
    role: str
    tier: str
    patch: str

    games_analyzed: int       # всего матчей в данных
    games_used: int           # после фильтра выбросов
    outlier_games: int

    # хронологически: старый → новый
    game_stats: list[GameStats]

    trends: dict[str, MetricTrend]         # metric → MetricTrend
    benchmark_deltas: dict[str, BenchmarkDelta]

    rank_changed: bool
    rank_direction: Optional[str]   # "up" | "down" | None

    patch_changed: bool

    # Удобные скалярные значения для Claude-промпта и фронтенда
    summary: dict[str, float]


# ---------------------------------------------------------------------------
# 1. Извлечение признаков из одного матча
# ---------------------------------------------------------------------------

def _extract_game_stats(match: dict, puuid: str) -> Optional[GameStats]:
    """
    Извлекаем GameStats из raw Riot match JSON.
    Возвращает None если игрок не найден или игра слишком короткая.
    """
    info         = match.get("info", {})
    participants = info.get("participants", [])

    # Ищем участника по puuid
    player: Optional[dict] = None
    for p in participants:
        if p.get("puuid") == puuid:
            player = p
            break
    if player is None:
        return None

    duration_sec = int(info.get("gameDuration", 0))
    if duration_sec < 300:          # < 5 минут — реметч / сдача на 1 мин
        return None
    duration_min = duration_sec / 60.0

    kills   = int(player.get("kills",   0))
    deaths  = int(player.get("deaths",  0))
    assists = int(player.get("assists", 0))
    cs      = (int(player.get("totalMinionsKilled", 0))
               + int(player.get("neutralMinionsKilled", 0)))
    vision  = float(player.get("visionScore", 0))
    dmg     = float(player.get("totalDamageDealtToChampions", 0))
    team_id = player.get("teamId", 100)
    champion   = player.get("championName", "")
    position   = player.get("teamPosition") or player.get("individualPosition", "")
    win        = bool(player.get("win", False))
    solo_early = int(player.get("solo_deaths_before_10", 0))

    # Командная статистика для KP и damage share
    team_kills = 0
    team_dmg   = 0.0
    for p in participants:
        if p.get("teamId") == team_id:
            team_kills += int(p.get("kills", 0))
            team_dmg   += float(p.get("totalDamageDealtToChampions", 0))

    kp = (kills + assists) / team_kills * 100.0 if team_kills > 0 else 0.0
    ds = dmg / team_dmg * 100.0 if team_dmg > 0 else 0.0

    # Патч: "15.10.416.3764" → "15.10"
    game_version = info.get("gameVersion", "")
    ver_parts    = game_version.split(".")
    patch        = f"{ver_parts[0]}.{ver_parts[1]}" if len(ver_parts) >= 2 else game_version

    match_id = match.get("metadata", {}).get("matchId", "")

    return GameStats(
        match_id=match_id,
        champion=champion,
        role=position,
        win=win,
        cs_per_min=round(cs / duration_min, 2),
        vision_per_min=round(vision / duration_min, 2),
        deaths=deaths,
        kills=kills,
        assists=assists,
        kill_participation=round(kp, 1),
        damage_share=round(ds, 1),
        solo_deaths_early=solo_early,
        duration_minutes=round(duration_min, 1),
        patch=patch,
    )


# ---------------------------------------------------------------------------
# 2. Фильтр выбросов
# ---------------------------------------------------------------------------
# Игра считается выбросом, если в ней ≥3 метрики выходят за пределы
# (mean ± 1.5 × std).  Это отсеивает дисконнекты и тролль-игры.

_LOW_BAD  = {"cs_per_min", "vision_per_min", "kill_participation", "damage_share"}
# deaths — обратная метрика: высокое значение = плохо


def _flag_outliers(games: list[GameStats]) -> None:
    """Помечает outlier=True непосредственно в списке."""
    if len(games) < 5:
        return    # недостаточно данных для статистики

    # Собираем значения
    vals: dict[str, list[float]] = {
        "cs_per_min":         [g.cs_per_min for g in games],
        "vision_per_min":     [g.vision_per_min for g in games],
        "kill_participation": [g.kill_participation for g in games],
        "damage_share":       [g.damage_share for g in games],
        "deaths":             [float(g.deaths) for g in games],
    }

    # mean + std для каждой метрики
    ms: dict[str, tuple[float, float]] = {}
    for m, v in vals.items():
        mean = statistics.mean(v)
        std  = statistics.stdev(v) if len(v) >= 2 else 0.0
        ms[m] = (mean, std)

    for i, g in enumerate(games):
        bad = 0

        for m in _LOW_BAD:
            mean, std = ms[m]
            v = vals[m][i]
            if std > 0 and v < mean - 1.5 * std:
                bad += 1

        # deaths: высокое = плохо
        mean_d, std_d = ms["deaths"]
        if std_d > 0 and g.deaths > mean_d + 1.5 * std_d:
            bad += 1

        if bad >= 3:
            g.outlier = True


# ---------------------------------------------------------------------------
# 3. Rolling-тренды
# ---------------------------------------------------------------------------

_TREND_THRESHOLD = 0.04   # 4 % изменение rolling_10 vs rolling_20 → тренд


def _rolling_mean(values: list[float], n: int) -> Optional[float]:
    """Среднее последних n значений. None если данных меньше 5."""
    window = values[-n:]
    if len(window) < 5:
        return None
    return round(statistics.mean(window), 3)


def _calc_trend(values: list[float]) -> MetricTrend:
    r10 = _rolling_mean(values, 10)
    r20 = _rolling_mean(values, 20)

    if r10 is None:
        return MetricTrend(r10, r20, "insufficient_data")

    if r20 is None:
        # Есть r10 но не r20 — данных достаточно для краткосрочного тренда
        return MetricTrend(r10, r20, "insufficient_data")

    if r20 == 0.0:
        direction = "stable"
    else:
        ratio = (r10 - r20) / abs(r20)
        if ratio > _TREND_THRESHOLD:
            direction = "improving"
        elif ratio < -_TREND_THRESHOLD:
            direction = "declining"
        else:
            direction = "stable"

    return MetricTrend(rolling_10=r10, rolling_20=r20, direction=direction)


# ---------------------------------------------------------------------------
# 4. Benchmark deltas
# ---------------------------------------------------------------------------

def _quartile_pos(value: float, p: Percentiles) -> Quartile:
    if value >= p.p75:
        return Quartile.TOP
    if value >= p.p50:
        return Quartile.ABOVE
    if value >= p.p25:
        return Quartile.BELOW
    return Quartile.BOTTOM


def _quartile_inv(value: float, p: Percentiles) -> Quartile:
    """Для deaths: меньше = лучше → инвертируем оценку."""
    if value <= p.p25:
        return Quartile.TOP
    if value <= p.p50:
        return Quartile.ABOVE
    if value <= p.p75:
        return Quartile.BELOW
    return Quartile.BOTTOM


def _mk_delta(
    metric: str,
    player_val: float,
    pct: Percentiles,
    inverse: bool = False,
) -> BenchmarkDelta:
    q = _quartile_inv(player_val, pct) if inverse else _quartile_pos(player_val, pct)
    return BenchmarkDelta(
        metric=metric,
        player_value=round(player_val, 2),
        benchmark_p25=pct.p25,
        benchmark_p50=pct.p50,
        benchmark_p75=pct.p75,
        quartile=q,
        delta_vs_median=round(player_val - pct.p50, 2),
    )


# ---------------------------------------------------------------------------
# 5. Главная функция
# ---------------------------------------------------------------------------

def analyze(
    player: PlayerData,
    benchmark: BenchmarkData,
    previous_rank: Optional[SummonerRank] = None,
) -> AnalysisResult:
    """
    Полный анализ одного игрока.

    Parameters
    ----------
    player        : данные из RiotClient.fetch_player_data()
    benchmark     : тировые бенчмарки из BenchmarksClient.get()
    previous_rank : прошлый ранг из БД (для детектирования ранговых сдвигов)

    Returns
    -------
    AnalysisResult
    """

    # --- 1. Извлечение per-game статистики ---
    all_games: list[GameStats] = []
    for match in player.matches:
        gs = _extract_game_stats(match, player.puuid)
        if gs is not None:
            all_games.append(gs)

    # Riot API возвращает матчи новейшими первыми; переворачиваем → хронологически
    all_games.reverse()

    # --- 1b. Фильтр по роли ---
    # Анализируем только игры на запрошенной роли, чтобы статистика и
    # benchmark-сравнение были честными (Топ сравнивается с Топом, и т.д.)
    target_role = benchmark.role.upper()
    all_games = [g for g in all_games if g.role.upper() == target_role]

    # --- 2. Outlier-фильтр ---
    _flag_outliers(all_games)
    clean = [g for g in all_games if not g.outlier]

    # --- 3. Тренды (только «чистые» игры) ---
    def _col(attr: str) -> list[float]:
        return [float(getattr(g, attr)) for g in clean]

    cs_vals  = _col("cs_per_min")
    vis_vals = _col("vision_per_min")
    dth_vals = _col("deaths")
    kp_vals  = _col("kill_participation")
    ds_vals  = _col("damage_share")

    # deaths: инвертируем для тренда ("improving" = смерти снижаются)
    dth_inv = [-d for d in dth_vals]

    trends = {
        "cs_per_min":         _calc_trend(cs_vals),
        "vision_per_min":     _calc_trend(vis_vals),
        "deaths":             _calc_trend(dth_inv),
        "kill_participation": _calc_trend(kp_vals),
        "damage_share":       _calc_trend(ds_vals),
    }

    # --- 4. Benchmark deltas ---
    def _avg(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    # Предпочитаем rolling_10; если недостаточно данных — среднее по всему
    r10_cs  = trends["cs_per_min"].rolling_10  or _avg(cs_vals)
    r10_vis = trends["vision_per_min"].rolling_10 or _avg(vis_vals)
    r10_dth = _avg(dth_vals)   # для бенчмарк-дельты не инвертируем
    r10_kp  = trends["kill_participation"].rolling_10 or _avg(kp_vals)

    benchmark_deltas: dict[str, BenchmarkDelta] = {
        "cs_per_min": _mk_delta(
            "cs_per_min", r10_cs, benchmark.cs_per_min
        ),
        "vision_per_min": _mk_delta(
            "vision_per_min", r10_vis, benchmark.vision_score_per_min
        ),
        "deaths": _mk_delta(
            "deaths", r10_dth, benchmark.deaths_per_game, inverse=True
        ),
        "kill_participation": _mk_delta(
            "kill_participation", r10_kp, benchmark.kill_participation
        ),
    }

    # --- 5. Ранговый сдвиг ---
    rank_changed   = False
    rank_direction: Optional[str] = None
    if previous_rank is not None and player.rank is not None:
        delta_lp = player.rank.numeric_rank - previous_rank.numeric_rank
        if delta_lp != 0:
            rank_changed   = True
            rank_direction = "up" if delta_lp > 0 else "down"

    # --- 6. Патч-контекст ---
    patches = {g.patch for g in all_games if g.patch}
    patch_changed  = len(patches) > 1
    current_patch  = all_games[-1].patch if all_games else benchmark.patch

    # --- 7. Summary ---
    wins_clean = sum(1 for g in clean if g.win)
    wr_clean   = round(wins_clean / len(clean) * 100, 1) if clean else 0.0

    summary: dict[str, float] = {
        "cs_per_min":              r10_cs,
        "vision_per_min":          r10_vis,
        "deaths_per_game":         r10_dth,
        "kill_participation":      r10_kp,
        "damage_share":            _avg(ds_vals),
        "winrate":                 wr_clean,
        "solo_deaths_early_avg":   round(_avg([float(g.solo_deaths_early) for g in clean]), 2),
    }

    return AnalysisResult(
        summoner=player.summoner_name,
        region=player.region,
        role=benchmark.role,
        tier=benchmark.tier,
        patch=current_patch,
        games_analyzed=len(all_games),
        games_used=len(clean),
        outlier_games=len(all_games) - len(clean),
        game_stats=all_games,
        trends=trends,
        benchmark_deltas=benchmark_deltas,
        rank_changed=rank_changed,
        rank_direction=rank_direction,
        patch_changed=patch_changed,
        summary=summary,
    )
