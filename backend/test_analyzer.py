"""
Тесты для analyzer.py.

Запуск:
  python test_analyzer.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from analyzer import (
    GameStats,
    MetricTrend,
    BenchmarkDelta,
    AnalysisResult,
    Quartile,
    _extract_game_stats,
    _flag_outliers,
    _rolling_mean,
    _calc_trend,
    _quartile_pos,
    _quartile_inv,
    _mk_delta,
    analyze,
)
from riot_client import PlayerData, SummonerRank
from benchmarks_client import BenchmarkData, Percentiles

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

def _make_participant(
    puuid: str = "test-puuid",
    kills: int = 5,
    deaths: int = 3,
    assists: int = 7,
    cs: int = 180,
    neutral: int = 0,
    vision: int = 24,
    dmg: int = 25000,
    win: bool = True,
    team_id: int = 100,
    position: str = "BOTTOM",
    champion: str = "Jinx",
    solo_before_10: int = 1,
) -> dict:
    return {
        "puuid": puuid,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "totalMinionsKilled": cs,
        "neutralMinionsKilled": neutral,
        "visionScore": vision,
        "totalDamageDealtToChampions": dmg,
        "win": win,
        "teamId": team_id,
        "teamPosition": position,
        "championName": champion,
        "solo_deaths_before_10": solo_before_10,
    }


def _make_match(
    match_id: str = "EUW1_12345",
    duration_sec: int = 1800,
    game_version: str = "15.10.416.3764",
    player_puuid: str = "test-puuid",
    player_kills: int = 5,
    player_deaths: int = 3,
    player_assists: int = 7,
    player_cs: int = 180,
    player_vision: int = 24,
    player_dmg: int = 25000,
    player_win: bool = True,
    team_extra_kills: int = 5,   # у остальных игроков команды вместе
    solo_before_10: int = 1,
) -> dict:
    """Минимальный raw Riot match JSON для тестов."""
    player = _make_participant(
        puuid=player_puuid,
        kills=player_kills,
        deaths=player_deaths,
        assists=player_assists,
        cs=player_cs,
        vision=player_vision,
        dmg=player_dmg,
        win=player_win,
        solo_before_10=solo_before_10,
    )
    # Один союзник и один противник для расчёта KP / damage share
    ally = _make_participant(
        puuid="ally-puuid",
        kills=team_extra_kills,
        deaths=2,
        assists=player_kills,
        cs=200,
        vision=20,
        dmg=20000,
        win=player_win,
        team_id=100,
        champion="Caitlyn",
    )
    enemy = _make_participant(
        puuid="enemy-puuid",
        kills=4,
        deaths=6,
        assists=3,
        cs=160,
        vision=18,
        dmg=22000,
        win=not player_win,
        team_id=200,
        champion="Ezreal",
    )
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameDuration": duration_sec,
            "gameVersion": game_version,
            "participants": [player, ally, enemy],
        },
    }


def _make_benchmark(
    role: str = "BOTTOM",
    tier: str = "GOLD",
) -> BenchmarkData:
    return BenchmarkData(
        champion="Jinx",
        role=role,
        tier=tier,
        patch="15.10",
        source="static",
        sample_size=0,
        winrate=50.2,
        cs_per_min=Percentiles(5.5, 7.0, 8.5),
        vision_score_per_min=Percentiles(0.65, 0.95, 1.30),
        deaths_per_game=Percentiles(2.4, 4.1, 6.0),
        kill_participation=Percentiles(42.0, 54.0, 66.0),
        scraped_at=time.time(),
    )


def _make_player(matches: list[dict], rank_tier: str = "GOLD") -> PlayerData:
    return PlayerData(
        summoner_name="TestPlayer#EUW",
        summoner_id="sid",
        puuid="test-puuid",
        account_id="aid",
        platform="euw1",
        region="europe",
        rank=SummonerRank(tier=rank_tier, division="II", lp=50, queue_type="RANKED_SOLO_5x5"),
        matches=matches,
    )


# ---------------------------------------------------------------------------
# _extract_game_stats
# ---------------------------------------------------------------------------

def test_extract_basic_fields() -> None:
    match = _make_match(
        duration_sec=1800,
        player_kills=5, player_deaths=3, player_assists=7,
        player_cs=180, player_vision=24,
    )
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract: returns GameStats")
    if gs:
        eq(gs.kills,   5,     "extract: kills")
        eq(gs.deaths,  3,     "extract: deaths")
        eq(gs.assists, 7,     "extract: assists")
        eq(gs.duration_minutes, 30.0, "extract: duration 1800s -> 30 min")
        eq(gs.champion, "Jinx",       "extract: champion name")
        eq(gs.win, True,              "extract: win flag")
        eq(gs.solo_deaths_early, 1,   "extract: solo_deaths_before_10")


def test_extract_cs_per_min() -> None:
    match = _make_match(duration_sec=1800, player_cs=210)
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract cs: got result")
    if gs:
        # 210 / 30 = 7.0
        eq(gs.cs_per_min, 7.0, "extract: cs_per_min = 210 cs / 30 min")


def test_extract_vision_per_min() -> None:
    match = _make_match(duration_sec=1800, player_vision=30)
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract vision: got result")
    if gs:
        eq(gs.vision_per_min, 1.0, "extract: vision_per_min = 30 / 30")


def test_extract_kill_participation() -> None:
    # player kills=5, assists=7, team_extra_kills=5 → team total = 10
    # KP = (5+7)/10 * 100 = 120% — нет, неправильно.
    # team_kills = player_kills + team_extra_kills = 5 + 5 = 10
    # KP = (5 + 7) / 10 * 100 = 120.0 ... это >100%, но тест проверяет формулу
    match = _make_match(player_kills=3, player_assists=4, team_extra_kills=7)
    # team_kills = 3 + 7 = 10; KP = (3+4)/10 * 100 = 70.0
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract kp: got result")
    if gs:
        eq(gs.kill_participation, 70.0, "extract: KP = (3+4)/10*100 = 70%")


def test_extract_damage_share() -> None:
    # player_dmg=25000, ally_dmg=20000 (same team) → team_dmg=45000
    # ds = 25000/45000 * 100 ≈ 55.6%
    match = _make_match(player_dmg=25000)
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract ds: got result")
    if gs:
        ok(50 < gs.damage_share < 60, f"extract: damage_share ~55.6% (got {gs.damage_share})")


def test_extract_patch_version() -> None:
    match = _make_match(game_version="15.10.416.3764")
    gs = _extract_game_stats(match, "test-puuid")
    ok(gs is not None, "extract patch: got result")
    if gs:
        eq(gs.patch, "15.10", "extract: patch '15.10.416.3764' -> '15.10'")


def test_extract_short_game_returns_none() -> None:
    """Игры короче 5 минут должны игнорироваться."""
    match = _make_match(duration_sec=240)  # 4 мин
    gs = _extract_game_stats(match, "test-puuid")
    eq(gs, None, "extract: game < 5 min returns None")


def test_extract_unknown_puuid_returns_none() -> None:
    match = _make_match()
    gs = _extract_game_stats(match, "nonexistent-puuid")
    eq(gs, None, "extract: unknown puuid returns None")


# ---------------------------------------------------------------------------
# _flag_outliers
# ---------------------------------------------------------------------------

def _make_gs_list(n: int = 10, **override) -> list[GameStats]:
    """Генерирует n нормальных GameStats с возможными переопределениями."""
    base = dict(
        match_id="ID", champion="Jinx", role="BOTTOM", win=True,
        cs_per_min=7.0, vision_per_min=1.0, deaths=4,
        kills=5, assists=7, kill_participation=60.0,
        damage_share=25.0, solo_deaths_early=0,
        duration_minutes=30.0, patch="15.10",
    )
    games = []
    for i in range(n):
        kw = {**base, "match_id": f"ID{i}"}
        kw.update(override)
        games.append(GameStats(**kw))
    return games


def test_outlier_normal_games_not_flagged() -> None:
    games = _make_gs_list(10)
    _flag_outliers(games)
    ok(all(not g.outlier for g in games), "outlier: normal games not flagged")


def test_outlier_bad_game_flagged() -> None:
    """Одна игра с очень низкими CS, vision, KP, DS — должна быть выбросом."""
    games = _make_gs_list(10)
    # Заменяем последнюю игру очень плохой
    games[-1] = GameStats(
        match_id="BAD",
        champion="Jinx", role="BOTTOM", win=False,
        cs_per_min=0.1,    # намного ниже нормы
        vision_per_min=0.01,
        deaths=20,
        kills=0, assists=0,
        kill_participation=0.0,
        damage_share=0.5,
        solo_deaths_early=5,
        duration_minutes=8.0,
        patch="15.10",
    )
    _flag_outliers(games)
    ok(games[-1].outlier, "outlier: catastrophic game flagged as outlier")
    ok(not games[0].outlier, "outlier: normal game stays clean")


def test_outlier_skipped_when_fewer_than_5() -> None:
    """Меньше 5 игр — фильтр не работает (нет статистики)."""
    games = _make_gs_list(4)
    games[-1] = GameStats(
        match_id="BAD", champion="Jinx", role="BOTTOM", win=False,
        cs_per_min=0.0, vision_per_min=0.0, deaths=30,
        kills=0, assists=0, kill_participation=0.0, damage_share=0.0,
        solo_deaths_early=10, duration_minutes=6.0, patch="15.10",
    )
    _flag_outliers(games)
    ok(not games[-1].outlier, "outlier: <5 games -> filter disabled")


# ---------------------------------------------------------------------------
# _rolling_mean + _calc_trend
# ---------------------------------------------------------------------------

def test_rolling_mean_exact_window() -> None:
    vals = [1.0] * 5 + [2.0] * 5
    # последние 10 включают оба блока → среднее (1.0*5 + 2.0*5)/10 = 1.5
    eq(_rolling_mean(vals, 10), 1.5, "rolling: mean of 10 elements")


def test_rolling_mean_fewer_than_5() -> None:
    eq(_rolling_mean([7.0, 7.0, 7.0], 10), None, "rolling: <5 elements returns None")


def test_rolling_mean_uses_tail() -> None:
    # последние 5 из 20 элементов — все 9.0
    vals = [1.0] * 15 + [9.0] * 5
    eq(_rolling_mean(vals, 5), 9.0, "rolling: uses tail window")


def test_calc_trend_improving() -> None:
    # rolling_10 растёт: последние 10 > предыдущие 10
    vals = [5.0] * 10 + [8.0] * 10
    t = _calc_trend(vals)
    eq(t.direction, "improving", "trend: rising values -> improving")
    ok(t.rolling_10 > t.rolling_20, "trend: rolling_10 > rolling_20 when improving")


def test_calc_trend_declining() -> None:
    vals = [8.0] * 10 + [5.0] * 10
    t = _calc_trend(vals)
    eq(t.direction, "declining", "trend: falling values -> declining")


def test_calc_trend_stable() -> None:
    vals = [7.0] * 20
    t = _calc_trend(vals)
    eq(t.direction, "stable", "trend: constant values -> stable")


def test_calc_trend_insufficient() -> None:
    vals = [7.0] * 4   # меньше 5
    t = _calc_trend(vals)
    eq(t.direction, "insufficient_data", "trend: <5 values -> insufficient_data")


# ---------------------------------------------------------------------------
# Quartile + benchmark delta
# ---------------------------------------------------------------------------

def _pct() -> Percentiles:
    return Percentiles(p25=5.5, p50=7.0, p75=8.5)


def test_quartile_top() -> None:
    eq(_quartile_pos(9.0, _pct()), Quartile.TOP, "quartile: 9.0 > p75=8.5 -> TOP")


def test_quartile_above() -> None:
    eq(_quartile_pos(7.5, _pct()), Quartile.ABOVE, "quartile: 7.5 in [7.0,8.5) -> ABOVE")


def test_quartile_below() -> None:
    eq(_quartile_pos(6.0, _pct()), Quartile.BELOW, "quartile: 6.0 in [5.5,7.0) -> BELOW")


def test_quartile_bottom() -> None:
    eq(_quartile_pos(5.0, _pct()), Quartile.BOTTOM, "quartile: 5.0 < p25=5.5 -> BOTTOM")


def test_quartile_inv_deaths_low_is_good() -> None:
    # deaths < p25 → отлично = TOP
    eq(_quartile_inv(2.0, _pct()), Quartile.TOP,   "quartile_inv: deaths<p25 -> TOP")
    eq(_quartile_inv(9.0, _pct()), Quartile.BOTTOM,"quartile_inv: deaths>p75 -> BOTTOM")


def test_benchmark_delta_above_median() -> None:
    delta = _mk_delta("cs_per_min", 8.0, _pct())
    eq(delta.quartile, Quartile.ABOVE, "delta: 8.0 in ABOVE range")
    eq(delta.delta_vs_median, 1.0,     "delta: 8.0 - 7.0 = 1.0")


def test_benchmark_delta_inverse() -> None:
    # deaths=3.0, pct(2.4, 4.1, 6.0) → deaths < p50 → ABOVE
    delta = _mk_delta("deaths", 3.0, Percentiles(2.4, 4.1, 6.0), inverse=True)
    eq(delta.quartile, Quartile.ABOVE, "delta inv: deaths=3.0 (< p50=4.1) -> ABOVE")
    ok(delta.delta_vs_median < 0, "delta inv: delta_vs_median negative (below median)")


# ---------------------------------------------------------------------------
# analyze() end-to-end
# ---------------------------------------------------------------------------

def test_analyze_basic() -> None:
    """Smoke-тест: analyze не падает и возвращает корректную структуру."""
    matches = [
        _make_match(
            match_id=f"ID{i}",
            player_kills=5, player_deaths=3, player_assists=7,
            player_cs=210, player_vision=30,
            game_version="15.10.416.3764",
        )
        for i in range(20)
    ]
    # Riot API отдаёт новейшие первыми — имитируем обратный порядок
    matches.reverse()

    player    = _make_player(matches)
    benchmark = _make_benchmark()

    result = analyze(player, benchmark)

    eq(result.summoner, "TestPlayer#EUW", "analyze: summoner name")
    eq(result.region, "europe",           "analyze: region")
    ok(result.games_analyzed >= 1,        "analyze: games_analyzed > 0")
    ok(result.games_used <= result.games_analyzed, "analyze: games_used <= total")
    ok("cs_per_min" in result.trends,     "analyze: cs_per_min trend present")
    ok("cs_per_min" in result.benchmark_deltas, "analyze: cs_per_min delta present")
    ok("cs_per_min" in result.summary,    "analyze: cs_per_min in summary")


def test_analyze_cs_per_min_value() -> None:
    """Проверяем корректность расчёта cs/min в end-to-end."""
    # 20 игр по 30 минут, cs=210 → 7.0 cs/min
    matches = [
        _make_match(
            match_id=f"ID{i}", duration_sec=1800,
            player_cs=210, player_vision=30,
        )
        for i in range(20)
    ]
    matches.reverse()
    result = analyze(_make_player(matches), _make_benchmark())
    eq(result.summary["cs_per_min"], 7.0, "analyze: summary cs_per_min = 7.0")


def test_analyze_trend_direction() -> None:
    """Если CS растёт в последних играх — тренд 'improving'."""
    # Первые 10 игр: 5.0 cs/min, последние 10: 9.0 cs/min
    early = [
        _make_match(match_id=f"E{i}", duration_sec=1800, player_cs=150)
        for i in range(10)
    ]
    late = [
        _make_match(match_id=f"L{i}", duration_sec=1800, player_cs=270)
        for i in range(10)
    ]
    # Riot: новейшие первыми → late сначала, потом early
    matches = late + early
    result = analyze(_make_player(matches), _make_benchmark())
    eq(result.trends["cs_per_min"].direction, "improving",
       "analyze: rising CS in last 10 games -> improving")


def test_analyze_rank_changed_up() -> None:
    matches = [_make_match() for _ in range(5)]
    player = _make_player(matches, rank_tier="PLATINUM")
    prev   = SummonerRank(tier="GOLD", division="I", lp=75, queue_type="RANKED_SOLO_5x5")

    result = analyze(player, _make_benchmark(), previous_rank=prev)

    ok(result.rank_changed,               "rank: changed flag True")
    eq(result.rank_direction, "up",       "rank: promoted -> direction='up'")


def test_analyze_rank_unchanged() -> None:
    matches = [_make_match() for _ in range(5)]
    player = _make_player(matches, rank_tier="GOLD")
    prev   = SummonerRank(tier="GOLD", division="II", lp=50, queue_type="RANKED_SOLO_5x5")

    result = analyze(player, _make_benchmark(), previous_rank=prev)

    ok(not result.rank_changed,      "rank: same LP -> unchanged")
    eq(result.rank_direction, None,  "rank: no direction when unchanged")


def test_analyze_patch_changed() -> None:
    early = [
        _make_match(match_id=f"E{i}", game_version="15.9.416.0")
        for i in range(5)
    ]
    late = [
        _make_match(match_id=f"L{i}", game_version="15.10.416.0")
        for i in range(5)
    ]
    matches = late + early   # Riot order: newest first
    result = analyze(_make_player(matches), _make_benchmark())
    ok(result.patch_changed, "patch: two different patches -> patch_changed=True")


def test_analyze_outlier_excluded() -> None:
    """Одна катастрофическая игра должна быть исключена из games_used."""
    normal = [
        _make_match(match_id=f"N{i}", player_cs=210, player_vision=30,
                    player_kills=5, player_deaths=3, player_assists=7)
        for i in range(10)
    ]
    bad = _make_match(
        match_id="BAD", duration_sec=600,
        player_cs=0, player_vision=0,
        player_kills=0, player_deaths=15, player_assists=0,
        player_dmg=100,
    )
    matches = [bad] + normal  # newest first
    result = analyze(_make_player(matches), _make_benchmark())

    ok(result.outlier_games >= 1,
       "outlier e2e: at least one game flagged as outlier")
    ok(result.games_used < result.games_analyzed,
       "outlier e2e: games_used < games_analyzed")


def test_analyze_benchmark_delta_quartile() -> None:
    """При высоком cs/min (>p75) дельта должна быть TOP."""
    # cs=300 за 30 мин = 10.0 cs/min > p75=8.5
    matches = [
        _make_match(match_id=f"I{i}", duration_sec=1800, player_cs=300)
        for i in range(20)
    ]
    matches.reverse()
    result = analyze(_make_player(matches), _make_benchmark())
    eq(result.benchmark_deltas["cs_per_min"].quartile, Quartile.TOP,
       "delta e2e: cs=10.0 > p75=8.5 -> TOP")


def test_analyze_winrate_in_summary() -> None:
    """Winrate рассчитывается по чистым играм."""
    wins  = [_make_match(match_id=f"W{i}", player_win=True)  for i in range(10)]
    loses = [_make_match(match_id=f"L{i}", player_win=False) for i in range(10)]
    matches = wins + loses   # newest first; 50/50
    result = analyze(_make_player(matches), _make_benchmark())
    eq(result.summary["winrate"], 50.0, "analyze: 10W 10L -> 50.0% winrate")


def test_analyze_role_filter() -> None:
    """
    При запросе роли TOP должны учитываться только игры на TOP,
    а не все матчи (BOTTOM, JUNGLE, ...).
    """
    def _make_match_role(match_id: str, position: str, cs: int) -> dict:
        """Матч с конкретной позицией и cs."""
        p = _make_participant(puuid="test-puuid", cs=cs, position=position)
        ally = _make_participant(puuid="ally", cs=200, position="JUNGLE", team_id=100)
        return {
            "metadata": {"matchId": match_id},
            "info": {
                "gameDuration": 1800,
                "gameVersion": "15.10.416.3764",
                "participants": [p, ally],
            },
        }

    # 10 игр на TOP с cs=180 (6.0 cs/min) + 10 игр на BOTTOM с cs=270 (9.0 cs/min)
    top_matches    = [_make_match_role(f"T{i}", "TOP",    180) for i in range(10)]
    bottom_matches = [_make_match_role(f"B{i}", "BOTTOM", 270) for i in range(10)]

    # Riot отдаёт новейшие первыми — перемешиваем как реальный ответ
    all_matches = top_matches + bottom_matches  # newest first

    benchmark_top = _make_benchmark(role="TOP")
    result = analyze(_make_player(all_matches), benchmark_top)

    # Должны использоваться только TOP-игры: 10 шт, cs_per_min ≈ 6.0
    eq(result.games_analyzed, 10, "role_filter: games_analyzed = 10 (только TOP)")
    ok(abs(result.summary["cs_per_min"] - 6.0) < 0.1,
       f"role_filter: cs_per_min ~6.0 (tolko TOP), got {result.summary['cs_per_min']}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== analyzer.py - unit tests ===\n")

    # _extract_game_stats
    test_extract_basic_fields()
    test_extract_cs_per_min()
    test_extract_vision_per_min()
    test_extract_kill_participation()
    test_extract_damage_share()
    test_extract_patch_version()
    test_extract_short_game_returns_none()
    test_extract_unknown_puuid_returns_none()

    # _flag_outliers
    test_outlier_normal_games_not_flagged()
    test_outlier_bad_game_flagged()
    test_outlier_skipped_when_fewer_than_5()

    # rolling + trend
    test_rolling_mean_exact_window()
    test_rolling_mean_fewer_than_5()
    test_rolling_mean_uses_tail()
    test_calc_trend_improving()
    test_calc_trend_declining()
    test_calc_trend_stable()
    test_calc_trend_insufficient()

    # quartile + delta
    test_quartile_top()
    test_quartile_above()
    test_quartile_below()
    test_quartile_bottom()
    test_quartile_inv_deaths_low_is_good()
    test_benchmark_delta_above_median()
    test_benchmark_delta_inverse()

    # end-to-end analyze()
    test_analyze_basic()
    test_analyze_cs_per_min_value()
    test_analyze_trend_direction()
    test_analyze_rank_changed_up()
    test_analyze_rank_unchanged()
    test_analyze_patch_changed()
    test_analyze_outlier_excluded()
    test_analyze_benchmark_delta_quartile()
    test_analyze_winrate_in_summary()
    test_analyze_role_filter()

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
