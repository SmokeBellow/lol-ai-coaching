"""
Тесты для db.py.

Запуск:
  python test_db.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

from db import (
    ESCALATE_THRESHOLD,
    STALE_THRESHOLD,
    get_active_mistakes,
    get_mistake,
    get_player,
    get_player_rank,
    init_db,
    process_analysis_mistakes,
    resolve_mistake,
    tick_absent_sessions,
    upsert_mistake,
    upsert_player,
)
from riot_client import SummonerRank

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
# Fixture helper
# ---------------------------------------------------------------------------

def _fresh_db():
    """Создаёт временную БД и возвращает (conn, path)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = init_db(path)
    return conn, path


def _cleanup(conn, path: str) -> None:
    conn.close()
    try:
        os.unlink(path)
    except PermissionError:
        pass   # Windows: файл иногда удерживается


def _gold_rank() -> SummonerRank:
    return SummonerRank(tier="GOLD", division="II", lp=50, queue_type="RANKED_SOLO_5x5")


def _plat_rank() -> SummonerRank:
    return SummonerRank(tier="PLATINUM", division="IV", lp=10, queue_type="RANKED_SOLO_5x5")


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_creates_tables() -> None:
    conn, path = _fresh_db()
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        ok("players"  in tables, "init: players table created")
        ok("mistakes" in tables, "init: mistakes table created")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# Players CRUD
# ---------------------------------------------------------------------------

def test_upsert_player_insert() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "puuid1", "Faker#KR1", "kr", _gold_rank())
        row = get_player(conn, "puuid1")
        ok(row is not None,               "player insert: row found")
        eq(row["summoner"], "Faker#KR1",  "player insert: summoner")
        eq(row["rank_tier"], "GOLD",      "player insert: rank_tier")
        eq(row["rank_lp"], 50,            "player insert: rank_lp")
    finally:
        _cleanup(conn, path)


def test_upsert_player_update() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "puuid1", "Faker#KR1", "kr", _gold_rank())
        upsert_player(conn, "puuid1", "Faker#KR1", "kr", _plat_rank())
        row = get_player(conn, "puuid1")
        eq(row["rank_tier"], "PLATINUM", "player update: rank updated to PLATINUM")
        eq(row["rank_lp"], 10,           "player update: lp updated")
    finally:
        _cleanup(conn, path)


def test_upsert_player_no_rank() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "puuid1", "Anon#EUW", "euw1")
        row = get_player(conn, "puuid1")
        ok(row is not None,         "player no_rank: row exists")
        ok(row["rank_tier"] is None,"player no_rank: rank_tier is NULL")
    finally:
        _cleanup(conn, path)


def test_get_player_missing() -> None:
    conn, path = _fresh_db()
    try:
        eq(get_player(conn, "nonexistent"), None, "player get: missing -> None")
    finally:
        _cleanup(conn, path)


def test_get_player_rank() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "na1", _gold_rank())
        rank = get_player_rank(conn, "p1")
        ok(rank is not None,       "rank get: not None")
        eq(rank.tier, "GOLD",      "rank get: tier")
        eq(rank.division, "II",    "rank get: division")
        eq(rank.lp, 50,            "rank get: lp")
    finally:
        _cleanup(conn, path)


def test_get_player_rank_no_rank() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "na1")
        rank = get_player_rank(conn, "p1")
        eq(rank, None, "rank get: player without rank -> None")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# Mistakes CRUD
# ---------------------------------------------------------------------------

def test_upsert_mistake_insert() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min",
                             "CS ниже медианы", "minor")
        ok(mid > 0, "mistake insert: id > 0")
        row = get_mistake(conn, mid)
        ok(row is not None,               "mistake insert: row found")
        eq(row["metric"], "cs_per_min",   "mistake insert: metric")
        eq(row["sessions_present"], 1,    "mistake insert: sessions_present=1")
        eq(row["sessions_absent"], 0,     "mistake insert: sessions_absent=0")
        eq(row["resolved"], 0,            "mistake insert: not resolved")
    finally:
        _cleanup(conn, path)


def test_upsert_mistake_update_increments_present() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "desc", "minor")
        upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "desc v2", "minor")
        row = get_mistake(conn, mid)
        eq(row["sessions_present"], 2, "mistake update: sessions_present=2")
        eq(row["sessions_absent"], 0,  "mistake update: absent reset to 0")
        eq(row["description"], "desc v2", "mistake update: description updated")
    finally:
        _cleanup(conn, path)


def test_upsert_mistake_role_uppercase() -> None:
    """Роль должна сохраняться в верхнем регистре."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        upsert_mistake(conn, "p1", "bottom", "cs_per_min", "desc", "minor")
        active = get_active_mistakes(conn, "p1", role="BOTTOM")
        ok(len(active) == 1, "mistake role: case-insensitive insert found")
    finally:
        _cleanup(conn, path)


def test_upsert_mistake_separate_roles() -> None:
    """Одна и та же метрика в разных ролях — разные записи."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        upsert_mistake(conn, "p1", "BOTTOM",  "cs_per_min", "d1", "minor")
        upsert_mistake(conn, "p1", "MIDDLE",  "cs_per_min", "d2", "minor")
        all_active = get_active_mistakes(conn, "p1")
        eq(len(all_active), 2, "mistake roles: two separate records for two roles")
    finally:
        _cleanup(conn, path)


def test_get_active_mistakes_filtered_by_role() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d1")
        upsert_mistake(conn, "p1", "MIDDLE", "vision_per_min", "d2")
        bottom = get_active_mistakes(conn, "p1", role="BOTTOM")
        eq(len(bottom), 1, "active mistakes: role filter returns 1 record")
        eq(bottom[0]["metric"], "cs_per_min", "active mistakes: correct metric")
    finally:
        _cleanup(conn, path)


def test_resolve_mistake() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        resolve_mistake(conn, mid)
        row = get_mistake(conn, mid)
        eq(row["resolved"], 1,            "resolve: resolved=1")
        ok(row["resolved_at"] is not None,"resolve: resolved_at set")
        active = get_active_mistakes(conn, "p1")
        eq(len(active), 0, "resolve: not in active list after resolve")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# Lifecycle: tick_absent_sessions
# ---------------------------------------------------------------------------

def test_tick_absent_increments() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        # Нет активных метрик → absent += 1
        tick_absent_sessions(conn, "p1", "BOTTOM", active_metrics=set())
        row = get_mistake(conn, mid)
        eq(row["sessions_absent"], 1, "tick absent: sessions_absent=1 after one tick")
    finally:
        _cleanup(conn, path)


def test_tick_absent_auto_resolve_at_threshold() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        resolved = []
        for _ in range(STALE_THRESHOLD):
            resolved = tick_absent_sessions(conn, "p1", "BOTTOM", active_metrics=set())

        row = get_mistake(conn, mid)
        eq(row["resolved"], 1, "tick: auto-resolved after STALE_THRESHOLD absent sessions")
        ok(mid in resolved,   "tick: resolved id in returned list")
    finally:
        _cleanup(conn, path)


def test_tick_present_metric_not_incremented() -> None:
    """Если метрика активна, sessions_absent не растёт."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        tick_absent_sessions(conn, "p1", "BOTTOM", active_metrics={"cs_per_min"})
        row = get_mistake(conn, mid)
        eq(row["sessions_absent"], 0, "tick: metric in active_metrics -> absent unchanged")
    finally:
        _cleanup(conn, path)


def test_tick_absent_resets_on_reappear() -> None:
    """После 3 тиков absent, если ошибка снова появляется — absent сбрасывается."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        # 3 тика absent
        for _ in range(3):
            tick_absent_sessions(conn, "p1", "BOTTOM", set())
        row = get_mistake(conn, mid)
        eq(row["sessions_absent"], 3, "tick reset: absent=3 after 3 ticks")

        # Ошибка снова появляется
        upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d again")
        row = get_mistake(conn, mid)
        eq(row["sessions_absent"], 0, "tick reset: absent reset to 0 after reappear")
        eq(row["sessions_present"], 2, "tick reset: sessions_present incremented")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# Lifecycle: escalation
# ---------------------------------------------------------------------------

def test_escalate_after_threshold() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        # Первый раз — minor
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d", severity="minor")
        # Вызываем upsert ещё ESCALATE_THRESHOLD-1 раз
        for _ in range(ESCALATE_THRESHOLD - 1):
            upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")

        row = get_mistake(conn, mid)
        eq(row["severity"], "escalated",
           f"escalate: severity='escalated' after {ESCALATE_THRESHOLD} sessions")
        eq(row["sessions_present"], ESCALATE_THRESHOLD,
           "escalate: sessions_present == ESCALATE_THRESHOLD")
    finally:
        _cleanup(conn, path)


def test_no_escalate_before_threshold() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        mid = upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d", severity="minor")
        for _ in range(ESCALATE_THRESHOLD - 2):
            upsert_mistake(conn, "p1", "BOTTOM", "cs_per_min", "d")
        row = get_mistake(conn, mid)
        ok(row["severity"] != "escalated",
           "escalate: severity stays minor before threshold")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# process_analysis_mistakes
# ---------------------------------------------------------------------------

def test_process_analysis_upserts_flagged() -> None:
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        flagged = [
            {"metric": "cs_per_min", "description": "CS ниже медианы", "severity": "minor"},
            {"metric": "vision_per_min", "description": "Vision слабый", "severity": "minor"},
        ]
        process_analysis_mistakes(conn, "p1", "BOTTOM", flagged)
        active = get_active_mistakes(conn, "p1", role="BOTTOM")
        eq(len(active), 2, "process: two mistakes upserted")
    finally:
        _cleanup(conn, path)


def test_process_analysis_ticks_absent_for_cleared_metric() -> None:
    """Ошибка, исчезнувшая в новой сессии, должна получить absent tick."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        # Первая сессия: cs_per_min плохой
        process_analysis_mistakes(conn, "p1", "BOTTOM",
                                  [{"metric": "cs_per_min", "description": "d", "severity": "minor"}])
        # Вторая сессия: все метрики нормальные (нет flagged)
        process_analysis_mistakes(conn, "p1", "BOTTOM", [])
        active = get_active_mistakes(conn, "p1")
        ok(len(active) == 1, "process tick: still active (absent=1 < threshold)")
        eq(active[0]["sessions_absent"], 1,
           "process tick: sessions_absent=1 after one missed session")
    finally:
        _cleanup(conn, path)


def test_process_analysis_auto_resolve_stale() -> None:
    """После STALE_THRESHOLD пустых сессий ошибка должна авто-резолвиться."""
    conn, path = _fresh_db()
    try:
        upsert_player(conn, "p1", "X#Y", "euw1")
        process_analysis_mistakes(conn, "p1", "BOTTOM",
                                  [{"metric": "cs_per_min", "description": "d"}])
        for _ in range(STALE_THRESHOLD):
            process_analysis_mistakes(conn, "p1", "BOTTOM", [])

        active = get_active_mistakes(conn, "p1")
        eq(len(active), 0,
           f"process stale: auto-resolved after {STALE_THRESHOLD} empty sessions")
    finally:
        _cleanup(conn, path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== db.py - unit tests ===\n")

    # init
    test_init_creates_tables()

    # players
    test_upsert_player_insert()
    test_upsert_player_update()
    test_upsert_player_no_rank()
    test_get_player_missing()
    test_get_player_rank()
    test_get_player_rank_no_rank()

    # mistakes basic
    test_upsert_mistake_insert()
    test_upsert_mistake_update_increments_present()
    test_upsert_mistake_role_uppercase()
    test_upsert_mistake_separate_roles()
    test_get_active_mistakes_filtered_by_role()
    test_resolve_mistake()

    # lifecycle absent ticks
    test_tick_absent_increments()
    test_tick_absent_auto_resolve_at_threshold()
    test_tick_present_metric_not_incremented()
    test_tick_absent_resets_on_reappear()

    # escalation
    test_escalate_after_threshold()
    test_no_escalate_before_threshold()

    # process_analysis_mistakes
    test_process_analysis_upserts_flagged()
    test_process_analysis_ticks_absent_for_cleared_metric()
    test_process_analysis_auto_resolve_stale()

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
