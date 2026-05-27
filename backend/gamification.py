"""
Gamification system: Quests, Achievements, and Level progression for LoL coaching.

Quests
------
  • Generated from the worst-performing benchmark metrics
  • Max 2 active quests per (puuid, role) at any time
  • Progress = qualifying role games played AFTER quest creation
  • Completion → status='completed'; new quest slot opens next analysis

Achievements
------------
  • Performance:  top-quartile CS, vision 30%+ above median, low deaths,
                  high KP, 5-win streak
  • Quest milestones: 1st and 5th quest completed
  • Rank progression: tier level gained since first analysis
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Quest templates
# ---------------------------------------------------------------------------

_QUEST_DEFS: dict[str, dict] = {
    "cs_per_min": {
        "title":            "Фармовый монстр",
        "desc_tmpl":        "Сыграй {n} {gw} с CS ≥ {v:.1f} за мин",
        "icon":             "⚔️",
        "higher_is_better": True,
        "target_pct":       0.90,   # 90 % от benchmark p50
    },
    "vision_per_min": {
        "title":            "Глаз разведки",
        "desc_tmpl":        "Сыграй {n} {gw} с Vision ≥ {v:.2f}/мин",
        "icon":             "👁",
        "higher_is_better": True,
        "target_pct":       0.85,
    },
    "deaths": {
        "title":            "Живучий",
        "desc_tmpl":        "Сыграй {n} {gw} с ≤ {v:.0f} смертей",
        "icon":             "🛡️",
        "higher_is_better": False,
        "target_pct":       1.05,  # ≤ 105 % от median deaths
    },
    "kill_participation": {
        "title":            "Командный игрок",
        "desc_tmpl":        "Сыграй {n} {gw} с KP ≥ {v:.0f}%",
        "icon":             "🤝",
        "higher_is_better": True,
        "target_pct":       0.85,
    },
}

_QUARTILE_ORDER = {"bottom": 0, "below": 1, "above": 2, "top": 3}

QUEST_TARGET_GAMES = 3   # всегда 3 игры на задание


def _games_word(n: int) -> str:
    """Russian inflection: 1 → игру, 2-4 → игры, 5+ → игр."""
    if 11 <= n % 100 <= 19:
        return "игр"
    return {1: "игру", 2: "игры", 3: "игры", 4: "игры"}.get(n % 10, "игр")


# ---------------------------------------------------------------------------
# Level / XP system
# ---------------------------------------------------------------------------
# XP per completed quest: 100 XP
# XP to advance from level N → N+1: N × 100
# Cumulative XP for level N: N*(N-1)/2 × 100
#
#   Level 1:   0 XP  (0 quests)
#   Level 2: 100 XP  (1 quest)
#   Level 3: 300 XP  (3 quests)
#   Level 4: 600 XP  (6 quests)
#   Level 5: 1000 XP (10 quests)

XP_PER_QUEST = 100


def compute_level(total_xp: int) -> dict:
    xp = max(0, total_xp)
    if xp == 0:
        return {
            "level": 1, "total_xp": 0,
            "xp_in_level": 0, "xp_for_next_level": 100, "progress_pct": 0.0,
        }
    # Solve N*(N-1)/2*100 <= xp → N = floor((1 + sqrt(1+8*xp/100))/2)
    level        = max(1, int((1 + math.sqrt(1 + 8 * xp / 100)) / 2))
    lvl_start    = level * (level - 1) // 2 * 100
    lvl_step     = level * 100          # XP needed to go from `level` → `level+1`
    xp_in_level  = xp - lvl_start
    return {
        "level":             level,
        "total_xp":         xp,
        "xp_in_level":      xp_in_level,
        "xp_for_next_level": lvl_step,
        "progress_pct":     round(xp_in_level / lvl_step * 100, 1),
    }


def _game_metric_value(gs, metric: str) -> float:
    return {
        "cs_per_min":         gs.cs_per_min,
        "vision_per_min":     gs.vision_per_min,
        "deaths":             float(gs.deaths),
        "kill_participation": gs.kill_participation,
    }.get(metric, 0.0)


# ---------------------------------------------------------------------------
# Achievement definitions
# ---------------------------------------------------------------------------

_ACHIEVEMENT_DEFS: dict[str, dict] = {
    "cs_master": {
        "title":       "Мастер CS",
        "description": "CS в топ 25% за 10+ игр",
        "icon":        "⚔️",
    },
    "vision_master": {
        "title":       "Всевидящее Oko",
        "description": "Vision Score на 30%+ выше медианы за 10+ игр",
        "icon":        "👁",
    },
    "few_deaths": {
        "title":       "Неприкасаемый",
        "description": "Менее 3 смертей в среднем за 10+ игр",
        "icon":        "🛡️",
    },
    "kp_beast": {
        "title":       "Душа команды",
        "description": "KP выше 65% в среднем за 5+ игр",
        "icon":        "🤝",
    },
    "win_streak_5": {
        "title":       "На взлёте",
        "description": "5 побед подряд",
        "icon":        "🔥",
    },
    "first_quest": {
        "title":       "Первый шаг",
        "description": "Выполнил первое задание",
        "icon":        "✅",
    },
    "quest_master": {
        "title":       "Целеустремлённый",
        "description": "Выполнил 5 заданий",
        "icon":        "🏅",
    },
    "rank_up": {
        "title":       "Повышение",
        "description": "Поднялся на следующий ранг с первого анализа",
        "icon":        "🚀",
    },
    "rank_double_up": {
        "title":       "Двойное повышение",
        "description": "Поднялся на 2+ ранга с первого анализа",
        "icon":        "🌟",
    },
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_gamification(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    result,               # AnalysisResult from analyzer.py
    benchmark,            # BenchmarkData from benchmarks_client.py
    all_match_ids: list[str],
    current_rank,         # Optional[SummonerRank]
) -> dict:
    """
    Full gamification pipeline:
      1. Update progress on existing active quests
      2. Generate a new quest if < 2 active; track newly created quest IDs
      3. Check & award achievements
      4. Compute level / XP

    Returns {
        "quests":           list[dict],
        "achievements":     list[dict],
        "new_achievements": list[dict],
        "new_quest_ids":    list[int],   # IDs created in THIS analysis
        "level":            dict,        # level / XP info
    }
    """
    role = role.upper()
    new_achievements: list[dict] = []

    # ── 1. Update quest progress ──────────────────────────────────────────
    _update_quest_progress(conn, puuid, role, result, all_match_ids)

    # ── 2. Generate new quest if slot available ───────────────────────────
    active_quests = _get_active_quests(conn, puuid, role)
    ids_before    = {q["id"] for q in active_quests}

    if len(active_quests) < 2:
        _generate_quest(
            conn=conn,
            puuid=puuid,
            role=role,
            benchmark_deltas=result.benchmark_deltas,
            all_match_ids=all_match_ids,
            existing_metrics={q["metric"] for q in active_quests},
        )

    active_after  = _get_active_quests(conn, puuid, role)
    new_quest_ids = [q["id"] for q in active_after if q["id"] not in ids_before]

    # ── 3. Check & award achievements ────────────────────────────────────
    completed_quests_count = _count_completed_quests(conn, puuid)
    rank_levels_up = _calc_rank_progression(conn, puuid, current_rank)

    ctx = {
        "completed_quests": completed_quests_count,
        "rank_levels_up":   rank_levels_up,
        "rank_improved":    rank_levels_up >= 1,
    }
    earned_keys = {a["key"] for a in _get_earned_achievements(conn, puuid)}

    for key, defn in _ACHIEVEMENT_DEFS.items():
        if key in earned_keys:
            continue
        if _check_achievement(key, result, ctx):
            ach = _earn_achievement(conn, puuid, key, defn)
            new_achievements.append(ach)

    # ── 4. Level / XP ────────────────────────────────────────────────────
    total_xp   = completed_quests_count * XP_PER_QUEST
    level_info = compute_level(total_xp)

    # ── 5. Collect final state ────────────────────────────────────────────
    return {
        "quests":           _get_quests_for_display(conn, puuid, role),
        "achievements":     _get_earned_achievements(conn, puuid),
        "new_achievements": new_achievements,
        "new_quest_ids":    new_quest_ids,
        "level":            level_info,
    }


# ---------------------------------------------------------------------------
# Quest DB helpers
# ---------------------------------------------------------------------------

def _parse_quest(row) -> dict:
    d = dict(row)
    d["baseline_match_ids"] = json.loads(d.get("baseline_match_ids") or "[]")
    d["higher_is_better"]   = bool(d["higher_is_better"])
    return d


def _get_active_quests(conn: sqlite3.Connection, puuid: str, role: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM quests WHERE puuid=? AND role=? AND status='active'"
        " ORDER BY created_at ASC",
        (puuid, role),
    ).fetchall()
    return [_parse_quest(r) for r in rows]


def _get_quests_for_display(conn: sqlite3.Connection, puuid: str, role: str) -> list[dict]:
    """Active quests + quests completed in the last 7 days."""
    cutoff = time.time() - 7 * 86_400
    rows = conn.execute(
        """
        SELECT * FROM quests WHERE puuid=? AND role=?
          AND (status='active'
               OR (status='completed' AND completed_at > ?))
        ORDER BY
          CASE status WHEN 'active' THEN 0 ELSE 1 END,
          created_at DESC
        LIMIT 6
        """,
        (puuid, role, cutoff),
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        q = _parse_quest(row)
        q.pop("baseline_match_ids", None)   # big list not needed by frontend
        result.append(q)
    return result


def _update_quest_progress(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    result,
    all_match_ids: list[str],
) -> None:
    """Re-count qualifying new games for each active quest and update DB."""
    active = _get_active_quests(conn, puuid, role)
    if not active:
        return

    # Non-outlier games indexed by match_id for fast lookup
    clean_by_mid = {g.match_id: g for g in result.game_stats if not g.outlier}
    all_mid_set  = set(all_match_ids)

    for quest in active:
        baseline_set = set(quest["baseline_match_ids"])
        new_mids = [mid for mid in all_match_ids if mid not in baseline_set]

        metric = quest["metric"]
        target = quest["target_value"]
        higher = quest["higher_is_better"]

        games_done = 0
        for mid in new_mids:
            gs = clean_by_mid.get(mid)
            if gs is None:
                continue
            val = _game_metric_value(gs, metric)
            if (higher and val >= target) or (not higher and val <= target):
                games_done += 1

        now = time.time()
        if games_done >= quest["target_games"] and quest["status"] == "active":
            conn.execute(
                "UPDATE quests SET games_done=?, status='completed', completed_at=? WHERE id=?",
                (games_done, now, quest["id"]),
            )
        elif games_done != quest["games_done"]:
            conn.execute(
                "UPDATE quests SET games_done=? WHERE id=?",
                (games_done, quest["id"]),
            )

    conn.commit()


def _generate_quest(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    benchmark_deltas: dict,
    all_match_ids: list[str],
    existing_metrics: set[str],
) -> None:
    """Generate at most 1 new quest targeting the worst un-tracked metric."""
    # Sort metrics from worst to best quartile
    sorted_metrics = sorted(
        [(m, d) for m, d in benchmark_deltas.items() if m in _QUEST_DEFS],
        key=lambda x: _QUARTILE_ORDER.get(
            x[1].quartile.value if hasattr(x[1].quartile, "value") else str(x[1].quartile),
            3,
        ),
    )

    for metric, delta in sorted_metrics:
        if metric in existing_metrics:
            continue

        defn       = _QUEST_DEFS[metric]
        higher     = defn["higher_is_better"]
        target_val = delta.benchmark_p50 * defn["target_pct"]

        n   = QUEST_TARGET_GAMES   # always 3 games
        gw  = _games_word(n)
        desc = defn["desc_tmpl"].format(n=n, gw=gw, v=target_val)

        conn.execute(
            """
            INSERT INTO quests
              (puuid, role, metric, title, description, icon,
               target_value, target_games, games_done,
               higher_is_better, baseline_match_ids, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?,  ?, ?, 0,  ?, ?,  'active', ?)
            """,
            (
                puuid, role, metric,
                defn["title"], desc, defn["icon"],
                target_val, n,
                int(higher),
                json.dumps(all_match_ids),
                time.time(),
            ),
        )
        conn.commit()
        return   # only one new quest per analysis


# ---------------------------------------------------------------------------
# Achievement helpers
# ---------------------------------------------------------------------------

def _check_win_streak(game_stats, n: int) -> bool:
    """True if the n most recent games (by position, oldest→newest) form a win streak."""
    streak = 0
    for g in reversed(game_stats):   # oldest→newest → reversed = newest first
        if g.win:
            streak += 1
            if streak >= n:
                return True
        else:
            streak = 0
    return False


def _check_achievement(key: str, result, ctx: dict) -> bool:
    deltas = result.benchmark_deltas

    def quartile(metric: str) -> str:
        d = deltas.get(metric)
        if not d:
            return "below"
        return d.quartile.value if hasattr(d.quartile, "value") else str(d.quartile)

    if key == "cs_master":
        return quartile("cs_per_min") == "top" and result.games_used >= 10

    if key == "vision_master":
        d = deltas.get("vision_per_min")
        if not d or result.games_used < 10:
            return False
        # delta_vs_median > 30% of p50 means player is 30%+ above median
        return d.delta_vs_median > 0 and d.delta_vs_median > d.benchmark_p50 * 0.30

    if key == "few_deaths":
        return result.summary.get("deaths_per_game", 99) < 3.0 and result.games_used >= 10

    if key == "kp_beast":
        return result.summary.get("kill_participation", 0) >= 65.0 and result.games_used >= 5

    if key == "win_streak_5":
        return _check_win_streak(result.game_stats, 5)

    if key == "first_quest":
        return ctx.get("completed_quests", 0) >= 1

    if key == "quest_master":
        return ctx.get("completed_quests", 0) >= 5

    if key == "rank_up":
        return ctx.get("rank_levels_up", 0) >= 1

    if key == "rank_double_up":
        return ctx.get("rank_levels_up", 0) >= 2

    return False


def _earn_achievement(conn: sqlite3.Connection, puuid: str, key: str, defn: dict) -> dict:
    now = time.time()
    conn.execute(
        """
        INSERT OR IGNORE INTO achievements (puuid, key, title, description, icon, earned_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (puuid, key, defn["title"], defn["description"], defn["icon"], now),
    )
    conn.commit()
    return {
        "key":         key,
        "title":       defn["title"],
        "description": defn["description"],
        "icon":        defn["icon"],
        "earned_at":   now,
    }


def _get_earned_achievements(conn: sqlite3.Connection, puuid: str) -> list[dict]:
    rows = conn.execute(
        "SELECT key, title, description, icon, earned_at FROM achievements"
        " WHERE puuid=? ORDER BY earned_at ASC",
        (puuid,),
    ).fetchall()
    return [dict(r) for r in rows]


def _count_completed_quests(conn: sqlite3.Connection, puuid: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM quests WHERE puuid=? AND status='completed'", (puuid,)
    ).fetchone()
    return row[0] if row else 0


def _calc_rank_progression(
    conn: sqlite3.Connection,
    puuid: str,
    current_rank,
) -> int:
    """
    Returns how many tier levels the player has climbed since their first analysis.
    0 if no change or insufficient data.
    """
    if current_rank is None:
        return 0

    row = conn.execute(
        "SELECT initial_rank_tier, initial_rank_division, initial_rank_lp"
        " FROM players WHERE puuid=?",
        (puuid,),
    ).fetchone()
    if row is None or not row["initial_rank_tier"]:
        return 0

    tier_order = [
        "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM",
        "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
    ]
    try:
        init_idx = tier_order.index(row["initial_rank_tier"].upper())
        cur_idx  = tier_order.index(current_rank.tier.upper())
        return max(0, cur_idx - init_idx)
    except ValueError:
        return 0
