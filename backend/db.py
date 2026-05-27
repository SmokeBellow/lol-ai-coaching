"""
SQLite data layer.

Таблицы
-------
  players          — профили игроков (puuid, ранг, метка времени анализа)
  mistakes         — отслеживаемые ошибки с жизненным циклом
  coaching_log     — история советов Claude (для follow-up контекста)
  analysis_cache   — кэш полного результата анализа (ключ: puuid+role+newest_match)

Правила жизненного цикла ошибок
---------------------------------
  • sessions_present — в скольких сессиях метрика оставалась плохой.
  • sessions_absent  — в скольких сессиях метрика выглядела нормальной.
  • STALE: sessions_absent  >= STALE_THRESHOLD (5) → ошибка авто-резолвится
    (игрок исправил проблему на 5 сессий подряд).
  • ESCALATE: sessions_present >= ESCALATE_THRESHOLD (8) → severity → "escalated"
    (ошибка сохраняется 8 сессий подряд — значит, требует акцента).
  • Scope: каждая ошибка привязана к (puuid, role, metric).

Примечание по concurrency
--------------------------
Для простоты используем синхронный sqlite3 + check_same_thread=False.
Все публичные функции принимают соединение как первый аргумент, чтобы
вызывающий код управлял транзакциями.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional

from riot_client import SummonerRank

# ---------------------------------------------------------------------------
# Константы жизненного цикла
# ---------------------------------------------------------------------------

STALE_THRESHOLD    = 5    # сессий без ошибки → авто-резолв
ESCALATE_THRESHOLD = 8    # сессий с ошибкой → severity = "escalated"

# ---------------------------------------------------------------------------
# Инициализация БД
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid                TEXT PRIMARY KEY,
    summoner             TEXT NOT NULL,
    platform             TEXT NOT NULL,
    rank_tier            TEXT,
    rank_division        TEXT,
    rank_lp              INTEGER,
    last_analyzed        REAL,
    created_at           REAL NOT NULL DEFAULT (strftime('%s', 'now')),
    initial_rank_tier    TEXT,
    initial_rank_division TEXT,
    initial_rank_lp      INTEGER
);

CREATE TABLE IF NOT EXISTS mistakes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid             TEXT    NOT NULL,
    role              TEXT    NOT NULL,
    metric            TEXT    NOT NULL,
    description       TEXT    NOT NULL,
    severity          TEXT    NOT NULL DEFAULT 'minor',
    sessions_present  INTEGER NOT NULL DEFAULT 1,
    sessions_absent   INTEGER NOT NULL DEFAULT 0,
    first_seen        REAL    NOT NULL,
    last_seen         REAL    NOT NULL,
    resolved          INTEGER NOT NULL DEFAULT 0,
    resolved_at       REAL,
    FOREIGN KEY (puuid) REFERENCES players(puuid)
);

CREATE INDEX IF NOT EXISTS idx_mistakes_puuid_role
    ON mistakes (puuid, role, resolved);

CREATE TABLE IF NOT EXISTS coaching_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid        TEXT    NOT NULL,
    role         TEXT    NOT NULL,
    patch        TEXT    NOT NULL,
    match_ids    TEXT    NOT NULL,   -- JSON-массив проанализированных match_id
    advice_json  TEXT    NOT NULL,   -- полный ответ Claude (JSON)
    stats_json   TEXT    NOT NULL,   -- снимок CS/vision/deaths/kp на момент совета
    games_count  INTEGER NOT NULL,
    created_at   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_coaching_puuid_role
    ON coaching_log (puuid, role, created_at);

CREATE TABLE IF NOT EXISTS analysis_cache (
    puuid         TEXT NOT NULL,
    role          TEXT NOT NULL,
    newest_match  TEXT NOT NULL,   -- match_id последней обработанной игры
    result_json   TEXT NOT NULL,   -- полный JSON-ответ /analyze
    cached_at     REAL NOT NULL,
    PRIMARY KEY (puuid, role)
);

CREATE TABLE IF NOT EXISTS request_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       REAL    NOT NULL,
    summoner         TEXT    NOT NULL,
    region           TEXT    NOT NULL,
    role             TEXT    NOT NULL,
    response_ms      INTEGER,        -- NULL если ошибка до завершения
    games_searched   INTEGER,        -- сколько матчей просмотрено (каскад)
    role_games_found INTEGER,        -- сколько игр на роли найдено
    from_cache       INTEGER NOT NULL DEFAULT 0,
    error            TEXT            -- текст ошибки или NULL
);

CREATE INDEX IF NOT EXISTS idx_request_log_created
    ON request_log (created_at);

CREATE TABLE IF NOT EXISTS quests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid               TEXT    NOT NULL,
    role                TEXT    NOT NULL,
    metric              TEXT    NOT NULL,
    title               TEXT    NOT NULL,
    description         TEXT    NOT NULL,
    icon                TEXT    NOT NULL DEFAULT '🎯',
    target_value        REAL    NOT NULL,
    target_games        INTEGER NOT NULL,
    games_done          INTEGER NOT NULL DEFAULT 0,
    higher_is_better    INTEGER NOT NULL DEFAULT 1,
    baseline_match_ids  TEXT    NOT NULL DEFAULT '[]',
    status              TEXT    NOT NULL DEFAULT 'active',
    created_at          REAL    NOT NULL,
    completed_at        REAL,
    FOREIGN KEY (puuid) REFERENCES players(puuid)
);

CREATE INDEX IF NOT EXISTS idx_quests_puuid_role
    ON quests (puuid, role, status);

CREATE TABLE IF NOT EXISTS achievements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid       TEXT NOT NULL,
    key         TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    icon        TEXT NOT NULL DEFAULT '🏆',
    earned_at   REAL NOT NULL,
    metadata    TEXT,
    UNIQUE (puuid, key),
    FOREIGN KEY (puuid) REFERENCES players(puuid)
);

CREATE INDEX IF NOT EXISTS idx_achievements_puuid
    ON achievements (puuid, earned_at);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """
    Idempotent schema migrations for columns added after initial release.
    Called once per init_db() — safe to run on every startup.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()}
    for col, defn in [
        ("initial_rank_tier",     "TEXT"),
        ("initial_rank_division", "TEXT"),
        ("initial_rank_lp",       "INTEGER"),
    ]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE players ADD COLUMN {col} {defn}")
    conn.commit()


def init_db(path: str = "lol_coaching.db") -> sqlite3.Connection:
    """Открывает (или создаёт) БД и применяет схему. Возвращает соединение."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row    # dict-like доступ к строкам
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Players CRUD
# ---------------------------------------------------------------------------

def upsert_player(
    conn: sqlite3.Connection,
    puuid: str,
    summoner: str,
    platform: str,
    rank: Optional[SummonerRank] = None,
) -> None:
    """
    Вставляет или обновляет профиль игрока.

    initial_rank_* задаётся ТОЛЬКО при первой вставке (не перезаписывается)
    — используется для отслеживания ранговой прогрессии.
    """
    now = time.time()
    conn.execute(
        """
        INSERT INTO players
            (puuid, summoner, platform, rank_tier, rank_division, rank_lp,
             last_analyzed, created_at,
             initial_rank_tier, initial_rank_division, initial_rank_lp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?,  ?, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            summoner       = excluded.summoner,
            platform       = excluded.platform,
            rank_tier      = excluded.rank_tier,
            rank_division  = excluded.rank_division,
            rank_lp        = excluded.rank_lp,
            last_analyzed  = excluded.last_analyzed
            -- initial_rank_* intentionally omitted: never updated after first insert
        """,
        (
            puuid, summoner, platform,
            rank.tier     if rank else None,
            rank.division if rank else None,
            rank.lp       if rank else None,
            now, now,
            # initial rank — only meaningful on first INSERT
            rank.tier     if rank else None,
            rank.division if rank else None,
            rank.lp       if rank else None,
        ),
    )
    conn.commit()


def get_player(conn: sqlite3.Connection, puuid: str) -> Optional[dict]:
    """Возвращает словарь с данными игрока или None."""
    row = conn.execute(
        "SELECT * FROM players WHERE puuid = ?", (puuid,)
    ).fetchone()
    return dict(row) if row else None


def get_player_by_summoner(conn: sqlite3.Connection, summoner: str) -> Optional[dict]:
    """Поиск игрока по summoner-имени (без учёта регистра)."""
    row = conn.execute(
        "SELECT * FROM players WHERE lower(summoner) = lower(?)", (summoner,)
    ).fetchone()
    return dict(row) if row else None


def get_player_rank(conn: sqlite3.Connection, puuid: str) -> Optional[SummonerRank]:
    """Возвращает последний сохранённый ранг игрока или None."""
    row = conn.execute(
        "SELECT rank_tier, rank_division, rank_lp FROM players WHERE puuid = ?",
        (puuid,),
    ).fetchone()
    if row is None or row["rank_tier"] is None:
        return None
    return SummonerRank(
        tier=row["rank_tier"],
        division=row["rank_division"] or "IV",
        lp=int(row["rank_lp"] or 0),
        queue_type="RANKED_SOLO_5x5",
    )


# ---------------------------------------------------------------------------
# Mistakes CRUD
# ---------------------------------------------------------------------------

def get_active_mistakes(
    conn: sqlite3.Connection,
    puuid: str,
    role: Optional[str] = None,
) -> list[dict]:
    """
    Возвращает активные (нерезолвленные) ошибки для игрока.
    Если role задан — только для этой роли.
    """
    if role:
        rows = conn.execute(
            "SELECT * FROM mistakes WHERE puuid=? AND role=? AND resolved=0"
            " ORDER BY sessions_present DESC",
            (puuid, role.upper()),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM mistakes WHERE puuid=? AND resolved=0"
            " ORDER BY sessions_present DESC",
            (puuid,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_mistake(conn: sqlite3.Connection, mistake_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM mistakes WHERE id = ?", (mistake_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_mistake(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    metric: str,
    description: str,
    severity: str = "minor",
) -> int:
    """
    Добавляет новую ошибку или обновляет существующую (same puuid+role+metric).

    При обновлении:
      • sessions_present += 1
      • sessions_absent сбрасывается в 0
      • last_seen = now
      • severity может повыситься до 'escalated' по пороговому правилу

    Возвращает id записи.
    """
    now  = time.time()
    role = role.upper()
    existing = conn.execute(
        "SELECT id, sessions_present, severity FROM mistakes"
        " WHERE puuid=? AND role=? AND metric=? AND resolved=0",
        (puuid, role, metric),
    ).fetchone()

    if existing:
        mid       = existing["id"]
        sessions  = existing["sessions_present"] + 1
        new_sev   = "escalated" if sessions >= ESCALATE_THRESHOLD else existing["severity"]
        conn.execute(
            """
            UPDATE mistakes SET
                sessions_present = ?,
                sessions_absent  = 0,
                description      = ?,
                severity         = ?,
                last_seen        = ?
            WHERE id = ?
            """,
            (sessions, description, new_sev, now, mid),
        )
    else:
        conn.execute(
            """
            INSERT INTO mistakes
              (puuid, role, metric, description, severity,
               sessions_present, sessions_absent, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?,  1, 0, ?, ?)
            """,
            (puuid, role, metric, description, severity, now, now),
        )
        mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.commit()
    return mid


def resolve_mistake(conn: sqlite3.Connection, mistake_id: int) -> None:
    """Помечает ошибку как решённую."""
    conn.execute(
        "UPDATE mistakes SET resolved=1, resolved_at=? WHERE id=?",
        (time.time(), mistake_id),
    )
    conn.commit()


def tick_absent_sessions(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    active_metrics: set[str],
) -> list[int]:
    """
    Вызывается после каждого анализа для обновления sessions_absent.

    Для всех активных ошибок (puuid, role), метрика которых НЕ входит
    в active_metrics (т.е. в этой сессии ошибка не проявилась):
      • sessions_absent += 1
      • если sessions_absent >= STALE_THRESHOLD → авто-резолв

    Возвращает список id ошибок, которые были авто-резолвлены.
    """
    role     = role.upper()
    now      = time.time()
    resolved = []

    rows = conn.execute(
        "SELECT id, metric, sessions_absent FROM mistakes"
        " WHERE puuid=? AND role=? AND resolved=0",
        (puuid, role),
    ).fetchall()

    for row in rows:
        if row["metric"] in active_metrics:
            continue   # ошибка активна в этой сессии — обновляется через upsert_mistake

        new_absent = row["sessions_absent"] + 1
        if new_absent >= STALE_THRESHOLD:
            conn.execute(
                "UPDATE mistakes SET resolved=1, resolved_at=? WHERE id=?",
                (now, row["id"]),
            )
            resolved.append(row["id"])
        else:
            conn.execute(
                "UPDATE mistakes SET sessions_absent=? WHERE id=?",
                (new_absent, row["id"]),
            )

    conn.commit()
    return resolved


# ---------------------------------------------------------------------------
# Batch helper для анализа (один вызов после analyze())
# ---------------------------------------------------------------------------

def process_analysis_mistakes(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    flagged: list[dict],
) -> None:
    """
    Принимает список словарей {"metric", "description", "severity"} —
    признаки, признанные ошибками текущей сессии.

    1. Для каждой признанной ошибки: upsert_mistake
    2. Для остальных активных ошибок: tick_absent_sessions
    """
    active_metrics: set[str] = set()

    for item in flagged:
        upsert_mistake(
            conn,
            puuid=puuid,
            role=role,
            metric=item["metric"],
            description=item["description"],
            severity=item.get("severity", "minor"),
        )
        active_metrics.add(item["metric"])

    tick_absent_sessions(conn, puuid, role, active_metrics)


# ---------------------------------------------------------------------------
# Coaching log
# ---------------------------------------------------------------------------

def save_coaching_log(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    patch: str,
    match_ids: list[str],
    advice: dict,
    stats: dict,
    games_count: int,
) -> None:
    """Сохраняет запись о совете Claude для дальнейшего follow-up."""
    import json
    conn.execute(
        """
        INSERT INTO coaching_log
            (puuid, role, patch, match_ids, advice_json, stats_json, games_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            puuid, role.upper(), patch,
            json.dumps(match_ids),
            json.dumps(advice, ensure_ascii=False),
            json.dumps(stats, ensure_ascii=False),
            games_count,
            time.time(),
        ),
    )
    conn.commit()


def get_last_coaching_log(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
) -> Optional[dict]:
    """
    Возвращает последнюю запись coaching_log для (puuid, role) или None.
    Результат содержит parsed advice_json, stats_json, match_ids (как list).
    """
    import json
    row = conn.execute(
        "SELECT * FROM coaching_log WHERE puuid=? AND role=?"
        " ORDER BY created_at DESC LIMIT 1",
        (puuid, role.upper()),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["match_ids"]   = json.loads(d["match_ids"])
    d["advice_json"] = json.loads(d["advice_json"])
    d["stats_json"]  = json.loads(d["stats_json"])
    return d


# ---------------------------------------------------------------------------
# Analysis cache
# ---------------------------------------------------------------------------

def get_cached_analysis(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    newest_match: str,
) -> Optional[dict]:
    """
    Возвращает распарсенный result_json если newest_match совпадает с кэшем,
    иначе None (cache miss — нужен полный пересчёт).
    """
    import json
    row = conn.execute(
        "SELECT result_json, newest_match FROM analysis_cache WHERE puuid=? AND role=?",
        (puuid, role.upper()),
    ).fetchone()
    if row is None or row["newest_match"] != newest_match:
        return None
    data = json.loads(row["result_json"])
    # Инвалидируем кэш старых версий без обязательных полей
    if "champion_stats" not in data:
        return None
    if not data.get("champion_stats"):   # пустой список — тоже недействителен
        return None
    # Инвалидируем кэш без квестов (до геймификации) — пустой список тоже не считается,
    # потому что на cache-hit мы пересчитываем квесты из БД
    if "quests" not in data:
        return None
    return data


def log_request(
    conn: sqlite3.Connection,
    summoner: str,
    region: str,
    role: str,
    response_ms: Optional[int] = None,
    games_searched: Optional[int] = None,
    role_games_found: Optional[int] = None,
    from_cache: bool = False,
    error: Optional[str] = None,
) -> None:
    """Записывает строку в request_log для мониторинга SLA."""
    conn.execute(
        """
        INSERT INTO request_log
            (created_at, summoner, region, role,
             response_ms, games_searched, role_games_found, from_cache, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(), summoner, region.upper(), role.upper(),
            response_ms, games_searched, role_games_found,
            int(from_cache), error,
        ),
    )
    conn.commit()


def save_analysis_cache(
    conn: sqlite3.Connection,
    puuid: str,
    role: str,
    newest_match: str,
    result: dict,
) -> None:
    """Сохраняет или перезаписывает кэш анализа."""
    import json
    conn.execute(
        """
        INSERT INTO analysis_cache (puuid, role, newest_match, result_json, cached_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(puuid, role) DO UPDATE SET
            newest_match = excluded.newest_match,
            result_json  = excluded.result_json,
            cached_at    = excluded.cached_at
        """,
        (puuid, role.upper(), newest_match, json.dumps(result, ensure_ascii=False), time.time()),
    )
    conn.commit()
