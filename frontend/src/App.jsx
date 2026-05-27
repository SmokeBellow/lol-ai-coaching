/**
 * LoL AI Coaching — SPA
 *
 * Режимы:
 *  ALL   → RoleComparePanel (игры по ролям, лучшая роль)
 *  Роль  → полный анализ + FlagsBar + SummaryCard + Claude coaching + follow_up + MistakeTracker
 */

import React, { useState, useCallback } from 'react'

// В продакшене (GitHub Pages) VITE_API_URL = Railway URL бэкенда.
// В разработке (localhost) — пустая строка, запросы идут через Vite proxy.
const API = import.meta.env.VITE_API_URL || ''

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
const C = {
  bg: '#0a0e1a', surface: '#111827', border: '#1e2d45',
  accent: '#3b82f6', accentLo: '#1d4ed8',
  text: '#c8d0e0', textDim: '#6b7280',
  success: '#22c55e', warn: '#f59e0b', danger: '#ef4444', purple: '#a855f7',
  gold: '#f59e0b',
}
const ROW = { display: 'flex', gap: 8, alignItems: 'center' }
const COL = { display: 'flex', flexDirection: 'column', gap: 8 }
const CARD = { background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: '16px 20px' }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const quartileColor  = q => ({ top: C.success, above: '#84cc16', below: C.warn, bottom: C.danger }[q] ?? C.text)
const quartileLabel  = q => ({ top: 'ТОП 25%', above: 'ВЫШЕ МЕД', below: 'НИЖЕ МЕД', bottom: 'БОТ 25%' }[q] ?? (q||'').toUpperCase())
const trendIcon      = d => ({ improving: '↑', declining: '↓', stable: '→', insufficient_data: '?' }[d] ?? '?')
const trendColor     = d => ({ improving: C.success, declining: C.danger, stable: C.textDim, insufficient_data: C.textDim }[d] ?? C.text)
const severityColor  = s => ({ minor: C.warn, moderate: '#f97316', major: C.danger, escalated: C.purple }[s] ?? C.text)
const fmtNum         = (n, d = 2) => n != null ? Number(n).toFixed(d) : '—'
const ROLE_RU        = { TOP: 'Топ', JUNGLE: 'Джунгли', MIDDLE: 'Мид', BOTTOM: 'Бот', UTILITY: 'Сапорт', ALL: 'Все роли' }
const TIER_COLOR     = { IRON: '#8b8b8b', BRONZE: '#cd7f32', SILVER: '#a8a9ad', GOLD: C.gold, PLATINUM: '#0ac8b9', EMERALD: '#30c462', DIAMOND: '#5966b0', MASTER: '#9d48e0', GRANDMASTER: '#e84057', CHALLENGER: '#f4c874' }

function Pill({ label, color }) {
  return (
    <span style={{ background: color+'22', border: `1px solid ${color}`, color, borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 700, letterSpacing: '0.04em' }}>
      {label}
    </span>
  )
}
function SectionTitle({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.1em', color: C.textDim, textTransform: 'uppercase', marginBottom: 8 }}>{children}</div>
}
function Divider() { return <div style={{ height: 1, background: C.border, margin: '4px 0' }} /> }

// ---------------------------------------------------------------------------
// 1. Input Form
// ---------------------------------------------------------------------------
const REGIONS = ['na', 'euw', 'kr', 'eune', 'br', 'oce', 'tr', 'ru', 'jp', 'lan', 'las']
const ROLES   = ['ALL', 'BOTTOM', 'MIDDLE', 'TOP', 'JUNGLE', 'UTILITY']

function InputForm({ onSubmit, loading }) {
  const [summoner, setSummoner] = useState('')
  const [region,   setRegion]   = useState('euw')
  const [role,     setRole]     = useState('ALL')
  const [count,    setCount]    = useState(20)

  const submit = e => { e.preventDefault(); if (summoner.trim()) onSubmit({ summoner: summoner.trim(), region, role, count }) }

  const inp = { background: '#0d1526', border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, padding: '8px 12px', fontSize: 14, outline: 'none' }

  return (
    <form onSubmit={submit} style={{ ...CARD, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
      <div style={COL}>
        <label style={{ fontSize: 11, color: C.textDim }}>SUMMONER (Name#TAG)</label>
        <input style={{ ...inp, width: 220 }} placeholder="Faker#KR1" value={summoner} onChange={e => setSummoner(e.target.value)} required />
      </div>
      <div style={COL}>
        <label style={{ fontSize: 11, color: C.textDim }}>РЕГИОН</label>
        <select style={{ ...inp, width: 90 }} value={region} onChange={e => setRegion(e.target.value)}>
          {REGIONS.map(r => <option key={r} value={r}>{r.toUpperCase()}</option>)}
        </select>
      </div>
      <div style={COL}>
        <label style={{ fontSize: 11, color: C.textDim }}>РОЛЬ</label>
        <select style={{ ...inp, width: 120 }} value={role} onChange={e => setRole(e.target.value)}>
          {ROLES.map(r => <option key={r} value={r}>{ROLE_RU[r] || r}</option>)}
        </select>
      </div>
      <div style={COL}>
        <label style={{ fontSize: 11, color: C.textDim }}>ИГРЫ</label>
        <select style={{ ...inp, width: 80 }} value={count} onChange={e => setCount(Number(e.target.value))}>
          {[10, 15, 20, 30, 50].map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      </div>
      <button type="submit" disabled={loading} style={{ background: loading ? C.accentLo : C.accent, border: 'none', borderRadius: 8, color: '#fff', padding: '9px 20px', fontSize: 14, fontWeight: 700, cursor: loading ? 'default' : 'pointer', minWidth: 110 }}>
        {loading ? 'Загрузка…' : 'Анализ'}
      </button>
    </form>
  )
}

// ---------------------------------------------------------------------------
// 2. ALL-ROLES: Role Compare Panel
// ---------------------------------------------------------------------------

function RoleBar({ value, max, color }) {
  const pct = max > 0 ? Math.round(value / max * 100) : 0
  return (
    <div style={{ background: C.border, borderRadius: 4, height: 6, flex: 1 }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.5s' }} />
    </div>
  )
}

function RoleComparePanel({ data }) {
  const { role_summary = {}, best_role, summoner, rank, games_total } = data
  const roles = Object.entries(role_summary).sort((a, b) => b[1].winrate - a[1].winrate)
  const maxGames = Math.max(...roles.map(([, s]) => s.games), 1)
  const tierColor = TIER_COLOR[rank?.tier] ?? C.textDim

  return (
    <div style={COL}>
      {/* Header */}
      <div style={{ ...CARD, ...ROW, justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div>
          <h2 style={{ fontSize: 20, color: '#fff', marginBottom: 2 }}>{summoner}</h2>
          {rank && <span style={{ color: tierColor, fontWeight: 700, fontSize: 14 }}>{rank.tier} {rank.division} {rank.lp} LP</span>}
        </div>
        <div style={{ textAlign: 'right', color: C.textDim, fontSize: 13 }}>
          <div>{games_total} игр проанализировано</div>
          {best_role && <div style={{ color: C.success, fontWeight: 600, marginTop: 2 }}>Лучшая роль: {ROLE_RU[best_role] || best_role}</div>}
        </div>
      </div>

      {/* Role cards */}
      {roles.map(([role, s]) => {
        const isBest = role === best_role
        return (
          <div key={role} style={{ ...CARD, borderColor: isBest ? C.success : C.border, position: 'relative' }}>
            {isBest && (
              <div style={{ position: 'absolute', top: 10, right: 14, fontSize: 11, color: C.success, fontWeight: 700 }}>
                ★ ЛУЧШАЯ РОЛЬ
              </div>
            )}
            <div style={{ ...ROW, justifyContent: 'space-between', marginBottom: 10 }}>
              <div style={{ ...ROW, gap: 10 }}>
                <span style={{ fontSize: 17, fontWeight: 800, color: isBest ? C.success : C.text }}>{s.role_label || ROLE_RU[role] || role}</span>
                <Pill label={`${s.games} игр`} color={C.textDim} />
                <Pill label={`${s.winrate}% WR`} color={s.winrate >= 55 ? C.success : s.winrate >= 45 ? C.warn : C.danger} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px 20px' }}>
              {[
                ['CS/min',     s.cs_per_min,      ''],
                ['Vision/min', s.vision_per_min,  ''],
                ['Deaths',     s.deaths_per_game,  ''],
                ['KP',         s.kill_participation, '%'],
              ].map(([label, val, unit]) => (
                <div key={label} style={{ ...ROW, gap: 6, fontSize: 13 }}>
                  <span style={{ color: C.textDim, minWidth: 80 }}>{label}</span>
                  <span style={{ fontWeight: 700, color: C.text }}>{fmtNum(val)}{unit}</span>
                </div>
              ))}
            </div>

            {/* Games bar */}
            <div style={{ ...ROW, marginTop: 10, gap: 8 }}>
              <span style={{ fontSize: 11, color: C.textDim, minWidth: 56 }}>{s.games} игр</span>
              <RoleBar value={s.games} max={maxGames} color={isBest ? C.success : C.accent} />
            </div>
          </div>
        )
      })}

      {roles.length === 0 && (
        <div style={{ ...CARD, color: C.textDim, fontSize: 13 }}>
          Недостаточно данных по ролям. Попробуй увеличить количество игр.
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 3. Flags Bar
// ---------------------------------------------------------------------------
function FlagsBar({ coaching, rankDirection, patchChanged, fromCache }) {
  const flagMap = {
    rank_up:           { label: '🏆 Ранг ВЫРОС',        color: C.success },
    rank_down:         { label: '↘ Ранг УПАЛ',          color: C.danger  },
    patch_changed:     { label: '⚡ Смена патча',       color: C.warn    },
    stale_data:        { label: '⏰ Устаревшие данные',  color: C.textDim },
    static_benchmark:  { label: '📊 Статичный бенчмарк', color: C.textDim },
  }
  const flags = [...(coaching?.flags || [])]
  if (rankDirection === 'up')   flags.push('rank_up')
  if (rankDirection === 'down') flags.push('rank_down')
  if (patchChanged)             flags.push('patch_changed')
  const uniq = [...new Set(flags)]

  return (
    <div style={{ ...ROW, flexWrap: 'wrap', gap: 6 }}>
      {fromCache && <Pill label="⚡ Из кэша (новых игр нет)" color={C.textDim} />}
      {uniq.map(f => { const d = flagMap[f] ?? { label: f, color: C.textDim }; return <Pill key={f} label={d.label} color={d.color} /> })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 4. Summary Card
// ---------------------------------------------------------------------------
function MetricRow({ label, value, delta, trend, unit = '' }) {
  if (value == null) return null
  return (
    <div style={{ ...ROW, justifyContent: 'space-between', padding: '6px 0', borderBottom: `1px solid ${C.border}` }}>
      <span style={{ color: C.textDim, fontSize: 13, minWidth: 160 }}>{label}</span>
      <span style={{ fontWeight: 700, fontSize: 15 }}>{fmtNum(value)}{unit}</span>
      {delta && (
        <span style={{ color: quartileColor(delta.quartile), fontSize: 12, minWidth: 96, textAlign: 'right' }}>
          {quartileLabel(delta.quartile)}
          <span style={{ color: C.textDim }}> ({delta.delta_vs_median >= 0 ? '+' : ''}{fmtNum(delta.delta_vs_median)})</span>
        </span>
      )}
      {trend && (
        <span style={{ color: trendColor(trend.direction), fontSize: 14, minWidth: 24, textAlign: 'center' }}>
          {trendIcon(trend.direction)}
        </span>
      )}
    </div>
  )
}

function SummaryCard({ summary, benchmark_deltas, trends }) {
  return (
    <div style={CARD}>
      <SectionTitle>Статистика — Rolling 10 игр</SectionTitle>
      <MetricRow label="CS / мин"          value={summary?.cs_per_min}          delta={benchmark_deltas?.cs_per_min}         trend={trends?.cs_per_min} />
      <MetricRow label="Vision / мин"      value={summary?.vision_per_min}      delta={benchmark_deltas?.vision_per_min}     trend={trends?.vision_per_min} />
      <MetricRow label="Смертей / игру"    value={summary?.deaths_per_game}     delta={benchmark_deltas?.deaths}             trend={trends?.deaths} />
      <MetricRow label="Kill Participation" value={summary?.kill_participation} delta={benchmark_deltas?.kill_participation} trend={trends?.kill_participation} unit="%" />
      <MetricRow label="Доля урона"        value={summary?.damage_share}        unit="%" />
      <MetricRow label="Winrate"           value={summary?.winrate}             unit="%" />
      <MetricRow label="Сольных смертей до 10м" value={summary?.solo_deaths_early_avg} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// 5. Follow-up (прогресс по прошлому совету)
// ---------------------------------------------------------------------------
function FollowUp({ text, newGames }) {
  if (!text || !newGames) return null
  return (
    <div style={{ ...CARD, borderColor: C.purple, background: C.purple+'11' }}>
      <SectionTitle>📈 Прогресс по прошлому совету ({newGames} новых игр)</SectionTitle>
      <p style={{ fontSize: 13, color: C.text, lineHeight: 1.6 }}>{text}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 6. Primary Focus + Coaching Points
// ---------------------------------------------------------------------------
function PrimaryFocus({ text }) {
  if (!text) return null
  return (
    <div style={{ ...CARD, borderColor: C.accent, background: C.accent+'11' }}>
      <SectionTitle>Главный фокус</SectionTitle>
      <div style={{ fontSize: 18, fontWeight: 700, color: '#fff', lineHeight: 1.4 }}>{text}</div>
    </div>
  )
}

function CoachingPoint({ point }) {
  return (
    <div style={{ borderLeft: `3px solid ${quartileColor(point.quartile)}`, paddingLeft: 12, paddingTop: 4, paddingBottom: 4 }}>
      <div style={{ ...ROW, gap: 6, marginBottom: 4 }}>
        <span style={{ color: C.textDim, fontSize: 12, fontWeight: 700, textTransform: 'uppercase' }}>
          {point.metric?.replace(/_/g, ' ')}
        </span>
        <Pill label={quartileLabel(point.quartile)} color={quartileColor(point.quartile)} />
        <span style={{ color: trendColor(point.trend), fontSize: 13 }}>
          {trendIcon(point.trend)} {point.trend}
        </span>
      </div>
      <div style={{ fontSize: 13, color: C.text, lineHeight: 1.5 }}>{point.suggestion}</div>
    </div>
  )
}

function CoachingSection({ summary, coaching_points }) {
  return (
    <div style={CARD}>
      <SectionTitle>Совет тренера</SectionTitle>
      {summary && <p style={{ color: C.text, fontSize: 13, lineHeight: 1.6, marginBottom: 14 }}>{summary}</p>}
      <div style={{ ...COL, gap: 14 }}>
        {(coaching_points || []).map((pt, i) => <CoachingPoint key={i} point={pt} />)}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 7. Mistake Tracker
// ---------------------------------------------------------------------------
function MistakeRow({ mistake, onResolve }) {
  const sev = mistake.severity || 'minor'
  return (
    <div style={{ ...ROW, justifyContent: 'space-between', padding: '8px 0', borderBottom: `1px solid ${C.border}`, flexWrap: 'wrap', gap: 6 }}>
      <div style={COL}>
        <div style={{ ...ROW, gap: 6 }}>
          <Pill label={sev.toUpperCase()} color={severityColor(sev)} />
          <span style={{ fontSize: 12, color: C.textDim, fontWeight: 600 }}>{mistake.metric?.replace(/_/g, ' ')}</span>
        </div>
        <span style={{ fontSize: 13, color: C.text }}>{mistake.description}</span>
        <span style={{ fontSize: 11, color: C.textDim }}>{mistake.sessions_present} сессий подряд · {mistake.sessions_absent} пропущено</span>
      </div>
      <button onClick={() => onResolve(mistake.id)} style={{ background: 'transparent', border: `1px solid ${C.border}`, borderRadius: 6, color: C.textDim, padding: '4px 10px', fontSize: 11, cursor: 'pointer' }}>
        Решено
      </button>
    </div>
  )
}

function MistakeTracker({ mistakes, onResolve }) {
  if (!mistakes?.length) {
    return (
      <div style={CARD}>
        <SectionTitle>Трекер ошибок</SectionTitle>
        <span style={{ color: C.textDim, fontSize: 13 }}>Активных ошибок нет — продолжай в том же духе!</span>
      </div>
    )
  }
  return (
    <div style={CARD}>
      <SectionTitle>Трекер ошибок ({mistakes.length} активных)</SectionTitle>
      {mistakes.map(m => <MistakeRow key={m.id} mistake={m} onResolve={onResolve} />)}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 8. Champion Stats Panel
// ---------------------------------------------------------------------------
function WinrateBar({ winrate }) {
  const color = winrate >= 55 ? C.success : winrate >= 50 ? '#84cc16' : winrate >= 45 ? C.warn : C.danger
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ background: C.border, borderRadius: 3, height: 5, width: 56, flexShrink: 0 }}>
        <div style={{ width: `${Math.min(winrate, 100)}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 13, minWidth: 42 }}>{winrate}%</span>
    </div>
  )
}

function ChampionStatsPanel({ champion_stats }) {
  if (!champion_stats?.length) return null
  const top = champion_stats.slice(0, 6)   // показываем не более 6 чемпионов

  return (
    <div style={CARD}>
      <SectionTitle>Чемпионы на роли</SectionTitle>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ color: C.textDim, fontSize: 11, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              {['Чемпион', 'Игр', 'Винрейт', 'KDA', 'CS/мин', 'Vision/мин'].map(h => (
                <th key={h} style={{ textAlign: h === 'Чемпион' ? 'left' : 'center', padding: '4px 10px', borderBottom: `1px solid ${C.border}`, fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {top.map((c, i) => {
              const kdaColor = c.kda >= 3 ? C.success : c.kda >= 2 ? '#84cc16' : c.kda >= 1 ? C.warn : C.danger
              return (
                <tr key={c.champion} style={{ borderBottom: `1px solid ${C.border}33`, background: i === 0 ? C.accent + '0a' : 'transparent' }}>
                  <td style={{ padding: '8px 10px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {i === 0 && <span style={{ fontSize: 10, color: C.accent, fontWeight: 700 }}>★</span>}
                      <span style={{ fontWeight: 700, color: '#fff' }}>{c.champion}</span>
                    </div>
                  </td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: C.textDim }}>
                    {c.games}
                  </td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                    <WinrateBar winrate={c.winrate} />
                  </td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 700, color: kdaColor }}>
                    {fmtNum(c.kda)}
                    <span style={{ display: 'block', fontSize: 10, color: C.textDim, fontWeight: 400 }}>
                      {fmtNum(c.kills, 1)}/{fmtNum(c.deaths, 1)}/{fmtNum(c.assists, 1)}
                    </span>
                  </td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: C.text }}>{fmtNum(c.cs_per_min)}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: C.text }}>{fmtNum(c.vision_per_min)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 9. Benchmark Panel
// ---------------------------------------------------------------------------
function BenchmarkPanel({ benchmark }) {
  if (!benchmark) return null
  const { source, sample_size, winrate, stale, cs_per_min, vision_score_per_min, deaths_per_game, kill_participation } = benchmark
  return (
    <div style={CARD}>
      <SectionTitle>Прозрачность бенчмарка</SectionTitle>
      <div style={{ ...ROW, gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <Pill label={`Источник: ${source}`} color={source === 'static' ? C.textDim : C.accent} />
        <Pill label={`Выборка: ${(sample_size || 0).toLocaleString('ru')}`} color={C.textDim} />
        <Pill label={`WR бенчмарка: ${fmtNum(winrate, 1)}%`} color={C.textDim} />
        {stale && <Pill label="УСТАРЕВШИЙ" color={C.warn} />}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, color: C.textDim }}>
        <thead>
          <tr>{['Метрика', 'p25', 'p50 (медиана)', 'p75'].map(h => (
            <th key={h} style={{ textAlign: 'left', padding: '4px 8px', borderBottom: `1px solid ${C.border}` }}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>
          {[['CS / мин', cs_per_min], ['Vision / мин', vision_score_per_min], ['Смерти', deaths_per_game], ['KP %', kill_participation]]
            .map(([label, p]) => p && (
              <tr key={label}>
                <td style={{ padding: '3px 8px' }}>{label}</td>
                <td style={{ padding: '3px 8px' }}>{fmtNum(p.p25)}</td>
                <td style={{ padding: '3px 8px', color: C.text, fontWeight: 600 }}>{fmtNum(p.p50)}</td>
                <td style={{ padding: '3px 8px' }}>{fmtNum(p.p75)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 10. Confidence Bar
// ---------------------------------------------------------------------------
function ConfidenceBar({ confidence }) {
  if (confidence == null) return null
  const pct   = Math.round(confidence * 100)
  const color = pct >= 75 ? C.success : pct >= 45 ? C.warn : C.danger
  return (
    <div style={{ ...COL, gap: 4, minWidth: 160 }}>
      <div style={{ ...ROW, justifyContent: 'space-between' }}>
        <span style={{ fontSize: 11, color: C.textDim, fontWeight: 700, letterSpacing: '0.08em' }}>ДОСТОВЕРНОСТЬ</span>
        <span style={{ fontSize: 12, color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ background: C.border, borderRadius: 4, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Result Panels
// ---------------------------------------------------------------------------
function SingleRoleResult({ data, onResolve }) {
  const tierColor = TIER_COLOR[data.rank?.tier] ?? C.textDim
  return (
    <div style={{ ...COL, gap: 16 }}>
      {/* Header */}
      <div style={{ ...ROW, justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h2 style={{ fontSize: 20, color: '#fff', marginBottom: 2 }}>{data.summoner}</h2>
          <span style={{ color: C.textDim, fontSize: 13 }}>
            {data.region?.toUpperCase()} · {ROLE_RU[data.role] || data.role} · {data.tier}
            {data.rank && <span style={{ color: tierColor, fontWeight: 600 }}> · {data.rank.tier} {data.rank.division} {data.rank.lp} LP</span>}
            {' '}· Патч {data.patch}
            {' '}· {data.games_used}/{data.games_analyzed} игр на роли
            {data.games_searched > data.games_analyzed && <span style={{ color: C.textDim }}> (проверено {data.games_searched})</span>}
            {data.new_games_since_prev > 0 && <span style={{ color: C.success }}> · +{data.new_games_since_prev} новых</span>}
          </span>
        </div>
        <ConfidenceBar confidence={data.coaching?.confidence} />
      </div>

      <FlagsBar coaching={data.coaching} rankDirection={data.rank_direction} patchChanged={data.patch_changed} fromCache={data.from_cache} />

      {data.low_sample && (
        <div style={{ ...CARD, borderColor: C.warn, background: C.warn + '11', display: 'flex', gap: 10, alignItems: 'flex-start' }}>
          <span style={{ fontSize: 18 }}>⚠️</span>
          <div>
            <div style={{ fontWeight: 700, color: C.warn, fontSize: 13, marginBottom: 2 }}>
              Мало данных — найдено {data.role_games_found} из {10} нужных игр
              {data.games_searched > data.role_games_found && ` (проверено ${data.games_searched} матчей)`}
            </div>
            <div style={{ color: C.textDim, fontSize: 12 }}>
              Тренды и бенчмарк-сравнение могут быть неточными. Сыграй больше игр на этой роли для полноценного анализа.
            </div>
          </div>
        </div>
      )}

      <FollowUp text={data.coaching?.follow_up} newGames={data.new_games_since_prev} />

      <PrimaryFocus text={data.coaching?.primary_focus} />

      <SummaryCard summary={data.summary} benchmark_deltas={data.benchmark_deltas} trends={data.trends} />

      <ChampionStatsPanel champion_stats={data.champion_stats} />

      <CoachingSection summary={data.coaching?.summary} coaching_points={data.coaching?.coaching_points} />

      <MistakeTracker mistakes={data.active_mistakes} onResolve={onResolve} />

      <BenchmarkPanel benchmark={data.benchmark} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Insufficient data fallback
// ---------------------------------------------------------------------------
function InsufficientDataCard({ data }) {
  return (
    <div style={{ ...CARD, borderColor: C.warn }}>
      <div style={{ ...ROW, gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: 22 }}>🎮</span>
        <div>
          <div style={{ fontWeight: 700, color: '#fff', fontSize: 16 }}>{data.summoner}</div>
          <div style={{ color: C.textDim, fontSize: 12 }}>{ROLE_RU[data.role] || data.role} · проверено {data.games_searched} игр</div>
        </div>
      </div>
      <p style={{ color: C.warn, fontSize: 14, lineHeight: 1.6 }}>{data.message}</p>
      <p style={{ color: C.textDim, fontSize: 12, marginTop: 8 }}>
        Для полноценного анализа нужно минимум 10 игр на роли. Попробуй выбрать другую роль или сыграй больше на этой.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Error / Loading
// ---------------------------------------------------------------------------
function ErrorBox({ message }) {
  return <div style={{ ...CARD, borderColor: C.danger, color: C.danger, fontSize: 13 }}><strong>Ошибка: </strong>{message}</div>
}

function Spinner() {
  return (
    <div style={{ textAlign: 'center', padding: 40, color: C.textDim, fontSize: 14 }}>
      <div style={{ display: 'inline-block', width: 32, height: 32, border: `3px solid ${C.border}`, borderTopColor: C.accent, borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      <p style={{ marginTop: 12 }}>Загружаем матчи и бенчмарки…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
export default function App() {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [data,    setData]    = useState(null)

  const handleSubmit = useCallback(async ({ summoner, region, role, count }) => {
    setLoading(true); setError(null); setData(null)
    const params = new URLSearchParams({ summoner, region, role, count })
    try {
      const res = await fetch(`${API}/analyze?${params}`)
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.detail || `HTTP ${res.status}`) }
      setData(await res.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  const handleResolve = useCallback(async (mistakeId) => {
    try {
      const res = await fetch(`${API}/mistakes/resolve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mistake_id: mistakeId }) })
      if (!res.ok) throw new Error('Не удалось закрыть ошибку')
      setData(prev => prev ? { ...prev, active_mistakes: (prev.active_mistakes || []).filter(m => m.id !== mistakeId) } : prev)
    } catch (e) { setError(e.message) }
  }, [])

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '24px 16px 60px', ...COL, gap: 20 }}>
      <div>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: '#fff', letterSpacing: '-0.02em' }}>LoL AI Тренер</h1>
        <p style={{ color: C.textDim, fontSize: 13, marginTop: 4 }}>Riot API · OP.GG · Claude AI · русский язык</p>
      </div>

      <InputForm onSubmit={handleSubmit} loading={loading} />

      {loading && <Spinner />}
      {error   && <ErrorBox message={error} />}

      {data && data.mode === 'all_roles'    && <RoleComparePanel data={data} />}
      {data && data.insufficient_data      && <InsufficientDataCard data={data} />}
      {data && !data.mode && !data.insufficient_data && <SingleRoleResult data={data} onResolve={handleResolve} />}
    </div>
  )
}
