"""Retail Brain business-simulation engine (no Streamlit dependency, unit-testable).

Adds a day-by-day simulation of calendar year 2026 on top of ``retail_brain_core``:

1. ClockState + pure transition functions (start/resume/pause/speed/advance/year-end).
2. DayIndex: resolves every (date, category) to an observed CSV row (Case A), a documented
   estimate (Case B) or an explicit "insufficient evidence" marker (Case C).
3. Vectorised forecasting that mirrors the nine agents' formulas (parity is unit-tested).
4. Three separate forecast concepts: frozen annual baseline, current operational forecast
   (baseline + approved, material decisions) and realized year-to-date ledger.
5. Day lifecycle: begin_day (start-of-day notices) and finalize_day (outcomes, inventory ledger,
   alerts), both idempotent.
6. SimulationStore: SQLite persistence (compare-and-set clock, idempotent ledger) so a browser
   refresh or app restart never loses the business history.

Nothing here writes to the source CSV.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

import retail_brain_core as core

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
READY, RUNNING, PAUSED, FINISHED = "ready", "running", "paused", "finished"
DEFAULT_SECONDS_PER_DAY = 30.0
SPEED_PRESETS = {"30 s per day (default)": 30.0, "15 s per day": 15.0, "5 s per day": 5.0}
HEARTBEAT_GRACE_S = 15.0          # a running clock whose app went quiet longer than this is auto-paused on reload
MAX_CATCHUP_DAYS = 12             # days finalized per tick when the browser tab was throttled
INITIAL_STOCK_DAYS_COVER = 1.0    # day-1 opening stock = this many days of forecast demand + safety buffer
MATERIALITY_PCT = 0.5             # a decision adjusts the forecast only if it moves remaining units or revenue by >= this %

SOURCE_OBSERVED = "Observed (CSV)"
SOURCE_ADJUSTED = "Observed + approved decision"
SOURCE_ESTIMATED = "Estimated (no CSV record)"
SOURCE_MISSING = "No data (insufficient evidence)"

OUTCOME_PREFIXES = ("AI_",)
OUTCOME_COLUMNS = ("Demand_QTY", "Actual_Sales_QTY", "Demand_Value_EUR", "Actual_Sales_Value_EUR")

FORECAST_COLS = ["date", "category", "source", "demand", "demand_low", "demand_high", "sales", "lost", "revenue",
                 "gross_profit", "waste_cost", "net_profit", "safety_stock", "staffing_pressure", "stockout_pct", "traffic"]
_NUM_FORECAST_COLS = FORECAST_COLS[3:]


def iso(d: Any) -> str:
    return pd.Timestamp(d).date().isoformat()


def _is_outcome(col: str) -> bool:
    return col in OUTCOME_COLUMNS or col.startswith(OUTCOME_PREFIXES)


def _clean(obj: Any) -> Any:
    """Make an object JSON-safe: numpy scalars to python, NaN/inf to None, Timestamps to ISO text."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, _date)):
        return pd.Timestamp(obj).isoformat()
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_clean(obj), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Clock (pure functions; time is always passed in so tests are deterministic)
# ---------------------------------------------------------------------------
@dataclass
class ClockState:
    run_id: str
    start_date: str
    end_date: str
    current_date: str
    day_number: int = 1
    status: str = READY
    seconds_per_day: float = DEFAULT_SECONDS_PER_DAY
    elapsed_in_day: float = 0.0            # seconds of the current day already used before last_resume_ts
    last_resume_ts: float | None = None    # epoch seconds when the clock last (re)started running
    day_started: str = ""                  # date for which start-of-day processing has run
    notice: str = ""
    version: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "ClockState":
        data = json.loads(text)
        return ClockState(**data)

    @property
    def total_days(self) -> int:
        return (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days + 1

    @property
    def is_running(self) -> bool:
        return self.status == RUNNING


def new_clock(start_date: Any, seconds_per_day: float = DEFAULT_SECONDS_PER_DAY, end_date: Any | None = None,
              run_id: str | None = None) -> ClockState:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp(year=start.year, month=12, day=31)
    if end < start:
        raise ValueError("end date precedes start date")
    return ClockState(run_id=run_id or uuid.uuid4().hex[:12], start_date=iso(start), end_date=iso(end),
                      current_date=iso(start), seconds_per_day=float(seconds_per_day))


def day_elapsed(s: ClockState, now: float) -> float:
    extra = max(0.0, now - s.last_resume_ts) if s.status == RUNNING and s.last_resume_ts is not None else 0.0
    return s.elapsed_in_day + extra


def day_progress(s: ClockState, now: float) -> float:
    if s.status == FINISHED:
        return 1.0
    return float(min(1.0, day_elapsed(s, now) / max(s.seconds_per_day, 1e-9)))


def seconds_to_next_day(s: ClockState, now: float) -> float:
    return max(0.0, s.seconds_per_day - day_elapsed(s, now))


def start_or_resume(s: ClockState, now: float) -> ClockState:
    if s.status in (RUNNING, FINISHED):
        return s
    return replace(s, status=RUNNING, last_resume_ts=now, notice="", version=s.version + 1)


def pause(s: ClockState, now: float, notice: str = "") -> ClockState:
    if s.status != RUNNING:
        return s
    return replace(s, status=PAUSED, elapsed_in_day=day_elapsed(s, now), last_resume_ts=None, notice=notice,
                   version=s.version + 1)


def set_speed(s: ClockState, seconds_per_day: float, now: float) -> ClockState:
    seconds_per_day = float(min(max(seconds_per_day, 1.0), 600.0))
    if abs(seconds_per_day - s.seconds_per_day) < 1e-9:
        return s
    frac = day_progress(s, now) if s.status != FINISHED else 1.0
    elapsed = frac * seconds_per_day
    return replace(s, seconds_per_day=seconds_per_day, elapsed_in_day=elapsed,
                   last_resume_ts=now if s.status == RUNNING else None, version=s.version + 1)


def day_due(s: ClockState, now: float) -> bool:
    return s.status == RUNNING and day_elapsed(s, now) >= s.seconds_per_day


def advance(s: ClockState, now: float) -> ClockState:
    """Close the current day and open the next one (or enter the year-end state)."""
    if not day_due(s, now):
        raise ValueError("advance() called before the day is due")
    leftover = max(0.0, day_elapsed(s, now) - s.seconds_per_day)
    if pd.Timestamp(s.current_date) >= pd.Timestamp(s.end_date):
        return replace(s, status=FINISHED, elapsed_in_day=0.0, last_resume_ts=None, version=s.version + 1,
                       notice="Year-end reached: the simulation is complete.")
    nxt = pd.Timestamp(s.current_date) + pd.Timedelta(days=1)
    return replace(s, current_date=iso(nxt), day_number=s.day_number + 1, elapsed_in_day=leftover,
                   last_resume_ts=now, version=s.version + 1)


def auto_pause_if_stale(s: ClockState, heartbeat_ts: float | None, now: float, grace: float = HEARTBEAT_GRACE_S) -> ClockState:
    """A 'running' clock whose browser session vanished must not silently skip days on the next visit."""
    if s.status != RUNNING or heartbeat_ts is None or s.last_resume_ts is None:
        return s
    if now - heartbeat_ts <= grace:
        return s
    used = s.elapsed_in_day + max(0.0, heartbeat_ts - s.last_resume_ts)
    return replace(s, status=PAUSED, elapsed_in_day=min(used, s.seconds_per_day * 0.999), last_resume_ts=None,
                   notice="The simulation was paused automatically while the app was closed. Press Resume to continue.",
                   version=s.version + 1)


# ---------------------------------------------------------------------------
# 2. Day resolution: observed (A), estimated (B) or insufficient evidence (C)
# ---------------------------------------------------------------------------
@dataclass
class Resolved:
    row: pd.Series | None
    source: str
    confidence: str
    note: str = ""


class DayIndex:
    """Lookup of CSV observations with documented estimation for dates the CSV does not cover."""

    TIER1_WINDOW_DAYS, TIER1_MIN_ROWS = 28, 3     # same weekday, +/-28 days of the year
    TIER2_WINDOW_DAYS, TIER2_MIN_ROWS = 45, 10    # any weekday, +/-45 days of the year

    def __init__(self, df: pd.DataFrame):
        d = df.copy()
        d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
        self.df = d.reset_index(drop=True)
        self.categories: list[str] = sorted(d["Category"].astype(str).unique())
        self._key = {(ts.date().isoformat(), str(c)): i for i, (ts, c) in enumerate(zip(self.df["Date"], self.df["Category"]))}
        self._frames: dict[str, pd.DataFrame] = {}
        for c, g in self.df.groupby("Category"):
            g = g.copy()
            g["_doy"], g["_dow"] = g["Date"].dt.dayofyear, g["Date"].dt.dayofweek
            self._frames[str(c)] = g
        self._cache: dict[tuple[str, str], Resolved] = {}
        self.first_date, self.last_date = self.df["Date"].min(), self.df["Date"].max()

    def has(self, date_iso: str, category: str) -> bool:
        return (date_iso, category) in self._key

    def resolve(self, date_iso: str, category: str) -> Resolved:
        key = (date_iso, category)
        if key in self._cache:
            return self._cache[key]
        if key in self._key:
            res = Resolved(self.df.iloc[self._key[key]], SOURCE_OBSERVED, "Observed")
        else:
            res = self._estimate(date_iso, category)
        self._cache[key] = res
        return res

    def _estimate(self, date_iso: str, category: str) -> Resolved:
        g = self._frames.get(category)
        if g is None:
            return Resolved(None, SOURCE_MISSING, "None", "Category has no history in the CSV.")
        ts = pd.Timestamp(date_iso)
        dd = (g["_doy"] - ts.dayofyear).abs()
        dd = np.minimum(dd, 366 - dd)
        sel = g[(g["_dow"] == ts.dayofweek) & (dd <= self.TIER1_WINDOW_DAYS)]
        if len(sel) >= self.TIER1_MIN_ROWS:
            tier, conf = (f"mean of {len(sel)} same-weekday CSV rows within +/-{self.TIER1_WINDOW_DAYS} days of the year", "Medium")
        else:
            sel = g[dd <= self.TIER2_WINDOW_DAYS]
            if len(sel) >= self.TIER2_MIN_ROWS:
                tier, conf = (f"mean of {len(sel)} CSV rows within +/-{self.TIER2_WINDOW_DAYS} days of the year (no weekday effect)", "Low")
            else:
                return Resolved(None, SOURCE_MISSING, "None",
                                f"Only {len(sel)} comparable CSV rows; at least {self.TIER1_MIN_ROWS} (weekday) or "
                                f"{self.TIER2_MIN_ROWS} (seasonal) are needed.")
        row: dict[str, Any] = {}
        for col in self.df.columns:
            if col == "Date":
                row[col] = ts
            elif col == "Category":
                row[col] = category
            elif _is_outcome(col):
                row[col] = np.nan                      # never invent observed outcomes
            elif col in ("Event_Season", "Our_Active_Promotion"):
                row[col] = "None"                      # events/campaigns are only those recorded in the CSV
            elif col == "Cold_Chain_Failure_Flag":
                row[col] = "No"
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                row[col] = float(sel[col].mean())
            else:
                mode = sel[col].mode()
                row[col] = str(mode.iloc[0]) if len(mode) else "None"
        row["Store_Traffic_Volume"] = float(round(row["Store_Traffic_Volume"]))
        return Resolved(pd.Series(row), SOURCE_ESTIMATED, conf, f"Estimated as the {tier}.")


def data_coverage(index: DayIndex, start: Any, end: Any) -> dict[str, Any]:
    """Coverage of the simulated calendar by CSV observations (all counts are explicit)."""
    dates = [iso(d) for d in pd.date_range(start, end)]
    cats = index.categories
    full = partial = none = 0
    cd_obs = cd_est = cd_missing = 0
    estimation_dates = set()
    for d in dates:
        n_obs = 0
        for c in cats:
            r = index.resolve(d, c)
            if r.source == SOURCE_OBSERVED:
                n_obs += 1
                cd_obs += 1
            elif r.source == SOURCE_ESTIMATED:
                cd_est += 1
                estimation_dates.add(d)
            else:
                cd_missing += 1
                estimation_dates.add(d)
        if n_obs == len(cats):
            full += 1
        elif n_obs:
            partial += 1
        else:
            none += 1
    in_window = index.df[(index.df["Date"] >= pd.Timestamp(start)) & (index.df["Date"] <= pd.Timestamp(end))]
    outside = len(index.df) - len(in_window)
    warnings: list[str] = []
    if outside:
        warnings.append(f"{outside} CSV row(s) fall outside the simulated window and are used only as history for estimates.")
    if partial:
        warnings.append(f"{partial} date(s) have observations for only some categories.")
    if none:
        warnings.append(f"{none} simulated date(s) have no CSV observation at all; {len(estimation_dates)} date(s) need estimates or are flagged as missing.")
    n = len(dates)
    return {
        "csv_earliest": index.first_date.date().isoformat(), "csv_latest": index.last_date.date().isoformat(),
        "sim_days": n, "dates_full": full, "dates_partial": partial, "dates_none": none,
        "pct_dates_with_observations": 100.0 * (full + partial) / n if n else 0.0,
        "pct_dates_fully_observed": 100.0 * full / n if n else 0.0,
        "dates_requiring_estimation": len(estimation_dates),
        "category_days_observed": cd_obs, "category_days_estimated": cd_est, "category_days_missing": cd_missing,
        "pct_category_days_observed": 100.0 * cd_obs / (n * len(cats)) if n and cats else 0.0,
        "rows_total": int(len(index.df)), "rows_outside_window": int(outside), "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 3. Plan entries and vectorised forecasting (mirrors the nine agents; parity unit-tested)
# ---------------------------------------------------------------------------
@dataclass
class PlanEntry:
    date: str
    category: str
    source: str
    confidence: str
    note: str
    profile: dict[str, Any] | None
    obs: dict[str, Any] | None


def build_year_plan(index: DayIndex, model: core.DemandModel, start: Any, end: Any,
                    categories: Iterable[str] | None = None) -> list[PlanEntry]:
    cats = list(categories or index.categories)
    out: list[PlanEntry] = []
    for d in pd.date_range(start, end):
        di = iso(d)
        for c in cats:
            r = index.resolve(di, c)
            if r.row is None:
                out.append(PlanEntry(di, c, r.source, r.confidence, r.note, None, None))
                continue
            prof = core.profile_from_row(r.row, model)
            obs = None
            if r.source == SOURCE_OBSERVED:
                row = r.row
                obs = {"demand": float(row["Demand_QTY"]), "sales": float(row["Actual_Sales_QTY"]),
                       "sales_value": float(row["Actual_Sales_Value_EUR"]) if "Actual_Sales_Value_EUR" in row.index and pd.notna(row["Actual_Sales_Value_EUR"]) else None,
                       "utility": float(row["Utility_Cost_Today_EUR"]) if "Utility_Cost_Today_EUR" in row.index and pd.notna(row["Utility_Cost_Today_EUR"]) else None}
            out.append(PlanEntry(di, c, r.source, r.confidence, r.note, prof, obs))
    return out


def waste_units(p: dict[str, Any], sales: float, demand: float) -> float:
    """Same formula as FreshnessWasteAgent, applied to any sales/demand volume."""
    perishable = p["shelf_life_days"] <= 7
    rate = p["base_waste_rate"] * ((1 + max(0.0, p["temperature"] - 25) * 0.04) if perishable else 1.0)
    units = sales * rate
    if p["cold_chain"]:
        units += core.COLD_CHAIN_WRITEOFF_SHARE * demand
    return float(units)


def staffing_pressure(model: core.DemandModel, p: dict[str, Any]) -> float:
    typical = model.median_traffic.get(p["category"], model.mean_traffic)
    load = p["traffic"] / max(typical, 1.0)
    return float(np.clip(load / max(p["staff_capacity_pct"] / 100.0, 0.2) - 1.0, 0, 2))


def forecast_profiles(model: core.DemandModel, levers: dict[str, Any], profiles: list[dict[str, Any]]) -> pd.DataFrame:
    """Batch version of DemandForecast + InventoryRisk + Replenishment + Freshness + Workforce + Profitability agents."""
    if not profiles:
        return pd.DataFrame(columns=_NUM_FORECAST_COLS)
    frame = model.context_frame(profiles)
    rate = model._rate(frame, profiles)
    traffic = np.array([float(p["traffic"]) for p in profiles])
    ctx = rate * traffic
    adj = np.array([model.lever_adjustment(p, levers) for p in profiles])
    pa, pr = adj[:, 0], adj[:, 1]
    q = model.interval_q
    demand = ctx * pa * pr
    low = np.maximum(0.0, ctx * (1 - q)) * pa * pr
    high = ctx * (1 + q) * pa * pr
    fill = np.array([model.fill_rate(p["delay_hrs"]) for p in profiles])
    lost = demand * (1 - fill)
    sales = demand - lost
    safety = np.maximum(0.0, (high - demand) + lost)
    price = np.array([float(p["price"]) for p in profiles])
    cost = np.array([float(p["unit_cost"]) for p in profiles])
    wu = np.array([waste_units(p, s, d) for p, s, d in zip(profiles, sales, demand)])
    waste_cost = wu * cost
    gross = sales * (price - cost)
    return pd.DataFrame({
        "demand": demand, "demand_low": low, "demand_high": high, "sales": sales, "lost": lost,
        "revenue": sales * price, "gross_profit": gross, "waste_cost": waste_cost, "net_profit": gross - waste_cost,
        "safety_stock": safety, "staffing_pressure": [staffing_pressure(model, p) for p in profiles],
        "stockout_pct": (1 - fill) * 100.0, "traffic": traffic})


# ---------------------------------------------------------------------------
# 4. Approved decisions applied to profiles
# ---------------------------------------------------------------------------
def decision_effective(decision: dict[str, Any]) -> str:
    return str(decision.get("effective_date") or "0001-01-01")


def adjust_profile(profile: dict[str, Any], decisions: Iterable[dict[str, Any]], date_iso: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Apply approved decisions effective on/before ``date_iso`` (oldest first).

    A decision that sets a calendar event is skipped on days where the CSV already records an event, so an
    event uplift is never counted twice. Returns (profile, applied decision ids, skipped decision ids).
    """
    out, applied, skipped = profile, [], []
    day_has_event = str(profile.get("event", "None")).lower() != "none"
    for d in decisions:
        if d.get("status") != "approved" or d.get("category") != profile["category"] or decision_effective(d) > date_iso:
            continue
        ch = d.get("changes", {})
        if "event_set" in ch and day_has_event:
            skipped.append(str(d.get("id")))
            continue
        out = core.apply_changes(out, ch)
        applied.append(str(d.get("id")))
    return out, applied, skipped


def forecast_table(model: core.DemandModel, levers: dict[str, Any], entries: list[PlanEntry],
                   decisions: Iterable[dict[str, Any]] = ()) -> pd.DataFrame:
    decisions = list(decisions)
    profiles, idx = [], []
    for i, e in enumerate(entries):
        if e.profile is None:
            continue
        p = adjust_profile(e.profile, decisions, e.date)[0] if decisions else e.profile
        profiles.append(p)
        idx.append(i)
    num = forecast_profiles(model, levers, profiles)
    base = pd.DataFrame({"date": [e.date for e in entries], "category": [e.category for e in entries],
                         "source": [e.source for e in entries]})
    full = pd.DataFrame(np.nan, index=range(len(entries)), columns=_NUM_FORECAST_COLS)
    if idx:
        full.iloc[idx] = num.to_numpy()
    return pd.concat([base, full], axis=1)[FORECAST_COLS]


# ---------------------------------------------------------------------------
# 5. Baseline (frozen), operational forecast, adjustments
# ---------------------------------------------------------------------------
def table_checksum(df: pd.DataFrame) -> str:
    return hashlib.sha1(df[FORECAST_COLS].round(6).to_csv(index=False).encode()).hexdigest()[:16]


def freeze_baseline(table: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    return {"checksum": table_checksum(table), "meta": _clean(meta),
            "table": table.to_json(orient="split", double_precision=12)}


def baseline_table(payload: dict[str, Any]) -> pd.DataFrame:
    from io import StringIO
    t = pd.read_json(StringIO(payload["table"]), orient="split", dtype=False)
    t["date"] = t["date"].astype(str).str[:10]
    return t[FORECAST_COLS]


def baseline_is_intact(payload: dict[str, Any]) -> bool:
    return table_checksum(baseline_table(payload)) == payload["checksum"]


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    t = table.copy()
    t["month"] = t["date"].str[:7]
    agg = {"demand": "sum", "sales": "sum", "lost": "sum", "revenue": "sum", "gross_profit": "sum", "waste_cost": "sum",
           "net_profit": "sum", "demand_low": "sum", "demand_high": "sum", "safety_stock": "mean", "staffing_pressure": "mean"}
    by_cat = t.groupby("category").agg(agg).reset_index()
    by_month = t.groupby("month").agg({"demand": "sum", "sales": "sum", "revenue": "sum", "gross_profit": "sum"}).reset_index()
    total = {k: float(t[k].sum()) for k in ("demand", "sales", "lost", "revenue", "gross_profit", "waste_cost", "net_profit", "demand_low", "demand_high")}
    total["avg_safety_stock_per_cat_day"] = float(t["safety_stock"].mean())
    total["avg_staffing_pressure"] = float(t["staffing_pressure"].mean())
    total["days"] = int(t["date"].nunique())
    total["category_days_estimated"] = int((t["source"] == SOURCE_ESTIMATED).sum())
    total["category_days_missing"] = int((t["source"] == SOURCE_MISSING).sum())
    return {"total": total, "by_category": by_cat, "by_month": by_month}


def operational_forecast(model, levers, plan: list[PlanEntry], decisions: Iterable[dict[str, Any]], from_date: str) -> pd.DataFrame:
    """Forecast of the days on/after ``from_date`` with approved decisions applied (concept B)."""
    return forecast_table(model, levers, [e for e in plan if e.date >= from_date], decisions)


def evaluate_adjustment(model, levers, plan: list[PlanEntry], prior_decisions: list[dict[str, Any]],
                        decision: dict[str, Any], from_date: str, created_sim_date: str,
                        materiality_pct: float = MATERIALITY_PCT) -> dict[str, Any]:
    """Quantify an approved decision's effect on the remaining-year forecast and decide whether it is material."""
    cat = decision["category"]
    eff = max(decision_effective(decision), from_date)
    # All remaining days of the category: the decision only acts from its effective date, but materiality is judged
    # against the whole remaining-year forecast, not just the days it touches.
    entries = [e for e in plan if e.category == cat and e.date >= from_date]
    without = forecast_table(model, levers, entries, prior_decisions)
    with_ = forecast_table(model, levers, entries, list(prior_decisions) + [decision])
    d_units = float(with_["demand"].sum() - without["demand"].sum())
    d_rev = float(with_["revenue"].sum() - without["revenue"].sum())
    d_gp = float(with_["gross_profit"].sum() - without["gross_profit"].sum())
    pct_u = 100.0 * d_units / max(float(without["demand"].sum()), 1e-9)
    pct_r = 100.0 * d_rev / max(float(without["revenue"].sum()), 1e-9)
    n_skip = sum(1 for e in entries if e.profile is not None and adjust_profile(e.profile, [decision], e.date)[2])
    material = max(abs(pct_u), abs(pct_r)) >= materiality_pct
    reason = (f"Approved decision '{decision.get('template', 'decision')}' ({decision.get('change_summary', '')}) "
              f"changes expected remaining demand by {pct_u:+.2f}% and revenue by {pct_r:+.2f}% in {cat}.")
    if not material:
        reason += f" Below the {materiality_pct:g}% materiality threshold, so the operational forecast is not adjusted."
    if n_skip:
        reason += f" Event uplift already recorded in the CSV was not applied again on {n_skip} day(s)."
    return {
        "id": str(decision.get("id")), "created_sim_date": created_sim_date, "effective_date": eff,
        "decision_id": str(decision.get("id")), "template": decision.get("template"), "category": cat,
        "change_summary": decision.get("change_summary", ""), "applied": bool(material), "reason": reason,
        "delta_units": d_units, "delta_revenue": d_rev, "delta_gross_profit": d_gp,
        "pct_units": pct_u, "pct_revenue": pct_r, "affected_categories": [cat],
        "days_affected": int(sum(1 for e in entries if e.date >= eff)),
        "method": ("Remaining days of the category are re-forecast with the same demand model and levers after applying the "
                   "decision's profile changes from its effective date; difference vs the forecast without it. "
                   "Estimate, not a guaranteed outcome."),
    }


def active_adjustments(adjustments: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decisions that move the operational forecast: approved and flagged material when they were evaluated."""
    material = {a["decision_id"] for a in adjustments if a.get("applied")}
    return [d for d in decisions if d.get("status") == "approved" and str(d.get("id")) in material]


# ---------------------------------------------------------------------------
# 6. Day lifecycle
# ---------------------------------------------------------------------------
def _alert(date_iso: str, phase: str, severity: str, category: str, kind: str, message: str, recommendation: str = "") -> dict[str, Any]:
    return {"key": f"{phase}|{date_iso}|{category}|{kind}", "date": date_iso, "phase": phase, "severity": severity,
            "category": category, "kind": kind, "message": message, "recommendation": recommendation}


def begin_day_notices(date_iso: str, entries: list[PlanEntry], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Start-of-day information: recorded events, decisions taking effect, incidents and data caveats."""
    out: list[dict[str, Any]] = []
    for e in entries:
        p = e.profile
        if e.source == SOURCE_ESTIMATED:
            out.append(_alert(date_iso, "start", "info", e.category, "estimated_day",
                              f"No CSV record for {e.category}; the day is estimated ({e.confidence} confidence).", e.note))
        elif e.source == SOURCE_MISSING:
            out.append(_alert(date_iso, "start", "warning", e.category, "missing_data",
                              f"Insufficient evidence for {e.category}: no actual or estimated values are shown.", e.note))
        if p is None:
            continue
        if str(p["event"]).lower() != "none":
            out.append(_alert(date_iso, "start", "info", e.category, "csv_event", f"Recorded event today: {p['event']}.",
                              "Plan staffing and stock for the recorded event."))
        if p["promo_active"]:
            out.append(_alert(date_iso, "start", "info", e.category, "campaign", f"Special campaign recorded for {e.category}."))
        if p["cold_chain"]:
            out.append(_alert(date_iso, "start", "critical", e.category, "cold_chain",
                              f"Cold-chain failure recorded for {e.category}.", "Inspect refrigerated stock and apply write-offs per policy."))
        if p["delay_hrs"] >= 4:
            out.append(_alert(date_iso, "start", "warning", e.category, "delivery_delay",
                              f"Delivery delay of {p['delay_hrs']:.1f} h recorded for {e.category}.", "Expect limited availability; consider expediting."))
    for d in decisions:
        if d.get("status") == "approved" and decision_effective(d) == date_iso:
            out.append(_alert(date_iso, "start", "info", str(d.get("category")), f"decision_{d.get('id')}",
                              f"Approved decision took effect: {d.get('template')} ({d.get('change_summary', '')}).", ""))
    return out


def _initial_opening(m: dict[str, Any]) -> float:
    return INITIAL_STOCK_DAYS_COVER * float(m["demand_units"]) + float(m["safety_stock_units"])


def finalize_category_day(entry: PlanEntry, model: core.DemandModel, levers: dict[str, Any], decisions: list[dict[str, Any]],
                          opening_stock: float | None, baseline_row: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d, cat = entry.date, entry.category
    row: dict[str, Any] = {"date": d, "category": cat}
    alerts: list[dict[str, Any]] = []
    if entry.profile is None:
        row.update({"source": SOURCE_MISSING, "confidence": entry.confidence, "status": "No data",
                    "opening_stock": opening_stock, "closing_stock": opening_stock, "receipts": 0.0})
        alerts.append(_alert(d, "end", "warning", cat, "missing_data", f"{cat}: no outcome recorded (insufficient evidence).", entry.note))
        return row, alerts
    base_p = entry.profile
    adj_p, applied, skipped = adjust_profile(base_p, decisions, d)
    m_adj = core.analyze(adj_p, model, levers, keep_log=False)
    m_base = core.analyze(base_p, model, levers, keep_log=False) if applied else m_adj

    if entry.obs is not None:
        d_obs, s_obs = entry.obs["demand"], entry.obs["sales"]
        if applied:
            factor = m_adj["demand_units"] / max(m_base["demand_units"], 1e-9)
            fill_ratio = model.fill_rate(adj_p["delay_hrs"]) / max(model.fill_rate(base_p["delay_hrs"]), 1e-6)
            demand = d_obs * factor
            sales = demand * float(np.clip((s_obs / max(d_obs, 1e-9)) * fill_ratio, 0.0, 1.0))
            source = SOURCE_ADJUSTED
            revenue = sales * adj_p["price"]
        else:
            demand, sales, source = d_obs, s_obs, SOURCE_OBSERVED
            revenue = entry.obs["sales_value"] if entry.obs["sales_value"] is not None else sales * adj_p["price"]
        utility = entry.obs["utility"] if entry.obs["utility"] is not None else adj_p.get("utility_cost")
    else:
        demand, sales, source = float(m_adj["demand_units"]), float(m_adj["sales_units"]), SOURCE_ESTIMATED
        revenue = sales * adj_p["price"]
        utility = adj_p.get("utility_cost")
    lost = max(0.0, demand - sales)
    unit_cost = float(adj_p["unit_cost"])
    w_units = waste_units(adj_p, sales, demand)
    cogs, waste_cost = sales * unit_cost, w_units * unit_cost
    gross = revenue - cogs
    utility = float(utility) if utility is not None else 0.0

    safety = float(m_adj["safety_stock_units"])
    opening = float(opening_stock) if opening_stock is not None else _initial_opening(m_adj)
    if lost > 1e-9:       # shelf ran dry: receipts only covered what was sold and wasted
        receipts = max(0.0, sales + w_units - opening)
    else:
        receipts = max(0.0, max(float(m_adj["demand_units"]) + safety, sales + w_units) - opening)
    closing = opening + receipts - sales - w_units

    row.update({
        "source": source, "confidence": entry.confidence, "status": m_adj["status"],
        "event": adj_p["event"], "promo_active": int(adj_p["promo_active"]), "price": float(adj_p["price"]),
        "traffic": float(adj_p["traffic"]), "temperature": float(adj_p["temperature"]), "delay_hrs": float(adj_p["delay_hrs"]),
        "staff_capacity_pct": float(adj_p["staff_capacity_pct"]), "queue_len": float(adj_p["queue_len"]),
        "demand": float(demand), "sales": float(sales), "lost": float(lost), "revenue": float(revenue),
        "cogs": float(cogs), "gross_profit": float(gross), "waste_units": float(w_units), "waste_cost": float(waste_cost),
        "utility_cost": utility, "total_cost": float(cogs + waste_cost + utility),
        "net_profit": float(gross - waste_cost), "operating_profit": float(gross - waste_cost - utility),
        "opening_stock": opening, "receipts": float(receipts), "closing_stock": float(closing), "safety_stock": safety,
        "forecast_demand": float(m_adj["demand_units"]), "forecast_low": float(m_adj["demand_low"]), "forecast_high": float(m_adj["demand_high"]),
        "baseline_demand": float(baseline_row["demand"]) if baseline_row and baseline_row.get("demand") is not None else None,
        "staffing_pressure": float(m_adj["staffing_pressure"]), "compliance_flags": int(m_adj["compliance_flags"]),
        "stockout_pct_expected": float(m_adj["stockout_risk_pct"]), "decisions_applied": applied, "decisions_skipped": skipped,
    })
    # ---- end-of-day alerts and recommendations ------------------------------------------------------------
    rep = m_adj["agent_reports"]
    if m_adj["status"] == "Critical":
        why = [rep[k]["detail"] for k in ("compliance_tasks", "inventory_risk", "freshness_waste") if rep[k]["status"] == "alert"]
        alerts.append(_alert(d, "end", "critical", cat, "critical_status", f"{cat} closed the day in Critical status.",
                             " ".join(why) or "Review the agent reports."))
    if demand > 0 and lost / demand >= 0.08:
        alerts.append(_alert(d, "end", "warning", cat, "stockout", f"{cat}: {lost:,.0f} units ({100*lost/demand:.0f}% of demand) went unfulfilled.",
                             f"Delivery delay {adj_p['delay_hrs']:.1f} h. Recommended safety buffer {safety:,.0f} units; review supplier lead time."))
    if adj_p["cold_chain"]:
        alerts.append(_alert(d, "end", "critical", cat, "waste_cold_chain", f"{cat}: cold-chain write-off of EUR {waste_cost:,.2f} booked.",
                             "Audit refrigeration and supplier handover; consider reducing stock cover."))
    if m_adj["staffing_pressure"] >= 0.5:
        alerts.append(_alert(d, "end", "warning", cat, "staffing", f"{cat}: staffing pressure {m_adj['staffing_pressure']:.2f}.",
                             f"Traffic {adj_p['traffic']:,.0f} with {adj_p['staff_capacity_pct']:.0f}% staff capacity; add cover at peak hours."))
    if entry.obs is not None and not applied and m_adj["demand_high"] > 0:
        if demand > m_adj["demand_high"]:
            alerts.append(_alert(d, "end", "info", cat, "forecast_miss_high", f"{cat}: demand {demand:,.0f} exceeded the 80% forecast interval ({m_adj['demand_low']:,.0f}-{m_adj['demand_high']:,.0f}).",
                                 "Check unmodelled drivers before changing the plan."))
        elif demand < m_adj["demand_low"]:
            alerts.append(_alert(d, "end", "info", cat, "forecast_miss_low", f"{cat}: demand {demand:,.0f} fell below the 80% forecast interval ({m_adj['demand_low']:,.0f}-{m_adj['demand_high']:,.0f}).",
                                 "Check for competitor action or availability problems."))
    return row, alerts


def finalize_day(date_iso: str, entries: list[PlanEntry], model, levers, decisions: list[dict[str, Any]],
                 opening: dict[str, float], baseline: pd.DataFrame | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, alerts = [], []
    base_lookup: dict[str, dict[str, Any]] = {}
    if baseline is not None and not baseline.empty:
        b = baseline[baseline["date"] == date_iso]
        base_lookup = {r["category"]: r for r in b.to_dict("records")}
    for e in entries:
        r, a = finalize_category_day(e, model, levers, decisions, opening.get(e.category), base_lookup.get(e.category))
        rows.append(r)
        alerts.extend(a)
    return rows, alerts


def year_end_alert(date_iso: str, ledger: pd.DataFrame, baseline: pd.DataFrame | None) -> dict[str, Any]:
    rev = float(ledger["revenue"].sum()) if not ledger.empty else 0.0
    msg = f"Year-end reached on {date_iso}: realized revenue EUR {rev:,.0f}"
    if baseline is not None and not baseline.empty:
        b = float(baseline["revenue"].sum())
        msg += f" versus the baseline forecast of EUR {b:,.0f} ({100*(rev/b-1):+.1f}%)." if b else "."
    return _alert(date_iso, "end", "info", "All", "year_end", msg, "Review the Business simulation page for the full year summary.")


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------
class SimulationStore:
    """SQLite store. The database row is the single source of truth for the clock, so refreshes, several tabs and
    restarts all see the same business. Day commits are atomic and idempotent."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS sim_runs (run_id TEXT PRIMARY KEY, created_at TEXT, data_fingerprint TEXT,
                archived INTEGER DEFAULT 0, config TEXT, state TEXT, version INTEGER, heartbeat REAL, events TEXT);
            CREATE TABLE IF NOT EXISTS sim_ledger (run_id TEXT, date TEXT, category TEXT, payload TEXT,
                PRIMARY KEY (run_id, date, category));
            CREATE TABLE IF NOT EXISTS sim_alerts (run_id TEXT, key TEXT, date TEXT, payload TEXT, PRIMARY KEY (run_id, key));
            CREATE TABLE IF NOT EXISTS sim_baseline (run_id TEXT PRIMARY KEY, checksum TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS sim_adjustments (run_id TEXT, id TEXT, payload TEXT, PRIMARY KEY (run_id, id));
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            yield conn
        finally:
            conn.close()

    # ---- runs --------------------------------------------------------------
    def create_run(self, state: ClockState, config: dict[str, Any], baseline: dict[str, Any], fingerprint: str, now: float) -> None:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            c.execute("INSERT INTO sim_runs VALUES (?,?,?,?,?,?,?,?,?)",
                      (state.run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), fingerprint, 0,
                       dumps(config), state.to_json(), state.version, now, "[]"))
            c.execute("INSERT INTO sim_baseline VALUES (?,?,?)", (state.run_id, baseline["checksum"], dumps(baseline)))
            c.execute("COMMIT")

    def latest_run(self, fingerprint: str) -> str | None:
        with self._conn() as c:
            r = c.execute("SELECT run_id FROM sim_runs WHERE archived=0 AND data_fingerprint=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                          (fingerprint,)).fetchone()
        return r[0] if r else None

    def archive_run(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE sim_runs SET archived=1 WHERE run_id=?", (run_id,))

    def load_state(self, run_id: str) -> ClockState | None:
        with self._conn() as c:
            r = c.execute("SELECT state FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
        return ClockState.from_json(r[0]) if r else None

    def heartbeat(self, run_id: str) -> float | None:
        with self._conn() as c:
            r = c.execute("SELECT heartbeat FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
        return float(r[0]) if r and r[0] is not None else None

    def touch(self, run_id: str, now: float) -> None:
        with self._conn() as c:
            c.execute("UPDATE sim_runs SET heartbeat=? WHERE run_id=?", (now, run_id))

    def mutate_state(self, run_id: str, fn: Callable[[ClockState], ClockState]) -> ClockState:
        """Atomic read-modify-write of the clock (used by start/pause/speed/auto-pause)."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT state FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
            cur = ClockState.from_json(r[0])
            new = fn(cur)
            if new is not cur and new.version != cur.version:
                c.execute("UPDATE sim_runs SET state=?, version=? WHERE run_id=?", (new.to_json(), new.version, run_id))
            c.execute("COMMIT")
        return new

    def commit_day(self, run_id: str, expected_version: int, new_state: ClockState, rows: list[dict[str, Any]],
                   alerts: list[dict[str, Any]]) -> bool:
        """Close a day atomically. Returns False (and writes nothing) if another session already moved the clock."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT version FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
            if not r or int(r[0]) != expected_version:
                c.execute("ROLLBACK")
                return False
            for row in rows:
                c.execute("INSERT OR IGNORE INTO sim_ledger VALUES (?,?,?,?)", (run_id, row["date"], row["category"], dumps(row)))
            for a in alerts:
                c.execute("INSERT OR IGNORE INTO sim_alerts VALUES (?,?,?,?)", (run_id, a["key"], a["date"], dumps(a)))
            c.execute("UPDATE sim_runs SET state=?, version=? WHERE run_id=?", (new_state.to_json(), new_state.version, run_id))
            c.execute("COMMIT")
        return True

    def mark_day_started(self, run_id: str, date_iso: str, alerts: list[dict[str, Any]]) -> None:
        """Record start-of-day processing once. Does not bump the clock version (no CAS conflicts with ticks)."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute("SELECT state FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
            cur = ClockState.from_json(r[0])
            if cur.day_started != date_iso and cur.current_date == date_iso:
                for a in alerts:
                    c.execute("INSERT OR IGNORE INTO sim_alerts VALUES (?,?,?,?)", (run_id, a["key"], a["date"], dumps(a)))
                c.execute("UPDATE sim_runs SET state=? WHERE run_id=?", (replace(cur, day_started=date_iso).to_json(), run_id))
            c.execute("COMMIT")

    # ---- reads -------------------------------------------------------------
    def config(self, run_id: str) -> dict[str, Any]:
        with self._conn() as c:
            r = c.execute("SELECT config FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r else {}

    def baseline(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT payload FROM sim_baseline WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def ledger(self, run_id: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM sim_ledger WHERE run_id=? ORDER BY date, category", (run_id,)).fetchall()
        return pd.DataFrame([json.loads(r[0]) for r in rows])

    def last_closing(self, run_id: str) -> dict[str, float]:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM sim_ledger WHERE run_id=? AND date=(SELECT MAX(date) FROM sim_ledger WHERE run_id=?)",
                             (run_id, run_id)).fetchall()
        out = {}
        for (p,) in rows:
            j = json.loads(p)
            if j.get("closing_stock") is not None:
                out[j["category"]] = float(j["closing_stock"])
        return out

    def alerts(self, run_id: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM sim_alerts WHERE run_id=? ORDER BY date DESC, rowid DESC", (run_id,)).fetchall()
        return pd.DataFrame([json.loads(r[0]) for r in rows])

    def add_adjustment(self, run_id: str, adj: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO sim_adjustments VALUES (?,?,?)", (run_id, str(adj["id"]), dumps(adj)))

    def adjustments(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM sim_adjustments WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_events(self, run_id: str, events: list[dict[str, Any]]) -> None:
        with self._conn() as c:
            c.execute("UPDATE sim_runs SET events=? WHERE run_id=?", (dumps(events), run_id))

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT events FROM sim_runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r and r[0] else []


# ---------------------------------------------------------------------------
# 8. Orchestration used by the Streamlit app (and by tests)
# ---------------------------------------------------------------------------
@dataclass
class SimContext:
    plan: list[PlanEntry]
    model: core.DemandModel
    levers: dict[str, Any]
    baseline: pd.DataFrame

    def __post_init__(self) -> None:
        self.by_date: dict[str, list[PlanEntry]] = {}
        for e in self.plan:
            self.by_date.setdefault(e.date, []).append(e)


def approved_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Approved decisions, oldest first (the app keeps events newest first)."""
    return [e for e in reversed(events) if e.get("status") == "approved"]


def begin_day(store: SimulationStore, run_id: str, ctx: SimContext, events: list[dict[str, Any]]) -> bool:
    s = store.load_state(run_id)
    if s is None or s.day_started == s.current_date:
        return False
    store.mark_day_started(run_id, s.current_date, begin_day_notices(s.current_date, ctx.by_date.get(s.current_date, []), approved_decisions(events)))
    return True


def process_due_days(store: SimulationStore, run_id: str, ctx: SimContext, events_provider: Callable[[], list[dict[str, Any]]],
                     now: float, max_days: int = MAX_CATCHUP_DAYS) -> int:
    """Finalize every day whose 30 s (or configured) interval has elapsed. Returns the number of days closed."""
    closed = 0
    for _ in range(max_days):
        s = store.load_state(run_id)
        if s is None or not day_due(s, now):
            break
        decisions = approved_decisions(events_provider())
        entries = ctx.by_date.get(s.current_date, [])
        rows, alerts = finalize_day(s.current_date, entries, ctx.model, ctx.levers, decisions, store.last_closing(run_id), ctx.baseline)
        new_state = advance(s, now)
        if new_state.status == FINISHED:
            led = pd.concat([store.ledger(run_id), pd.DataFrame(rows)], ignore_index=True)
            alerts.append(year_end_alert(s.current_date, led, ctx.baseline))
        if not store.commit_day(run_id, s.version, new_state, rows, alerts):
            break
        closed += 1
        if new_state.status == FINISHED:
            break
    return closed
