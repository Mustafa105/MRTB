"""Unit tests for simulation_engine (run: python -m pytest -q test_simulation_engine.py)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import retail_brain_core as core
import simulation_engine as sim

CSV = Path(__file__).with_name("retail_operations_data.csv")


@pytest.fixture(scope="module")
def data():
    df, _ = core.load_operations_csv(CSV)
    return df


@pytest.fixture(scope="module")
def model(data):
    return core.train_model(data)


@pytest.fixture(scope="module")
def levers(model):
    return core.resolve_levers(model.lever_est, core.DEFAULT_ASSUMED_ELASTICITY, core.DEFAULT_ASSUMED_PROMO_UPLIFT)


@pytest.fixture(scope="module")
def index(data):
    return sim.DayIndex(data)


@pytest.fixture(scope="module")
def plan(index, model):
    return sim.build_year_plan(index, model, "2026-01-01", "2026-12-31")


@pytest.fixture(scope="module")
def baseline(plan, model, levers):
    return sim.forecast_table(model, levers, plan)


@pytest.fixture()
def ctx(plan, model, levers, baseline):
    return sim.SimContext(plan, model, levers, baseline)


@pytest.fixture()
def store(tmp_path):
    return sim.SimulationStore(tmp_path / "sim.sqlite3")


def _make_run(store, ctx, spd=30.0, now=1000.0):
    clock = sim.new_clock("2026-01-01", spd)
    store.create_run(clock, {"x": 1}, sim.freeze_baseline(ctx.baseline, {}), "fp", now)
    return clock.run_id


# --------------------------------------------------------------------------- clock
def test_clock_runs_and_advances_exactly_per_interval():
    s = sim.new_clock("2026-01-01")
    assert s.status == sim.READY and s.current_date == "2026-01-01" and s.day_number == 1
    s = sim.start_or_resume(s, 100.0)
    assert not sim.day_due(s, 129.9) and sim.day_due(s, 130.0)
    s2 = sim.advance(s, 130.0)
    assert (s2.current_date, s2.day_number) == ("2026-01-02", 2)
    assert sim.day_elapsed(s2, 130.0) == pytest.approx(0.0)
    assert sim.day_progress(s2, 145.0) == pytest.approx(0.5)


def test_pause_freezes_progress_and_resume_continues():
    s = sim.start_or_resume(sim.new_clock("2026-01-01"), 0.0)
    s = sim.pause(s, 12.0)
    assert s.status == sim.PAUSED and sim.day_elapsed(s, 999.0) == pytest.approx(12.0)
    s = sim.start_or_resume(s, 1000.0)
    assert sim.seconds_to_next_day(s, 1010.0) == pytest.approx(8.0)


def test_late_tick_keeps_leftover_time_and_never_skips_a_day():
    s = sim.start_or_resume(sim.new_clock("2026-01-01"), 0.0)
    now = 75.0   # tab was throttled: 2.5 days elapsed
    dates = []
    while sim.day_due(s, now):
        dates.append(s.current_date)
        s = sim.advance(s, now)
    assert dates == ["2026-01-01", "2026-01-02"] and s.current_date == "2026-01-03"
    assert sim.day_elapsed(s, now) == pytest.approx(15.0)


def test_year_end_state():
    s = sim.new_clock("2026-12-30", 5.0)
    s = sim.start_or_resume(s, 0.0)
    s = sim.advance(s, 5.0)
    assert s.current_date == "2026-12-31" and s.status == sim.RUNNING and s.day_number == 2
    s = sim.advance(s, 10.0)
    assert s.status == sim.FINISHED and s.current_date == "2026-12-31"
    assert sim.start_or_resume(s, 11.0).status == sim.FINISHED and not sim.day_due(s, 1e9)
    assert sim.new_clock("2026-01-01").total_days == 365


def test_speed_change_preserves_fraction_of_day():
    s = sim.start_or_resume(sim.new_clock("2026-01-01", 30.0), 0.0)
    s = sim.set_speed(s, 5.0, 15.0)   # halfway through the day
    assert sim.day_progress(s, 15.0) == pytest.approx(0.5)
    assert sim.seconds_to_next_day(s, 15.0) == pytest.approx(2.5)


def test_stale_running_clock_is_auto_paused_without_skipping_days():
    s = sim.start_or_resume(sim.new_clock("2026-01-01"), 0.0)
    s = sim.auto_pause_if_stale(s, heartbeat_ts=10.0, now=10_000.0)
    assert s.status == sim.PAUSED and s.current_date == "2026-01-01"
    assert sim.day_elapsed(s, 10_000.0) == pytest.approx(10.0) and "paused automatically" in s.notice
    live = sim.start_or_resume(sim.new_clock("2026-01-01"), 0.0)
    assert sim.auto_pause_if_stale(live, heartbeat_ts=9.0, now=10.0).status == sim.RUNNING


# ------------------------------------------------------------------- CSV integration
def test_csv_covers_whole_year_with_observed_rows(index):
    cov = sim.data_coverage(index, "2026-01-01", "2026-12-31")
    assert cov["sim_days"] == 365 and cov["dates_full"] == 365 and cov["dates_requiring_estimation"] == 0
    assert cov["pct_dates_with_observations"] == pytest.approx(100.0)
    assert (cov["csv_earliest"], cov["csv_latest"]) == ("2026-01-01", "2026-12-31")


def test_observed_row_is_used_unchanged(index, data):
    r = index.resolve("2026-03-15", "Birra")
    raw = data[(data["Date"] == "2026-03-15") & (data["Category"] == "Birra")].iloc[0]
    assert r.source == sim.SOURCE_OBSERVED and r.row["Demand_QTY"] == raw["Demand_QTY"]


def test_case_b_estimate_and_case_c_missing_never_fabricate_outcomes(data):
    # remove one week for all categories (Case B) and make Detersivi nearly empty around Jan (Case C)
    cut = data[~data["Date"].between("2026-06-10", "2026-06-16")]
    cut = cut[~((cut["Category"] == "Detersivi") & ~cut["Date"].between("2026-06-20", "2026-09-30"))]
    idx = sim.DayIndex(cut)
    est = idx.resolve("2026-06-12", "Birra")
    assert est.source == sim.SOURCE_ESTIMATED and est.confidence == "Medium"
    assert np.isnan(est.row["Demand_QTY"]) and np.isnan(est.row["Actual_Sales_QTY"])
    assert est.row["Event_Season"] == "None" and est.row["Cold_Chain_Failure_Flag"] == "No"
    assert pd.Timestamp(est.row["Date"]) == pd.Timestamp("2026-06-12")
    missing = idx.resolve("2026-01-10", "Detersivi")
    assert missing.source == sim.SOURCE_MISSING and missing.row is None
    cov = sim.data_coverage(idx, "2026-01-01", "2026-12-31")
    assert cov["dates_none"] == 7 and cov["category_days_missing"] > 0 and cov["category_days_estimated"] >= 7
    assert cov["dates_requiring_estimation"] >= 7 and cov["pct_dates_with_observations"] < 100


# ------------------------------------------------- vectorised forecast == nine agents
def test_vectorised_forecast_matches_agent_pipeline(plan, model, levers):
    sample = [e for e in plan if e.date in ("2026-01-01", "2026-08-15", "2026-11-27", "2026-12-24")]
    cold = [e for e in plan if e.profile and e.profile["cold_chain"]][:2]
    promo = [e for e in plan if e.profile and e.profile["promo_active"]][:2]
    entries = sample + cold + promo
    tab = sim.forecast_table(model, levers, entries)
    for (_, row), e in zip(tab.iterrows(), entries):
        m = core.analyze(e.profile, model, levers, keep_log=False)
        assert row["demand"] == pytest.approx(m["demand_units"], rel=1e-9)
        assert row["sales"] == pytest.approx(m["sales_units"], rel=1e-9)
        assert row["revenue"] == pytest.approx(m["revenue_eur"], rel=1e-9)
        assert row["gross_profit"] == pytest.approx(m["gross_profit_eur"], rel=1e-9)
        assert row["waste_cost"] == pytest.approx(m["waste_cost_eur"], rel=1e-9)
        assert row["safety_stock"] == pytest.approx(m["safety_stock_units"], rel=1e-9)
        assert row["staffing_pressure"] == pytest.approx(m["staffing_pressure"], rel=1e-9)
        assert row["stockout_pct"] == pytest.approx(m["stockout_risk_pct"], rel=1e-9)


# ---------------------------------------------------------------- baseline and adjustments
def test_baseline_is_frozen_and_checksummed(baseline):
    payload = sim.freeze_baseline(baseline, {"note": "x"})
    assert sim.baseline_is_intact(payload)
    tampered = dict(payload, checksum="0" * 16)
    assert not sim.baseline_is_intact(tampered)
    assert len(sim.baseline_table(payload)) == 365 * 4
    s = sim.summarize(baseline)["total"]
    assert s["days"] == 365 and s["demand"] > 0 and s["category_days_estimated"] == 0


def _decision(cat="Birra", eff="2026-03-01", changes=None, status="approved", tpl="Promotion opportunity"):
    return {"id": f"d-{cat}-{eff}", "category": cat, "status": status, "effective_date": eff, "template": tpl,
            "changes": changes or {"promo_active_set": 1, "traffic_pct": 5}, "change_summary": "test"}


def test_operational_forecast_only_changes_from_effective_date_and_baseline_stays_fixed(plan, model, levers, baseline):
    before_sum = sim.table_checksum(baseline)
    dec = _decision("Birra", "2026-03-01", {"price_pct": -10})
    op = sim.operational_forecast(model, levers, plan, [dec], "2026-02-01")
    base = baseline[baseline["date"] >= "2026-02-01"].reset_index(drop=True)
    same = (op["date"] < "2026-03-01") | (op["category"] != "Birra")
    assert np.allclose(op.loc[same, "demand"], base.loc[same, "demand"])
    changed = ~same
    assert (op.loc[changed, "demand"] > base.loc[changed, "demand"]).all()     # cheaper -> more demand (elasticity < 0)
    assert sim.table_checksum(baseline) == before_sum                          # baseline untouched


def test_pending_rejected_or_future_decisions_do_not_move_forecast(plan, model, levers, baseline):
    decs = [_decision(status="pending"), _decision(status="rejected", changes={"price_pct": -30}),
            _decision(eff="2027-01-01", changes={"price_pct": -30})]
    op = sim.operational_forecast(model, levers, plan, decs, "2026-01-01")
    assert np.allclose(op["demand"], baseline["demand"])


def test_event_uplift_is_not_counted_twice_when_csv_already_records_event(plan, model, levers):
    dec = _decision("Birra", "2026-01-01", {"event_set": "Black Friday Campaign", "traffic_pct": 20}, tpl="Seasonal event peak")
    op = sim.operational_forecast(model, levers, plan, [dec], "2026-11-20")
    base = sim.operational_forecast(model, levers, plan, [], "2026-11-20")
    bf = op[(op["category"] == "Birra")]
    bb = base[(base["category"] == "Birra")]
    csv_event_days = bf["date"].isin(["2026-11-27"] + [f"2026-12-{d}" for d in range(20, 26)]).to_numpy()
    assert np.allclose(bf.loc[csv_event_days, "demand"].to_numpy(), bb.loc[csv_event_days, "demand"].to_numpy())   # skipped: CSV already has the event
    assert (bf.loc[~csv_event_days, "demand"].to_numpy() > bb.loc[~csv_event_days, "demand"].to_numpy()).all()    # applied on ordinary days
    adj = sim.evaluate_adjustment(model, levers, plan, [], dec, "2026-11-20", "2026-11-20")
    assert "already recorded" in adj["reason"]


def test_materiality_threshold_gates_forecast_adjustment(plan, model, levers):
    tiny = _decision("Birra", "2026-12-30", {"price_pct": -1})
    big = _decision("Birra", "2026-02-01", {"price_pct": -15})
    a_tiny = sim.evaluate_adjustment(model, levers, plan, [], tiny, "2026-01-20", "2026-01-20")
    a_big = sim.evaluate_adjustment(model, levers, plan, [], big, "2026-01-20", "2026-01-20")
    assert not a_tiny["applied"] and "materiality" in a_tiny["reason"]
    assert a_big["applied"] and a_big["delta_units"] > 0 and a_big["affected_categories"] == ["Birra"]
    for k in ("reason", "effective_date", "method", "pct_units", "created_sim_date"):
        assert a_big[k] not in (None, "")
    active = sim.active_adjustments([a_tiny, a_big], [tiny, big])
    assert [d["id"] for d in active] == [big["id"]]


# ------------------------------------------------------------------- day lifecycle + store
def test_full_year_run_uses_real_csv_and_is_consistent(ctx, store, data):
    rid = _make_run(store, ctx, spd=1.0, now=0.0)
    store.mutate_state(rid, lambda s: sim.start_or_resume(s, 0.0))
    sim.begin_day(store, rid, ctx, [])
    closed = 0
    now = 0.0
    while True:
        now += 1.0
        n = sim.process_due_days(store, rid, ctx, lambda: [], now, max_days=400)
        closed += n
        sim.begin_day(store, rid, ctx, [])        # what every Streamlit rerun does
        s = store.load_state(rid)
        if s.status == sim.FINISHED:
            break
    assert closed == 365 and s.current_date == "2026-12-31"
    led = store.ledger(rid)
    assert len(led) == 365 * 4 and not led.duplicated(["date", "category"]).any()
    # realized YTD equals the CSV exactly (no decisions): no double counting, no invented sales
    assert led["sales"].sum() == pytest.approx(data["Actual_Sales_QTY"].sum())
    assert led["demand"].sum() == pytest.approx(data["Demand_QTY"].sum())
    assert led["revenue"].sum() == pytest.approx(data["Actual_Sales_Value_EUR"].sum())
    assert (led["source"] == sim.SOURCE_OBSERVED).all()
    # inventory continuity: closing(day n) == opening(day n+1); never negative; balance holds
    led = led.sort_values(["category", "date"])
    for _, g in led.groupby("category"):
        assert np.allclose(g["closing_stock"].to_numpy()[:-1], g["opening_stock"].to_numpy()[1:])
        assert (g["closing_stock"] >= -1e-6).all()
        assert np.allclose(g["opening_stock"] + g["receipts"] - g["sales"] - g["waste_units"], g["closing_stock"])
    alerts = store.alerts(rid)
    assert (alerts["kind"] == "year_end").sum() == 1 and "csv_event" in set(alerts["kind"])
    assert set(alerts.loc[alerts["kind"] == "csv_event", "date"]) == set(data.loc[data["Event_Season"] != "None", "Date"].dt.strftime("%Y-%m-%d"))


def test_commit_is_idempotent_and_cas_protects_against_double_advance(ctx, store):
    rid = _make_run(store, ctx, spd=30.0, now=0.0)
    store.mutate_state(rid, lambda s: sim.start_or_resume(s, 0.0))
    s0 = store.load_state(rid)
    rows, alerts = sim.finalize_day("2026-01-01", ctx.by_date["2026-01-01"], ctx.model, ctx.levers, [], {}, ctx.baseline)
    ns = sim.advance(s0, 30.0)
    assert store.commit_day(rid, s0.version, ns, rows, alerts) is True
    assert store.commit_day(rid, s0.version, ns, rows, alerts) is False       # second tab loses the race
    assert len(store.ledger(rid)) == 4 and store.load_state(rid).current_date == "2026-01-02"


def test_refresh_resumes_same_run_and_history(ctx, store):
    rid = _make_run(store, ctx, spd=5.0, now=0.0)
    store.mutate_state(rid, lambda s: sim.start_or_resume(s, 0.0))
    sim.process_due_days(store, rid, ctx, lambda: [], 17.0)                   # 3 days closed, 2 s into day 4
    store2 = sim.SimulationStore(store.path)                                  # "browser refresh": fresh objects
    assert store2.latest_run("fp") == rid
    s = store2.load_state(rid)
    assert s.current_date == "2026-01-04" and len(store2.ledger(rid)) == 12
    assert sim.day_elapsed(s, 17.0) == pytest.approx(2.0)
    assert sim.baseline_is_intact(store2.baseline(rid))


def test_decision_changes_outcomes_only_after_effective_date(ctx, store, data):
    rid = _make_run(store, ctx, spd=1.0, now=0.0)
    dec = _decision("Birra", "2026-01-04", {"price_pct": -20})
    store.mutate_state(rid, lambda s: sim.start_or_resume(s, 0.0))
    sim.process_due_days(store, rid, ctx, lambda: [dec], 6.0)                 # closes Jan 1..Jan 6
    led = store.ledger(rid)
    b = led[led["category"] == "Birra"].set_index("date")
    raw = data[data["Category"] == "Birra"].copy()
    raw.index = raw["Date"].dt.strftime("%Y-%m-%d")
    for d in ("2026-01-01", "2026-01-02", "2026-01-03"):
        assert b.loc[d, "source"] == sim.SOURCE_OBSERVED and b.loc[d, "sales"] == raw.loc[d, "Actual_Sales_QTY"]
    for d in ("2026-01-04", "2026-01-05"):
        assert b.loc[d, "source"] == sim.SOURCE_ADJUSTED and b.loc[d, "demand"] > raw.loc[d, "Demand_QTY"]
        assert b.loc[d, "price"] == pytest.approx(raw.loc[d, "Esselunga_Price_EUR"] * 0.8)
    other = led[(led["category"] == "Ortofrutta") & (led["date"] >= "2026-01-04")]
    assert (other["source"] == sim.SOURCE_OBSERVED).all()
    sim.begin_day(store, rid, ctx, [dec])        # start-of-day notices for Jan 7 are idempotent
    sim.begin_day(store, rid, ctx, [dec])
    assert (store.alerts(rid)["date"] == "2026-01-07").sum() == (store.alerts(rid).drop_duplicates("key")["date"] == "2026-01-07").sum()


def test_missing_category_day_is_recorded_without_fabricated_values(data, model, levers):
    cut = data[~((data["Category"] == "Detersivi") & ~data["Date"].between("2026-06-20", "2026-09-30"))]
    idx = sim.DayIndex(cut)
    plan = sim.build_year_plan(idx, model, "2026-01-01", "2026-01-03")
    entries = [e for e in plan if e.date == "2026-01-02"]
    rows, alerts = sim.finalize_day("2026-01-02", entries, model, levers, [], {}, None)
    det = next(r for r in rows if r["category"] == "Detersivi")
    assert det["source"] == sim.SOURCE_MISSING and "sales" not in det and "revenue" not in det
    assert any(a["kind"] == "missing_data" for a in alerts)
