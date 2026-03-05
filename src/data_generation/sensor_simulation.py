"""
Phase 1 — Railway Sensor Data Generator (final)

Key design decisions:
  - Failure events scale with --steps so any dataset size works
  - ~35% of steps have at least one component degrading
  - risk_score derived from degradation STATE + Gaussian noise
    (NOT from sensor threshold formula — prevents data leakage)
  - Multiple simultaneous failures to simulate realistic cascades
  - Lower risk thresholds (low>=0.15, med>=0.35, high>=0.65) for
    a more balanced label distribution

Usage:
  python -m src.data_generation.sensor_simulator                 # 20000 steps
  python -m src.data_generation.sensor_simulator --steps 20000
"""

import csv, math, random, argparse
from pathlib import Path

random.seed(42)

COMPONENT_ORDER = [
    "T01","T02","T03","T04","T05","T06","T07","T08","T09","T10",
    "SW1","SW2","SW3","SW4","SG1","SG2","SG3","BR1","BR2","BR3",
]
COMPONENT_TYPES = {
    **{f"T0{i}":"track" for i in range(1,10)}, "T10":"track",
    **{f"SW{i}":"switch" for i in range(1,5)},
    **{f"SG{i}":"signal" for i in range(1,4)},
    **{f"BR{i}":"bridge" for i in range(1,4)},
}
BASELINES = {
    "track":  {"vibration":(0.35,0.04),"temperature":(42.0,2.0),"load":(65.0,4.0),"current":(14.5,0.6)},
    "switch": {"vibration":(0.55,0.06),"temperature":(48.0,2.5),"load":(45.0,3.0),"current":(18.5,0.8)},
    "signal": {"vibration":(0.15,0.02),"temperature":(35.0,1.5),"load":(20.0,2.0),"current":(8.5,0.4)},
    "bridge": {"vibration":(0.25,0.03),"temperature":(55.0,3.0),"load":(85.0,5.0),"current":(22.0,1.0)},
}
SENSORS = ["vibration","temperature","load","current"]
FIELDNAMES = [
    "record_id","time_step","component_id","component_type",
    "vibration","temperature","load","current",
    "vibration_delta","temperature_delta",
    "risk_score","health_index","risk_level",
    "failure_mode","is_anomaly","is_degrading","degradation_state",
]

# Risk thresholds — lower than typical so propagation nodes get labelled too
THRESH_HIGH   = 0.65
THRESH_MEDIUM = 0.35
THRESH_LOW    = 0.15


def _make_events(S: int) -> list:
    """
    Build failure event schedule scaled to S total steps.
    Covers ~40% of all steps with some form of degradation.
    s() scales a base-5000 step value to actual S.
    """
    def s(v): return int(v * S / 5000)
    return [
        # ── Cluster 1 (early) ──────────────────────────────────
        {"start":s(150),  "end":s(550),  "origin":"T05","mode":"mechanical_wear",
         "peak":0.88,"props":{"SW3":s(80),"T04":s(100),"T06":s(100),"BR2":s(120)}},
        {"start":s(250),  "end":s(500),  "origin":"SW1","mode":"electrical_fault",
         "peak":0.76,"props":{"T01":s(60),"T02":s(60),"SG1":s(100)}},
        # ── Cluster 2 ─────────────────────────────────────────
        {"start":s(700),  "end":s(1100), "origin":"BR2","mode":"overheating",
         "peak":0.91,"props":{"T05":s(100),"T06":s(100),"SW3":s(150)}},
        {"start":s(800),  "end":s(1050), "origin":"T08","mode":"mechanical_wear",
         "peak":0.73,"props":{"T07":s(60),"T09":s(60),"SW4":s(80),"BR3":s(100)}},
        {"start":s(900),  "end":s(1100), "origin":"SG2","mode":"electrical_fault",
         "peak":0.71,"props":{"SW2":s(80),"SW3":s(80)}},
        # ── Cluster 3 (heavy cascade) ──────────────────────────
        {"start":s(1400), "end":s(1900), "origin":"T05","mode":"mechanical_wear",
         "peak":0.96,"props":{"SW2":s(100),"SW3":s(100),"BR2":s(150),"T04":s(120),"T06":s(120)}},
        {"start":s(1500), "end":s(1850), "origin":"BR1","mode":"overheating",
         "peak":0.83,"props":{"T02":s(80),"T03":s(80),"SW1":s(120)}},
        # ── Cluster 4 ─────────────────────────────────────────
        {"start":s(2100), "end":s(2450), "origin":"SW4","mode":"electrical_fault",
         "peak":0.79,"props":{"T07":s(80),"T08":s(80),"T09":s(80),"SG3":s(120)}},
        {"start":s(2300), "end":s(2650), "origin":"T03","mode":"mechanical_wear",
         "peak":0.69,"props":{"T02":s(80),"T04":s(80),"SW1":s(100),"SW2":s(100)}},
        # ── Cluster 5 ─────────────────────────────────────────
        {"start":s(2900), "end":s(3350), "origin":"BR3","mode":"overheating",
         "peak":0.86,"props":{"T08":s(100),"T09":s(100),"SW4":s(120)}},
        {"start":s(3050), "end":s(3450), "origin":"T01","mode":"mechanical_wear",
         "peak":0.74,"props":{"T02":s(60),"SW1":s(80)}},
        {"start":s(3150), "end":s(3550), "origin":"SG3","mode":"electrical_fault",
         "peak":0.77,"props":{"SW3":s(80),"SW4":s(80)}},
        # ── Cluster 6 (major finale) ───────────────────────────
        {"start":s(3900), "end":s(4550), "origin":"T05","mode":"mechanical_wear",
         "peak":0.97,"props":{"SW2":s(80),"SW3":s(80),"BR2":s(100),
                              "T04":s(100),"T06":s(100),"SG2":s(150)}},
        {"start":s(4000), "end":s(4450), "origin":"BR2","mode":"overheating",
         "peak":0.94,"props":{"T05":s(60),"T06":s(60),"SW3":s(80)}},
        {"start":s(4100), "end":s(4650), "origin":"SW1","mode":"electrical_fault",
         "peak":0.81,"props":{"T01":s(80),"T02":s(80),"T03":s(100),"SG1":s(120)}},
        # ── Late events ───────────────────────────────────────
        {"start":s(4550), "end":s(4850), "origin":"T10","mode":"mechanical_wear",
         "peak":0.72,"props":{"T09":s(80),"SW4":s(100)}},
        {"start":s(4650), "end":s(4950), "origin":"SG1","mode":"electrical_fault",
         "peak":0.75,"props":{"SW1":s(80),"SW2":s(80)}},
    ]


def _get_severity(cid: str, step: int, events: list) -> tuple:
    best_sev, best_mode = 0.0, "normal"
    for ev in events:
        if step < ev["start"] or step > ev["end"]:
            continue
        dur     = ev["end"] - ev["start"]
        if dur == 0: continue
        elapsed = step - ev["start"]
        if elapsed < dur * 0.3:
            sev = ev["peak"] * (1 - math.exp(-4 * elapsed / max(1, dur * 0.3)))
        elif elapsed > dur * 0.8:
            sev = ev["peak"] * max(0.0, 1 - (elapsed - dur*0.8) / max(1, dur*0.2))
        else:
            sev = ev["peak"]

        if cid == ev["origin"]:
            if sev > best_sev:
                best_sev  = sev
                best_mode = ev["mode"]
        elif cid in ev.get("props", {}):
            delay      = ev["props"][cid]
            prop_start = ev["start"] + delay
            if step >= prop_start:
                pe   = step - prop_start
                prop = sev * 0.55 * (1 - math.exp(-3 * pe / max(1, dur - delay)))
                if prop > best_sev:
                    best_sev  = prop
                    best_mode = ev["mode"] + "_propagated"
    return best_sev, best_mode


def _generate_row(cid: str, step: int, rid: int,
                  prev: dict, events: list) -> dict:
    ctype    = COMPONENT_TYPES[cid]
    baseline = BASELINES[ctype]
    sev, mode = _get_severity(cid, step, events)

    readings, deltas = {}, {}
    for s in SENSORS:
        bm, bs = baseline[s]
        if "mechanical_wear" in mode:
            mult = {"vibration":1+sev*3.2,"temperature":1+sev*0.9,
                    "load":1+sev*0.4,"current":1+sev*0.6}[s]
        elif "overheating" in mode:
            mult = {"vibration":1+sev*0.5,"temperature":1+sev*3.8,
                    "load":1+sev*0.3,"current":1+sev*2.0}[s]
        elif "electrical_fault" in mode:
            mult = {"vibration":1+sev*0.4,"temperature":1+sev*1.0,
                    "load":max(0.4, 1-sev*0.4),"current":1+sev*3.5}[s]
        else:
            mult = 1.0

        drift = 0.02 * math.sin(step*0.01 + hash(cid+s) % 100)
        noise = random.gauss(0, bs * (1 + sev * 0.6))
        val   = max(0.0, bm * mult + noise + drift * bm)
        readings[s] = round(val, 4)
        deltas[s]   = round(val - prev.get(s, bm), 4)

    # Risk score from degradation STATE + noise — not from sensor formula
    risk_score   = round(min(1.0, max(0.0, sev + random.gauss(0, 0.04))), 4)
    health_index = round(max(0.0, 1.0 - sev + random.gauss(0, 0.03)), 4)

    if   risk_score >= THRESH_HIGH:   rl = "high"
    elif risk_score >= THRESH_MEDIUM: rl = "medium"
    elif risk_score >= THRESH_LOW:    rl = "low"
    else:                             rl = "normal"

    if   sev < 0.10: ds = "healthy"
    elif sev < 0.30: ds = "early_wear"
    elif sev < 0.60: ds = "moderate_wear"
    elif sev < 0.80: ds = "severe_wear"
    else:            ds = "critical"

    return {
        "record_id":         rid,
        "time_step":         step,
        "component_id":      cid,
        "component_type":    ctype,
        "vibration":         readings["vibration"],
        "temperature":       readings["temperature"],
        "load":              readings["load"],
        "current":           readings["current"],
        "vibration_delta":   deltas["vibration"],
        "temperature_delta": deltas["temperature"],
        "risk_score":        risk_score,
        "health_index":      health_index,
        "risk_level":        rl,
        "failure_mode":      mode,
        "is_anomaly":        risk_score >= THRESH_MEDIUM,
        "is_degrading":      risk_score >= THRESH_LOW,
        "degradation_state": ds,
    }


def generate(n_steps: int = 20000,
             out_path: str = "data/raw/railway_sensor_data.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    events = _make_events(n_steps)
    rid    = 0
    prev   = {c: {"vibration":0.35,"temperature":42.0,
                  "load":65.0,"current":14.5}
              for c in COMPONENT_ORDER}
    counts = {"normal":0,"low":0,"medium":0,"high":0,"medium":0}
    counts = {k:0 for k in ["normal","low","medium","high"]}

    n_comps = len(COMPONENT_ORDER)
    total   = n_steps * n_comps

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — SENSOR DATA GENERATION")
    print(f"{'='*60}")
    print(f"  Components    : {n_comps}")
    print(f"  Time steps    : {n_steps:,}")
    print(f"  Total records : {total:,}")
    print(f"  Failure events: {len(events)}")
    print(f"  Output        : {out_path}")
    print(f"{'='*60}\n")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for step in range(n_steps):
            for cid in COMPONENT_ORDER:
                row = _generate_row(cid, step, rid, prev[cid], events)
                writer.writerow(row)
                prev[cid] = {s: row[s] for s in SENSORS}
                counts[row["risk_level"]] += 1
                rid += 1
            if step % 2000 == 0 and step > 0:
                done = rid / total * 100
                print(f"  step {step:>6,}/{n_steps}  records={rid:,}  ({done:.0f}%)")

    print(f"\n  ✅ Done — {rid:,} records written to {out_path}")
    print(f"\n  Label distribution:")
    for k, v in counts.items():
        bar = "█" * int(v / rid * 50)
        print(f"    {k:8s}: {v:8,}  ({v/rid*100:5.1f}%)  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Sensor Data Generator")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Number of time steps (default: 20000 → 400K records)")
    parser.add_argument("--out",   default="data/raw/railway_sensor_data.csv")
    args = parser.parse_args()
    generate(args.steps, args.out)


if __name__ == "__main__":
    main()