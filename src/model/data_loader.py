"""
Phase 4 — Data Loader v2

KEY FIX: Forecasting formulation instead of current-state prediction.

Input  window: sensor readings from steps [i .. i+T-1]   (past 12 steps)
Label  window: risk_level/score at step    [i+T+horizon]  (N steps in future)

This makes the task genuinely hard — the model must learn patterns
that PRECEDE failures, not just read current thresholds.

horizon=6 means: predict what will happen 6 steps from now.
"""

import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

COMPONENT_ORDER = [
    "T01","T02","T03","T04","T05","T06","T07","T08","T09","T10",
    "SW1","SW2","SW3","SW4",
    "SG1","SG2","SG3",
    "BR1","BR2","BR3",
]
N_NODES    = len(COMPONENT_ORDER)
COMP_INDEX = {c: i for i, c in enumerate(COMPONENT_ORDER)}
SENSORS    = ["vibration", "temperature", "load", "current"]

# Input features: raw sensors + rolling stats — NO risk_score, NO risk_level
FEATURE_COLS = (
    SENSORS +
    [f"{s}_mean"  for s in SENSORS] +
    [f"{s}_std"   for s in SENSORS] +
    [f"{s}_trend" for s in SENSORS]
)
N_FEATURES = len(FEATURE_COLS)   # 16

RISK_LABEL_MAP = {"normal": 0, "low": 1, "medium": 2, "high": 3}


# ── Helpers ──────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    try:    return float(val)
    except: return default

def _risk_label(r):
    return RISK_LABEL_MAP.get(str(r.get("risk_level","normal")).lower(), 0)

def _risk_score(r):
    return _safe_float(r.get("risk_score", 0.0))


# ── Rolling buffer ───────────────────────────────────────────────────

class _Buf:
    def __init__(self, n=20): self._d = deque(maxlen=n)
    def push(self, v): self._d.append(float(v))
    def mean(self):
        b=list(self._d); return sum(b)/len(b) if b else 0.0
    def std(self):
        b=list(self._d)
        if len(b)<2: return 0.0
        m=sum(b)/len(b)
        return math.sqrt(sum((x-m)**2 for x in b)/(len(b)-1))
    def trend(self):
        b=list(self._d); n=len(b)
        if n<2: return 0.0
        xm=(n-1)/2; ym=sum(b)/n
        num=sum((i-xm)*(y-ym) for i,y in enumerate(b))
        den=sum((i-xm)**2 for i in range(n))
        return num/den if den else 0.0

class _CompState:
    def __init__(self, w=20): self.b={s:_Buf(w) for s in SENSORS}
    def update(self, r):
        for s in SENSORS: self.b[s].push(_safe_float(r.get(s,0.0)))
    def vec(self, r):
        raw=[_safe_float(r.get(s,0.0)) for s in SENSORS]
        return (raw
                + [self.b[s].mean()  for s in SENSORS]
                + [self.b[s].std()   for s in SENSORS]
                + [self.b[s].trend() for s in SENSORS])

def _enrich(records, window=20):
    """Compute rolling features in-place — works on raw CSV records."""
    states={}
    for rec in records:
        cid=rec.get("component_id","T01")
        if cid not in states: states[cid]=_CompState(window)
        states[cid].update(rec)
        for col,val in zip(FEATURE_COLS, states[cid].vec(rec)):
            rec[col]=val
    return records

def _vec(r):
    return [_safe_float(r.get(c,0.0)) for c in FEATURE_COLS]


# ════════════════════════════════════════════════════════════════════
# SequenceBuilder — forecasting formulation
#
# For each position i:
#   input  = sensor features at steps [i .. i+T-1]
#   label  = risk_level/score at step  [i+T-1+horizon]
#
# The model must predict FUTURE risk from PAST sensor patterns.
# This is genuinely hard and cannot be memorised.
# ════════════════════════════════════════════════════════════════════

class SequenceBuilder:
    def __init__(self, seq_len=12, stride=6, horizon=6):
        self.seq_len = seq_len
        self.stride  = stride
        self.horizon = horizon   # how many steps ahead to predict

    def build(self, records):
        # Group by time_step
        step_map = defaultdict(dict)
        for r in records:
            step = int(_safe_float(r.get("time_step", r.get("record_id", 0))))
            step_map[step][r.get("component_id","T01")] = r

        steps = sorted(step_map.keys())
        T, N, F = self.seq_len, N_NODES, N_FEATURES
        h = self.horizon

        Xl, ycl, yrl = [], [], []

        for i in range(0, len(steps) - T - h + 1, self.stride):
            input_steps  = steps[i : i+T]          # past T steps
            label_step   = steps[i+T-1+h]          # future step to predict

            # ── Input tensor ──────────────────────────────────────
            x  = torch.zeros(T, N, F)
            for t, step in enumerate(input_steps):
                for cid, rec in step_map[step].items():
                    if cid not in COMP_INDEX: continue
                    x[t, COMP_INDEX[cid]] = torch.tensor(
                        _vec(rec), dtype=torch.float32
                    )

            # ── Labels from future step ───────────────────────────
            yc = torch.zeros(N, dtype=torch.long)
            yr = torch.zeros(N)
            for cid, rec in step_map[label_step].items():
                if cid not in COMP_INDEX: continue
                n = COMP_INDEX[cid]
                yc[n] = _risk_label(rec)
                yr[n] = _risk_score(rec)

            Xl.append(x); ycl.append(yc); yrl.append(yr)

        if not Xl:
            return (torch.zeros(0,T,N,F),
                    torch.zeros(0,N,dtype=torch.long),
                    torch.zeros(0,N))

        return torch.stack(Xl), torch.stack(ycl), torch.stack(yrl)


# ── Dataset ──────────────────────────────────────────────────────────

class RailwayDataset(Dataset):
    def __init__(self, X, y_cls, y_reg, edge_index):
        self.X=X; self.y_cls=y_cls; self.y_reg=y_reg; self.edge_index=edge_index
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        return self.X[i], self.y_cls[i], self.y_reg[i], self.edge_index
    @property
    def shape(self): return tuple(self.X.shape)


# ── Edge index ───────────────────────────────────────────────────────

def build_edge_index():
    edges=[
        ("T01","T02"),("T02","T03"),("T03","T04"),("T04","T05"),
        ("T05","T06"),("T06","T07"),("T07","T08"),("T08","T09"),("T09","T10"),
        ("SW1","T01"),("SW1","T02"),("SW1","T03"),
        ("SW2","T03"),("SW2","T04"),("SW2","T05"),
        ("SW3","T05"),("SW3","T06"),("SW3","T07"),
        ("SW4","T07"),("SW4","T08"),("SW4","T09"),
        ("SG1","SW1"),("SG1","SW2"),
        ("SG2","SW2"),("SG2","SW3"),
        ("SG3","SW3"),("SG3","SW4"),
        ("BR1","T02"),("BR1","T03"),
        ("BR2","T05"),("BR2","T06"),
        ("BR3","T08"),("BR3","T09"),
    ]
    src,dst=[],[]
    for a,b in edges:
        if a in COMP_INDEX and b in COMP_INDEX:
            src+=[COMP_INDEX[a],COMP_INDEX[b]]
            dst+=[COMP_INDEX[b],COMP_INDEX[a]]
    return torch.tensor([src,dst],dtype=torch.long)


# ── Loaders ──────────────────────────────────────────────────────────

def load_from_csv(path="data/raw/railway_sensor_data.csv",
                  seq_len=12, stride=6, horizon=6, window=20,
                  train_ratio=0.7, val_ratio=0.15):
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found.")
    records=[]
    with open(p) as fh:
        for row in csv.DictReader(fh): records.append(dict(row))
    print(f"[DataLoader] Loaded {len(records):,} records from {path}")
    print(f"[DataLoader] Computing rolling features (window={window})...")
    records=_enrich(records,window)
    return _split(records,seq_len,stride,horizon,train_ratio,val_ratio)


def load_from_json(path="data/processed/processed_features_sample.json",
                   seq_len=12, stride=6, horizon=6,
                   train_ratio=0.7, val_ratio=0.15):
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found.")
    with open(p) as fh: records=json.load(fh)
    print(f"[DataLoader] Loaded {len(records):,} records from {path}")
    return _split(records,seq_len,stride,horizon,train_ratio,val_ratio)


def _split(records,seq_len,stride,horizon,train_ratio,val_ratio):
    print(f"[DataLoader] Building sequences — seq_len={seq_len} "
          f"stride={stride} horizon={horizon} (predict {horizon} steps ahead)")
    builder=SequenceBuilder(seq_len=seq_len,stride=stride,horizon=horizon)
    X,y_cls,y_reg=builder.build(records)
    edge_index=build_edge_index()
    n=len(X)
    if n==0:
        raise ValueError("No sequences built. Check records have time_step field.")

    flat=y_cls.view(-1).tolist()
    total=len(flat)
    print(f"[DataLoader] Future label distribution:")
    for k,v in [("normal",0),("low",1),("medium",2),("high",3)]:
        cnt=flat.count(v)
        print(f"             {k:8s}: {cnt:8,}  ({cnt/total*100:.1f}%)")

    t1=int(n*train_ratio); t2=int(n*(train_ratio+val_ratio))
    def ds(sl): return RailwayDataset(X[sl],y_cls[sl],y_reg[sl],edge_index)
    train_ds,val_ds,test_ds=ds(slice(0,t1)),ds(slice(t1,t2)),ds(slice(t2,n))
    print(f"[DataLoader] Sequences — train:{len(train_ds)}  "
          f"val:{len(val_ds)}  test:{len(test_ds)}")
    print(f"[DataLoader] Shape     — {train_ds.shape}  (windows, T, N, F)")
    print(f"[DataLoader] Edges     — {edge_index.shape[1]}")
    return train_ds,val_ds,test_ds