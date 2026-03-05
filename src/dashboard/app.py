"""
Railway Failure Detection Dashboard — Rich Edition
===================================================
Run: streamlit run src/dashboard/app.py
"""

import json, time, math
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────
PRED_PATH   = Path("data/predictions/live_predictions.json")
EVAL_PATH   = Path("data/processed/evaluation_results.json")
RETRAIN_LOG = Path("data/processed/retrain_log.json")
CKPT_PATH   = Path("checkpoints/best_model.pt")

# ── Graph topology ─────────────────────────────────────────────────
# Positions carefully laid out to look like a real railway schematic
NODE_POS = {
    # Track chain (horizontal spine)
    "T01":(0,5),  "T02":(1.5,5), "T03":(3,5),  "T04":(4.5,5),
    "T05":(6,5),  "T06":(7.5,5), "T07":(9,5),  "T08":(10.5,5),
    "T09":(12,5), "T10":(13.5,5),
    # Switches (below tracks at junctions)
    "SW1":(1.5,3), "SW2":(4.5,3), "SW3":(7.5,3), "SW4":(10.5,3),
    # Signals (at bottom, connecting switches)
    "SG1":(3,1),   "SG2":(6,1),   "SG3":(9,1),
    # Bridges (above tracks)
    "BR1":(2.25,7.5), "BR2":(6.75,7.5), "BR3":(11.25,7.5),
}

EDGES = [
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

RISK_COLOR  = {"normal":"#22c55e","low":"#eab308","medium":"#f97316","high":"#ef4444"}
RISK_BG     = {"normal":"#14532d","low":"#713f12","medium":"#7c2d12","high":"#7f1d1d"}
COMP_SYMBOL = {"T":"square","S":"circle","B":"diamond"}

COMPONENT_ORDER = [
    "T01","T02","T03","T04","T05","T06","T07","T08","T09","T10",
    "SW1","SW2","SW3","SW4","SG1","SG2","SG3","BR1","BR2","BR3"
]

# ── Data loaders ───────────────────────────────────────────────────
@st.cache_data(ttl=3)
def load_predictions():
    if not PRED_PATH.exists(): return []
    try:
        with open(PRED_PATH) as f: return json.load(f)
    except: return []

@st.cache_data(ttl=30)
def load_eval():
    if not EVAL_PATH.exists(): return {}
    try:
        with open(EVAL_PATH) as f: return json.load(f)
    except: return {}

@st.cache_data(ttl=30)
def load_retrain():
    if not RETRAIN_LOG.exists(): return []
    try:
        with open(RETRAIN_LOG) as f: return json.load(f)
    except: return []

# ── Chart builders ─────────────────────────────────────────────────

def network_graph(per_node: dict, root_cause: str) -> go.Figure:
    fig = go.Figure()

    # Draw edges — colour by risk of connected nodes
    for src, dst in EDGES:
        x0,y0 = NODE_POS[src]; x1,y1 = NODE_POS[dst]
        src_risk = per_node.get(src,{}).get("risk_class","normal")
        dst_risk = per_node.get(dst,{}).get("risk_class","normal")
        # Edge glows red if either endpoint is high risk
        edge_levels = [src_risk, dst_risk]
        if "high" in edge_levels:    ec,ew = "#ef4444",3
        elif "medium" in edge_levels:ec,ew = "#f97316",2
        elif "low" in edge_levels:   ec,ew = "#eab308",1.5
        else:                        ec,ew = "#334155",1
        fig.add_trace(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None], mode="lines",
            line=dict(color=ec, width=ew), hoverinfo="none", showlegend=False
        ))

    # Draw nodes
    for comp, pos in NODE_POS.items():
        info    = per_node.get(comp, {})
        rclass  = info.get("risk_class","normal")
        rscore  = info.get("risk_score",0.0)
        rca     = info.get("root_cause_attn",0.0)
        color   = RISK_COLOR[rclass]
        size    = 22 + rscore * 28
        is_root = comp == root_cause

        # Root cause gets a special ring
        if is_root:
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]], mode="markers",
                marker=dict(size=size+12, color="rgba(239,68,68,0.3)",
                            line=dict(color="#ef4444",width=3)),
                hoverinfo="none", showlegend=False
            ))

        ctype = "T" if comp.startswith("T") else ("S" if comp[0]=="S" else "B")
        symbol = {"T":"square","S":"circle","B":"diamond"}[ctype]

        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]], mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color="white" if not is_root else "#fbbf24",
                                  width=2 if not is_root else 3)),
            text=[f"{'⭐' if is_root else ''}{comp}"],
            textposition="top center",
            textfont=dict(size=10, color="white", family="monospace"),
            hovertemplate=(
                f"<b>{comp}</b><br>"
                f"Risk: <b>{rclass.upper()}</b><br>"
                f"Score: {rscore:.3f}<br>"
                f"Root cause attn: {rca:.4f}"
                + ("<br><b>⭐ MOST LIKELY ROOT CAUSE</b>" if is_root else "")
                + "<extra></extra>"
            ),
            showlegend=False
        ))

    # Zone labels
    for label, x, y in [("TRACKS",6.75,5.8),("SWITCHES",6,3.8),
                          ("SIGNALS",6,1.8),("BRIDGES",6.75,6.8)]:
        fig.add_annotation(x=x,y=y,text=label,showarrow=False,
                           font=dict(size=9,color="#475569"),
                           bgcolor="rgba(15,23,42,0.5)")

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        margin=dict(l=10,r=10,t=10,b=10), height=420,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                   range=[-0.8,14.5]),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                   range=[0,9]),
    )
    return fig


def risk_heatmap(predictions: list) -> go.Figure:
    """Risk score heatmap over time for all components."""
    if len(predictions) < 2:
        return go.Figure()

    last_n = predictions[-30:]
    times  = [p.get("timestamp","")[-8:-3] for p in last_n]
    comps  = COMPONENT_ORDER

    z = []
    for comp in comps:
        row = [p.get("per_node",{}).get(comp,{}).get("risk_score",0.0)
               for p in last_n]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=times, y=comps,
        colorscale=[[0,"#22c55e"],[0.3,"#eab308"],
                    [0.6,"#f97316"],[1,"#ef4444"]],
        zmin=0, zmax=1,
        hovertemplate="Component: %{y}<br>Time: %{x}<br>Risk: %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="Risk",font=dict(color="white")),
                      tickfont=dict(color="white")),
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="white", height=380,
        margin=dict(l=60,r=20,t=30,b=50),
        title=dict(text="Risk Score Heatmap — Last 30 Predictions",
                   font=dict(color="white",size=14)),
        xaxis=dict(tickangle=-45, gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", tickfont=dict(size=10)),
    )
    return fig


def risk_trend(predictions: list, comp: str) -> go.Figure:
    """Line chart of risk score over time for one component."""
    if len(predictions) < 2: return go.Figure()
    last_n = predictions[-50:]
    times  = list(range(len(last_n)))
    scores = [p.get("per_node",{}).get(comp,{}).get("risk_score",0.0)
              for p in last_n]
    classes= [p.get("per_node",{}).get(comp,{}).get("risk_class","normal")
              for p in last_n]
    colors = [RISK_COLOR[c] for c in classes]

    fig = go.Figure()
    # Threshold bands
    fig.add_hrect(y0=0.75,y1=1.0,fillcolor="rgba(239,68,68,0.08)",line_width=0)
    fig.add_hrect(y0=0.50,y1=0.75,fillcolor="rgba(249,115,22,0.08)",line_width=0)
    fig.add_hrect(y0=0.25,y1=0.50,fillcolor="rgba(234,179,8,0.08)",line_width=0)
    fig.add_hline(y=0.75,line_dash="dash",line_color="#ef4444",
                  annotation_text="HIGH",annotation_font_color="#ef4444")
    fig.add_hline(y=0.50,line_dash="dash",line_color="#f97316",
                  annotation_text="MEDIUM",annotation_font_color="#f97316")
    fig.add_hline(y=0.25,line_dash="dot",line_color="#eab308",
                  annotation_text="LOW",annotation_font_color="#eab308")

    fig.add_trace(go.Scatter(
        x=times, y=scores, mode="lines+markers",
        line=dict(color="#6366f1",width=2),
        marker=dict(color=colors,size=6,line=dict(color="white",width=1)),
        hovertemplate="Step %{x}<br>Risk: %{y:.3f}<extra></extra>",
        name=comp
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="white", height=220,
        margin=dict(l=10,r=10,t=10,b=30),
        yaxis=dict(range=[0,1],gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        showlegend=False,
    )
    return fig


def root_cause_bar(per_node: dict) -> go.Figure:
    comps = COMPONENT_ORDER
    attns = [per_node.get(c,{}).get("root_cause_attn",0.0) for c in comps]
    riscs = [per_node.get(c,{}).get("risk_class","normal") for c in comps]
    colors= [RISK_COLOR[r] for r in riscs]
    pairs = sorted(zip(attns,comps,colors), reverse=True)
    attns_s,comps_s,colors_s = zip(*pairs) if pairs else ([],[],[])

    fig = go.Figure(go.Bar(
        x=list(comps_s), y=list(attns_s),
        marker_color=list(colors_s),
        marker_line=dict(color="white",width=0.5),
        hovertemplate="%{x}: %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="white", height=240,
        margin=dict(l=10,r=10,t=10,b=30),
        yaxis=dict(gridcolor="#334155",title="Attention"),
        xaxis=dict(gridcolor="#334155"),
    )
    return fig


def all_risk_bar(per_node: dict) -> go.Figure:
    comps  = COMPONENT_ORDER
    scores = [per_node.get(c,{}).get("risk_score",0.0) for c in comps]
    riscs  = [per_node.get(c,{}).get("risk_class","normal") for c in comps]
    colors = [RISK_COLOR[r] for r in riscs]

    fig = go.Figure(go.Bar(
        x=comps, y=scores, marker_color=colors,
        marker_line=dict(color="white",width=0.5),
        hovertemplate="%{x}<br>Risk: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.75, line_dash="dash", line_color="#ef4444")
    fig.add_hline(y=0.50, line_dash="dash", line_color="#f97316")
    fig.add_hline(y=0.25, line_dash="dot",  line_color="#eab308")
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="white", height=240,
        margin=dict(l=10,r=10,t=10,b=30),
        yaxis=dict(range=[0,1],gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
    )
    return fig


def propagation_map(predictions: list) -> go.Figure:
    """
    Shows which components went high-risk over time — propagation view.
    Each row = component, each col = prediction step.
    Cell color = risk class.
    """
    if len(predictions) < 3: return go.Figure()
    last_n   = predictions[-20:]
    comps    = COMPONENT_ORDER
    risk_num = {"normal":0,"low":1,"medium":2,"high":3}

    z = []
    for comp in comps:
        row = [risk_num.get(p.get("per_node",{}).get(comp,{})
               .get("risk_class","normal"),0) for p in last_n]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, y=comps,
        colorscale=[[0,"#22c55e"],[0.33,"#eab308"],
                    [0.66,"#f97316"],[1,"#ef4444"]],
        zmin=0, zmax=3, showscale=False,
        hovertemplate="<b>%{y}</b><br>Step %{x}<br>%{z}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="white", height=380,
        margin=dict(l=60,r=20,t=30,b=30),
        title=dict(text="Failure Propagation View — Last 20 Steps",
                   font=dict(color="white",size=14)),
        xaxis=dict(title="Prediction step",gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155",tickfont=dict(size=10)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# Page
# ════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Railway Failure Detection",
        page_icon="🚆", layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    .stApp { background:#0f172a; color:#e2e8f0; font-family:'Inter',sans-serif; }
    .metric-box { background:#1e293b; border-radius:10px; padding:16px 20px;
                  border-left:4px solid #6366f1; margin:4px 0; }
    .metric-val { font-size:2rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
    .metric-lbl { font-size:0.75rem; color:#94a3b8; text-transform:uppercase;
                  letter-spacing:0.1em; }
    .alert-card { border-radius:8px; padding:12px 16px; margin:6px 0;
                  border-left:4px solid; }
    .alert-CRITICAL { border-color:#ef4444; background:#1c0a0a; }
    .alert-HIGH     { border-color:#f97316; background:#1c1006; }
    .alert-MEDIUM   { border-color:#eab308; background:#1c1a06; }
    .alert-LOW      { border-color:#22c55e; background:#061c0a; }
    .node-badge { display:inline-block; padding:2px 8px; border-radius:4px;
                  font-size:0.8rem; font-family:'JetBrains Mono',monospace;
                  font-weight:700; margin:2px; }
    .badge-normal { background:#14532d; color:#22c55e; }
    .badge-low    { background:#713f12; color:#eab308; }
    .badge-medium { background:#7c2d12; color:#f97316; }
    .badge-high   { background:#7f1d1d; color:#ef4444; }
    .section-header { font-size:0.7rem; text-transform:uppercase;
                      letter-spacing:0.15em; color:#475569;
                      border-bottom:1px solid #1e293b;
                      padding-bottom:6px; margin:12px 0 8px 0; }
    div[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🚆 Railway Monitor")
        st.caption("Graph-Based Failure Detection")
        st.divider()

        auto_refresh = st.toggle("Live refresh (5s)", value=True)
        refresh_rate = st.slider("Refresh interval (s)", 2, 30, 5)
        st.divider()

        st.markdown('<div class="section-header">COMPONENT DRILL-DOWN</div>',
                    unsafe_allow_html=True)
        selected_comp = st.selectbox("Inspect component", COMPONENT_ORDER,
                                      index=4)   # default T05
        st.divider()

        st.markdown('<div class="section-header">SYSTEM STATUS</div>',
                    unsafe_allow_html=True)
        pred_ok = PRED_PATH.exists()
        ckpt_ok = CKPT_PATH.exists()
        eval_ok = EVAL_PATH.exists()
        st.markdown(f"{'🟢' if pred_ok else '🔴'} Live predictions")
        st.markdown(f"{'🟢' if ckpt_ok else '🔴'} Model checkpoint")
        st.markdown(f"{'🟢' if eval_ok else '🔴'} Evaluation results")
        st.divider()

        if st.button("🔄 Force refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # ── Load data ─────────────────────────────────────────────────
    predictions = load_predictions()
    eval_data   = load_eval()
    retrain_log = load_retrain()
    latest      = predictions[-1] if predictions else None

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
      <span style='font-size:1.8rem;font-weight:700;'>🚆 Smart Railway</span>
      <span style='font-size:1.8rem;font-weight:300;color:#94a3b8;'>
        Failure Detection System</span>
      <div style='font-size:0.8rem;color:#475569;margin-top:4px;
           font-family:JetBrains Mono,monospace;'>
        Temporal GNN · Kafka Streaming · Real-time Root Cause Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    if latest is None:
        st.info("""
        **Waiting for live predictions...**

        Start the full pipeline:
        ```
        Terminal A:  python -m src.kafka.consumer
        Terminal B:  python -m src.kafka.live_sensor_simulator --interval 2
        Terminal C:  streamlit run src/dashboard/app.py
        ```
        Predictions appear after ~24 seconds (model warm-up).
        """)
        if auto_refresh:
            time.sleep(refresh_rate)
            st.cache_data.clear()
            st.rerun()
        return

    per_node    = latest.get("per_node", {})
    root_src    = latest.get("most_likely_source","")
    high_risk   = latest.get("high_risk_nodes",[])
    n_high      = sum(1 for p in predictions
                      if p.get("high_risk_nodes"))

    # ── KPI Row ───────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    overall = eval_data.get("overall",{})

    with k1:
        st.metric("Total Predictions", f"{len(predictions):,}")
    with k2:
        hi = sum(1 for c in COMPONENT_ORDER
                 if per_node.get(c,{}).get("risk_class") == "high")
        st.metric("🔴 High Risk Now", hi,
                  delta="CRITICAL" if hi > 0 else "Normal")
    with k3:
        med = sum(1 for c in COMPONENT_ORDER
                  if per_node.get(c,{}).get("risk_class") == "medium")
        st.metric("🟠 Medium Risk Now", med)
    with k4:
        st.metric("Root Cause", root_src or "—")
    with k5:
        acc = overall.get("accuracy")
        st.metric("Model Accuracy", f"{acc:.3f}" if acc else "—")
    with k6:
        auc = overall.get("auc_roc")
        st.metric("AUC-ROC", f"{auc:.3f}" if auc else "—")

    st.divider()

    # ── Main layout ───────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">RAILWAY NETWORK — LIVE RISK MAP</div>',
                    unsafe_allow_html=True)
        ts = latest.get("timestamp","")[:19].replace("T"," ")
        st.caption(f"Last update: {ts}  |  "
                   f"Step: {latest.get('step',0)}")
        st.plotly_chart(network_graph(per_node, root_src),
                        use_container_width=True)

        # Legend
        lc = st.columns(4)
        for col, (label, cls) in zip(lc, [
            ("Normal","normal"),("Low","low"),
            ("Medium","medium"),("High","high")
        ]):
            col.markdown(
                f'<span class="node-badge badge-{cls}">⬤ {label}</span>',
                unsafe_allow_html=True
            )

    with col_right:
        st.markdown('<div class="section-header">ROOT CAUSE ANALYSIS</div>',
                    unsafe_allow_html=True)

        if root_src:
            rc_class = per_node.get(root_src,{}).get("risk_class","normal")
            st.markdown(
                f'<div class="alert-card alert-{rc_class.upper()}">'
                f'<div style="font-size:1.1rem;font-weight:700;">⭐ {root_src}</div>'
                f'<div style="font-size:0.85rem;color:#94a3b8;">Most likely failure origin</div>'
                f'<div style="font-size:0.9rem;margin-top:4px;">'
                f'Risk class: <b>{rc_class.upper()}</b> · '
                f'Score: <b>{per_node.get(root_src,{}).get("risk_score",0):.3f}</b>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        if high_risk:
            st.markdown("**Affected components:**")
            badges = " ".join(
                f'<span class="node-badge badge-'
                f'{per_node.get(c,{}).get("risk_class","normal")}">{c}</span>'
                for c in high_risk
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.success("✅ All components operating normally")

        st.markdown('<div class="section-header" style="margin-top:16px;">'
                    'ROOT CAUSE ATTENTION</div>', unsafe_allow_html=True)
        st.plotly_chart(root_cause_bar(per_node), use_container_width=True)

    st.divider()

    # ── Component drill-down ──────────────────────────────────────
    st.markdown(f'<div class="section-header">COMPONENT DRILL-DOWN — {selected_comp}</div>',
                unsafe_allow_html=True)

    comp_info  = per_node.get(selected_comp, {})
    comp_class = comp_info.get("risk_class","normal")
    comp_score = comp_info.get("risk_score",0.0)
    comp_rca   = comp_info.get("root_cause_attn",0.0)

    dc1,dc2,dc3,dc4 = st.columns(4)
    dc1.metric("Risk Class", comp_class.upper())
    dc2.metric("Risk Score", f"{comp_score:.4f}")
    dc3.metric("Root Cause Attn", f"{comp_rca:.5f}")
    dc4.metric("Is Root Cause", "⭐ YES" if selected_comp == root_src else "No")

    st.plotly_chart(risk_trend(predictions, selected_comp),
                    use_container_width=True)

    st.divider()

    # ── Risk bar + propagation ─────────────────────────────────────
    rb_col, pm_col = st.columns(2)

    with rb_col:
        st.markdown('<div class="section-header">CURRENT RISK — ALL COMPONENTS</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(all_risk_bar(per_node), use_container_width=True)

    with pm_col:
        st.markdown('<div class="section-header">FAILURE PROPAGATION VIEW</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(propagation_map(predictions), use_container_width=True)

    st.divider()

    # ── Risk heatmap over time ─────────────────────────────────────
    st.markdown('<div class="section-header">RISK HEATMAP OVER TIME</div>',
                unsafe_allow_html=True)
    st.plotly_chart(risk_heatmap(predictions), use_container_width=True)

    st.divider()

    # ── Prediction history table ──────────────────────────────────
    st.markdown('<div class="section-header">PREDICTION HISTORY</div>',
                unsafe_allow_html=True)
    if len(predictions) > 1:
        rows = []
        for p in predictions[-40:][::-1]:
            hi_nodes = ", ".join(p.get("high_risk_nodes",[]))
            avg_risk = sum(v.get("risk_score",0)
                          for v in p.get("per_node",{}).values()) / max(1,20)
            rows.append({
                "Timestamp":     p.get("timestamp","")[:19].replace("T"," "),
                "Root Cause":    p.get("most_likely_source",""),
                "High Risk":     hi_nodes or "None",
                "Avg Risk":      f"{avg_risk:.3f}",
                "Step":          p.get("step",""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=280,
                     hide_index=True)

    st.divider()

    # ── Model performance ──────────────────────────────────────────
    st.markdown('<div class="section-header">MODEL PERFORMANCE</div>',
                unsafe_allow_html=True)
    if eval_data:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy",  f"{overall.get('accuracy',0):.4f}")
        m2.metric("AUC-ROC",   f"{overall.get('auc_roc',0):.4f}")
        m3.metric("MAE",       f"{overall.get('mae',0):.5f}")
        m4.metric("RMSE",      f"{overall.get('rmse',0):.5f}")

        per_class = eval_data.get("per_class",{})
        if per_class:
            rows = []
            for cls, metrics in per_class.items():
                rows.append({
                    "Class":     cls,
                    "Precision": f"{metrics.get('precision',0):.4f}",
                    "Recall":    f"{metrics.get('recall',0):.4f}",
                    "F1":        f"{metrics.get('f1',0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
    else:
        st.info("Run `python -m src.model.evaluator` to see model metrics here.")

    if retrain_log:
        with st.expander("🔁 Retraining history"):
            rows = [{"Date":e.get("timestamp","")[:10],
                     "Reason":e.get("reason",""),
                     "Accuracy":e.get("metrics",{}).get("accuracy",""),
                     "AUC":e.get("metrics",{}).get("auc_roc","")}
                    for e in retrain_log[::-1]]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    # ── Auto-refresh ───────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_rate)
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()