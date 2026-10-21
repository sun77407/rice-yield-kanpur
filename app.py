"""Rice yield and advisory app for Kanpur Nagar / Dehat (offline, loopback only).

Launch: streamlit run app.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.app import service as svc

MODEL_C, TREND_C, MUTED = "#2a6fdb", "#d95f02", "#6b7280"
RES = Path("models/results")
SOURCES = Path("kb/raw/SOURCES.md")

st.set_page_config(page_title="Kanpur rice yield and advisory", layout="wide")
st.title("Kanpur rice: yield estimate and offline advisory")
st.caption(
    "Kanpur Nagar and Kanpur Dehat, Kharif season. Scenarios are analogs of past seasons "
    "(2010-2024), not forecasts. Everything runs locally; no data leaves this machine."
)


def predict_tab() -> None:
    """Render the Predict tab."""
    c1, c2, c3, c4 = st.columns(4)
    district = c1.selectbox("District", list(svc.DISTRICTS))
    planting = c2.date_input("Planting date", value=date(2026, 7, 1))
    area = c3.number_input("Land area (ha)", min_value=0.1, max_value=1000.0, value=1.0, step=0.5)
    scenario = c4.selectbox("Weather scenario", list(svc.SCENARIOS), index=1)
    noraw = st.toggle(
        "Sensitivity: exclude raw_peak_ndvi",
        help="Refits without the noisiest NDVI feature (ADR-031/032). Pixel R2 falls from 0.90 "
        "to 0.60 but the district-level conclusion is the same: the time trend is hard to beat.",
    )
    variant = "noraw" if noraw else "full"
    note = svc.planting_note(planting)
    if note:
        st.warning(note)
    y = svc.predict(district, planting, float(area), scenario, variant)
    st.session_state["pred"] = (y, planting)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P10 (t/ha)", f"{y.p10:.2f}")
    m2.metric("P50 (t/ha)", f"{y.p50:.2f}")
    m3.metric("P90 (t/ha)", f"{y.p90:.2f}")
    m4.metric("Time-trend reference", f"{y.trend:.2f}", help=f"+/-{y.trend_rmse:.2f} leave-one-out")
    tot = y.total_tonnes
    st.write(
        f"For {area:g} ha: **{tot[1]:.1f} t** expected ({tot[0]:.1f} to {tot[2]:.1f} t, "
        "P10 to P90)."
    )

    fig = go.Figure()
    fig.add_hrect(
        y0=y.trend - y.trend_rmse,
        y1=y.trend + y.trend_rmse,
        fillcolor=TREND_C,
        opacity=0.12,
        line_width=0,
    )
    fig.add_hline(
        y=y.trend,
        line_color=TREND_C,
        line_width=2,
        annotation_text="time trend (ADR-032)",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=["Model"],
            y=[y.p50],
            mode="markers",
            marker=dict(size=12, color=MODEL_C),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[y.p90 - y.p50],
                arrayminus=[y.p50 - y.p10],
                color=MODEL_C,
                thickness=2,
                width=8,
            ),
            name="Model P10-P50-P90",
            hovertemplate="P50 %{y:.2f} t/ha<extra></extra>",
        )
    )
    fig.add_hline(
        y=y.apy_mean,
        line_color=MUTED,
        line_dash="dot",
        line_width=1,
        annotation_text="2010-2022 mean",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=380,
        yaxis_title="Yield (t/ha)",
        xaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    (st.success if y.verdict == "agree" else st.warning)(y.message)
    st.caption(
        f"Analog years for the {scenario} scenario in {district}: {y.analog_years}. "
        f"Years up to 2020 ({y.in_sample_years}) were in the model's training data, so those "
        "estimates are in-sample. The interval adds the model's leave-one-year-out errors "
        "against APY. The planting date does not enter the yield model; it only frames the "
        "advisory. Headline skill (leave-one-year-out vs APY): model R2 0.38, time trend 0.46."
    )


def _sources() -> dict[str, str]:
    return {}


def advisory_tab() -> None:
    """Render the Advisory tab."""
    if "pred" not in st.session_state:
        st.info("Set the inputs on the Predict tab first.")
        return
    y, planting = st.session_state["pred"]
    st.write(f"{y.district}, {y.scenario} scenario, P50 {y.p50:.2f} t/ha (trend {y.trend:.2f}).")
    if st.button("Generate advisory", type="primary"):
        with st.spinner("Retrieving passages, writing (qwen2.5:14b), judging (llama3)..."):
            st.session_state["adv"] = (y, svc.run_advisory(y, planting))
    if "adv" not in st.session_state:
        return
    y, run = st.session_state["adv"]
    f, adv = run.faithfulness, run.advisory
    a, b, c, d = st.columns(4)
    a.metric("Fully supported", f"{f.frac_supported:.0%}")
    b.metric("Partially supported", f"{f.frac_partial:.0%}")
    c.metric("Unsupported", f"{f.frac_unsupported:.0%}")
    d.metric(
        "Round trip",
        f"{run.total_s:.0f} s",
        help=f"write {run.generate_s:.0f} s, judge {run.judge_s:.0f} s",
    )
    st.caption(
        f"Faithfulness: each advice sentence is split into claims and judged by {f.judge_model} "
        "(a different model from the writer) against only the passages that sentence cites. "
        f"Cosine relevance (diagnostic only): {adv.cosine_relevance_score:.0%} of sentences."
    )
    if f.frac_supported < 0.7:
        st.warning("Fewer than 70% of the advice sentences are fully supported by their citations.")
    st.subheader("Advisory")
    st.markdown(f"**Scenario.** {adv.scenario_text}")
    icon = {"supported": "OK", "partial": "PARTIAL", "unsupported": "UNSUPPORTED"}
    for sv in f.sentences:
        st.markdown(f"- {sv.sentence}  \n  `{icon[sv.verdict]}`")
    st.subheader("Retrieved passages")
    for i, p in enumerate(adv.passages, 1):
        with st.expander(f"[{i}] {p.title} ({p.source_type})"):
            st.write(p.text)
            st.caption(f"Source: {p.source_citation}")
    with st.expander("Claim-level verdicts"):
        for sv in f.sentences:
            st.markdown(f"**{sv.sentence}** (cites {sv.cited})")
            for cl in sv.claims:
                st.markdown(f"- {cl.verdict}: {cl.claim} - _{cl.reason}_")


def _metrics_table() -> pd.DataFrame:
    rows = []
    for name in (
        "xgboost",
        "xgb_no_raw_peak",
        "xgb_no_satellite",
        "linear",
        "ridge",
        "lightgbm",
        "random_forest",
    ):
        p = RES / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        loyo = d.get("agg_apy_loyo", {})
        rows.append(
            {
                "model": name,
                "test R2 (pixel, 2021-24)": d["test"]["combined"]["r2"],
                "test R2 Nagar": d["test"]["KanpurNagar"]["r2"],
                "test R2 Dehat": d["test"]["KanpurDehat"]["r2"],
                "LOYO R2 vs APY (pooled)": loyo.get("pooled", {}).get("r2"),
                "LOYO R2 Nagar": loyo.get("KanpurNagar", {}).get("r2"),
                "LOYO R2 Dehat": loyo.get("KanpurDehat", {}).get("r2"),
            }
        )
    rows.append(
        {"model": "per-district linear time trend (baseline)", "LOYO R2 vs APY (pooled)": 0.464}
    )
    return pd.DataFrame(rows)


def diagnostics_tab() -> None:
    """Render the Diagnostics tab."""
    st.subheader("Model skill")
    st.dataframe(_metrics_table().round(3), hide_index=True, use_container_width=True)
    st.warning(
        "Pixel-level test R2 (about 0.90) is not yield skill: it drops to 0.07 with no satellite "
        "features. The honest figure is the district-aggregated leave-one-year-out R2 against "
        "APY, where the time trend (0.46) beats the best model (0.38). Dehat is weak (0.18)."
    )
    st.subheader("SHAP top features (XGBoost)")
    img = Path("figures/shap_summary.png")
    if img.exists():
        st.image(str(img))
    imp = pd.read_csv(RES / "shap_importance.csv", index_col=0).head(10)
    st.dataframe(imp.rename(columns={"mean_abs_shap": "mean |SHAP|"}).round(3))
    st.subheader("Data lineage")
    lin = []
    for m in sorted(Path("data/raw").glob("*/manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        lin.append(
            {
                "source": m.parent.name,
                "pulled": str(d.get("pulled_at") or d.get("timestamp", "")),
                "files": d.get("file_count", ""),
                "rows": d.get("row_count", ""),
                "url": str(d.get("source_url", ""))[:80],
            }
        )
    st.dataframe(pd.DataFrame(lin), hide_index=True, use_container_width=True)
    st.subheader("Faithfulness history")
    batch = RES / "advisory_faithfulness.json"
    if batch.exists():
        s = json.loads(batch.read_text(encoding="utf-8"))["summary"]
        st.write(
            f"30 scenario runs (llama3 claim-level judge): fully supported "
            f"{s['mean_frac_supported']:.0%}, partial {s['mean_frac_partial']:.0%}, unsupported "
            f"{s['mean_frac_unsupported']:.0%}. Cosine relevance on the same sentences: "
            f"{s['cosine_relevance_mean']:.0%}."
        )
    h = svc.read_history()
    if len(h):
        st.dataframe(h, hide_index=True, use_container_width=True)
        fig = go.Figure(
            go.Scatter(y=h.frac_supported, mode="lines+markers", line=dict(color=MODEL_C, width=2))
        )
        fig.update_layout(
            height=250,
            yaxis=dict(title="fully supported", range=[0, 1]),
            xaxis_title="app advisory #",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No advisories generated in the app yet.")
    st.caption(re.sub(r"\s+", " ", "Source attribution for every passage: kb/raw/SOURCES.md."))


t1, t2, t3 = st.tabs(["Predict", "Advisory", "Diagnostics"])
with t1:
    predict_tab()
with t2:
    advisory_tab()
with t3:
    diagnostics_tab()
