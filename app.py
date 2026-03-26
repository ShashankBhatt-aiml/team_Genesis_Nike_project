import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Nike AI Decision Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="stApp"] {
  background: #0b0b0b;
  color: #ffffff;
  font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

[data-testid="stAppViewContainer"] > .main > div {
  max-width: 1320px;
  padding-top: 1rem;
  padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
  background: #0f0f0f;
  border-right: 1px solid rgba(255,255,255,0.08);
}

.subtle { color: rgba(255,255,255,0.70); font-size: 0.95rem; line-height: 1.35; }
.section-space { margin-top: 26px; }
.section-title {
  font-size: 1.1rem; font-weight: 800; letter-spacing: -0.01em;
  margin-bottom: 10px; color: #ffffff;
}

.hero {
  text-align: center;
  padding: 22px 0 10px 0;
  animation: fadeInUp 0.7s ease both;
}
.hero-title {
  font-size: clamp(2.2rem, 4.8vw, 4rem);
  font-weight: 900;
  letter-spacing: -0.03em;
  line-height: 1.04;
  background: linear-gradient(90deg, #ffffff 0%, #d5d5d5 40%, #9f9f9f 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-subtitle {
  margin-top: 8px;
  font-size: 1.06rem;
  font-weight: 750;
  letter-spacing: -0.01em;
}
.divider {
  height: 1px;
  margin: 14px auto;
  width: 86%;
  background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.16), rgba(255,255,255,0.0));
}

.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow:
    0 10px 26px rgba(0,0,0,0.32),
    0 0 0 1px rgba(255,255,255,0.02) inset;
  transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.card:hover {
  transform: scale(1.02);
  box-shadow:
    0 14px 36px rgba(0,0,0,0.45),
    0 0 22px rgba(54,211,153,0.14);
  border-color: rgba(255,255,255,0.18);
}

.kpi-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 14px;
  backdrop-filter: blur(12px);
  box-shadow:
    0 12px 28px rgba(0,0,0,0.35),
    0 0 0 1px rgba(255,255,255,0.025) inset;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover {
  transform: scale(1.03);
  box-shadow:
    0 18px 42px rgba(0,0,0,0.5),
    0 0 24px rgba(255,255,255,0.12);
}
.kpi-label { font-size: 0.88rem; color: rgba(255,255,255,0.72); font-weight: 700; }
.kpi-value { font-size: 2.0rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.05; margin-top: 6px; }

.chip {
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:0.79rem; font-weight:800; border:1px solid rgba(255,255,255,0.16);
}
.recommend-border {
  border-left: 4px solid rgba(54,211,153,0.95);
  padding-left: 11px;
}

.fade-page { animation: fadePage 0.7s ease both; }
.delay-1 { animation: fadeInUp 0.65s ease both; animation-delay: 0.08s; }
.delay-2 { animation: fadeInUp 0.65s ease both; animation-delay: 0.16s; }
.delay-3 { animation: fadeInUp 0.65s ease both; animation-delay: 0.24s; }
.delay-4 { animation: fadeInUp 0.65s ease both; animation-delay: 0.32s; }

@keyframes fadePage {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


DATA_PATH = os.path.join(os.path.dirname(__file__), "train.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(str).str.upper().eq("TRUE").astype(int)
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(["Store", "Dept"], sort=False)["Weekly_Sales"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)
    df["rolling_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["rolling_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year
    df["dow"] = df["Date"].dt.dayofweek
    return df


FEATURE_COLS = [
    "Store",
    "Dept",
    "IsHoliday",
    "month",
    "weekofyear",
    "year",
    "dow",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_4",
    "rolling_std_4",
]


@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float


@st.cache_resource(show_spinner=False)
def train_model(df_feat: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, Metrics]:
    train_df = df_feat.dropna(subset=FEATURE_COLS + ["Weekly_Sales"]).copy()
    train_df = train_df.fillna(0).sort_values("Date")

    cutoff = train_df["Date"].quantile(0.8)
    fit_df = train_df[train_df["Date"] < cutoff]
    val_df = train_df[train_df["Date"] >= cutoff]

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=240,
        random_state=42,
    )
    model.fit(fit_df[FEATURE_COLS], fit_df["Weekly_Sales"])
    pred = model.predict(val_df[FEATURE_COLS])
    return model, Metrics(r2=float(r2_score(val_df["Weekly_Sales"], pred)), mae=float(mean_absolute_error(val_df["Weekly_Sales"], pred)))


@st.cache_data(show_spinner=False)
def feature_influence_proxy(df_feat: pd.DataFrame) -> pd.DataFrame:
    sample = df_feat.dropna(subset=FEATURE_COLS + ["Weekly_Sales"]).sample(min(60000, len(df_feat)), random_state=42)
    # Correlation-based influence (fast and explainable)
    influences = []
    target = sample["Weekly_Sales"]
    for col in FEATURE_COLS:
        corr = sample[col].corr(target)
        influences.append((col, float(abs(corr) if pd.notna(corr) else 0.0)))
    out = pd.DataFrame(influences, columns=["Feature", "Influence"]).sort_values("Influence", ascending=False)
    return out


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x:+.1f}%"


def build_filtered(df: pd.DataFrame, store_sel: str, dept_sel: str, date_start: pd.Timestamp, date_end: pd.Timestamp) -> pd.DataFrame:
    f = df[(df["Date"] >= date_start) & (df["Date"] <= date_end)].copy()
    if store_sel != "All Stores":
        f = f[f["Store"] == int(store_sel)]
    if dept_sel != "All Departments":
        f = f[f["Dept"] == int(dept_sel)]
    return f


def scenario_multiplier(mode: str) -> float:
    if mode == "Holiday Mode":
        return 1.12
    if mode == "High Demand Mode":
        return 1.22
    return 1.00


def series_forecast(
    model: HistGradientBoostingRegressor,
    full_df: pd.DataFrame,
    store_sel: str,
    dept_sel: str,
    holiday_toggle: bool,
    promotion_toggle: bool,
    mode: str,
    demand_uplift_pct: float,
    horizon_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      history_weekly: Date, Weekly_Sales
      future_forecast: Date, Predicted
    """
    if store_sel != "All Stores" and dept_sel != "All Departments":
        s_id = int(store_sel)
        d_id = int(dept_sel)
        hist = full_df[(full_df["Store"] == s_id) & (full_df["Dept"] == d_id)].sort_values("Date").copy()
        hist_weekly = hist.groupby("Date", as_index=False)["Weekly_Sales"].sum()
        if len(hist) < 10:
            return hist_weekly, pd.DataFrame(columns=["Date", "Predicted"])

        sales = hist["Weekly_Sales"].astype(float).tolist()
        last_date = hist["Date"].max()
        preds: list[float] = []
        for i in range(1, horizon_weeks + 1):
            d = last_date + pd.Timedelta(weeks=i)
            lag_1, lag_2, lag_3 = sales[-1] if len(preds) == 0 else preds[-1], sales[-2] if len(preds) == 0 else (sales[-1] if len(preds) == 1 else preds[-2]), sales[-3] if len(preds) == 0 else (sales[-2] if len(preds) == 1 else (sales[-1] if len(preds) == 2 else preds[-3]))
            recent = (sales + preds)[-4:]
            row = pd.DataFrame(
                [
                    {
                        "Store": s_id,
                        "Dept": d_id,
                        "IsHoliday": 1 if holiday_toggle else 0,
                        "month": d.month,
                        "weekofyear": int(d.isocalendar().week),
                        "year": d.year,
                        "dow": d.dayofweek,
                        "lag_1": float(lag_1),
                        "lag_2": float(lag_2),
                        "lag_3": float(lag_3),
                        "rolling_mean_4": float(np.mean(recent)),
                        "rolling_std_4": float(np.std(recent, ddof=0)),
                    }
                ]
            )
            p = float(model.predict(row)[0])
            preds.append(max(0.0, p))

        fut = pd.DataFrame({"Date": [last_date + pd.Timedelta(weeks=i) for i in range(1, horizon_weeks + 1)], "Predicted": preds})
    else:
        # Aggregate fallback forecast when user selects broad drilldown level.
        base = full_df.copy()
        if store_sel != "All Stores":
            base = base[base["Store"] == int(store_sel)]
        if dept_sel != "All Departments":
            base = base[base["Dept"] == int(dept_sel)]
        hist_weekly = base.groupby("Date", as_index=False)["Weekly_Sales"].sum().sort_values("Date")
        if len(hist_weekly) < 12:
            return hist_weekly, pd.DataFrame(columns=["Date", "Predicted"])
        last_date = hist_weekly["Date"].max()
        recent = hist_weekly["Weekly_Sales"].tail(8).values
        baseline = float(np.mean(recent))
        # slight trend component
        slope = float((np.mean(recent[-4:]) - np.mean(recent[:4])) / 4.0)
        preds = [max(0.0, baseline + slope * i) for i in range(1, horizon_weeks + 1)]
        fut = pd.DataFrame({"Date": [last_date + pd.Timedelta(weeks=i) for i in range(1, horizon_weeks + 1)], "Predicted": preds})

    # apply scenario/promo/what-if multipliers
    m = scenario_multiplier(mode)
    if promotion_toggle:
        m *= 1.08
    m *= 1.0 + (demand_uplift_pct / 100.0)
    fut["Predicted"] = fut["Predicted"] * m
    return hist_weekly, fut


def trend_chart(history_w: pd.DataFrame, forecast_w: pd.DataFrame, mae: float) -> go.Figure:
    fig = go.Figure()
    if len(history_w) > 0:
        fig.add_trace(
            go.Scatter(
                x=history_w["Date"],
                y=history_w["Weekly_Sales"],
                mode="lines",
                name="Historical",
                line=dict(color="rgba(255,255,255,0.56)", width=2.4),
                hovertemplate="%{x|%Y-%m-%d}<br>Actual=%{y:,.0f}<extra></extra>",
            )
        )
    if len(forecast_w) > 0:
        upper = forecast_w["Predicted"] + mae
        lower = np.maximum(0.0, forecast_w["Predicted"] - mae)
        fig.add_trace(
            go.Scatter(
                x=forecast_w["Date"].tolist() + forecast_w["Date"][::-1].tolist(),
                y=np.concatenate([upper.values, lower.values[::-1]]),
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(255,255,255,0.09)",
                hoverinfo="skip",
                showlegend=False,
                name="Confidence",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_w["Date"],
                y=forecast_w["Predicted"],
                mode="lines",
                name="Forecast",
                line=dict(color="#ffffff", width=3.6, shape="spline"),
                hovertemplate="%{x|%Y-%m-%d}<br>Forecast=%{y:,.0f}<extra></extra>",
            )
        )
        # glowing endpoint marker
        fig.add_trace(
            go.Scatter(
                x=[forecast_w["Date"].iloc[-1]],
                y=[forecast_w["Predicted"].iloc[-1]],
                mode="markers",
                marker=dict(size=15, color="rgba(54,211,153,0.95)", line=dict(width=2, color="#0b0b0b")),
                name="Forecast point",
                showlegend=False,
            )
        )

        # Animate forecast drawing effect
        frames = []
        for i in range(1, len(forecast_w) + 1):
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(x=forecast_w["Date"].iloc[:i], y=forecast_w["Predicted"].iloc[:i]),
                    ],
                    name=f"f{i}",
                )
            )
        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 1.0,
                    "y": 1.18,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 120, "redraw": False}, "transition": {"duration": 120}, "fromcurrent": True}],
                        }
                    ],
                }
            ]
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        hovermode="x unified",
        transition_duration=320,
    )
    return fig


def risk_level(demand: float, supply: float) -> str:
    if demand <= 0:
        return "Low"
    ratio = supply / demand
    if ratio < 0.90 or ratio > 1.40:
        return "High"
    if ratio < 1.00 or ratio > 1.20:
        return "Medium"
    return "Low"


def risk_color(level: str) -> str:
    return {"Low": "#36D399", "Medium": "#FBBF24", "High": "#EF4444"}[level]


def main() -> None:
    st.markdown("<div class='fade-page'>", unsafe_allow_html=True)

    # Loading experience
    with st.spinner("Analyzing demand patterns..."):
        df_raw = load_data(DATA_PATH)
        df_feat = build_features(df_raw)
        model, metrics = train_model(df_feat)
        influence_df = feature_influence_proxy(df_feat)
        time.sleep(0.35)

    # -------- Sidebar Controls (interactive filters) --------
    st.sidebar.markdown("### 🎛️ Interactive Controls")
    all_stores = sorted(df_raw["Store"].unique().tolist())
    store_options = ["All Stores"] + [str(s) for s in all_stores]
    store_sel = st.sidebar.selectbox("Store selector", store_options, index=1 if len(store_options) > 1 else 0)

    if store_sel == "All Stores":
        dept_pool = sorted(df_raw["Dept"].unique().tolist())
    else:
        dept_pool = sorted(df_raw[df_raw["Store"] == int(store_sel)]["Dept"].unique().tolist())
    dept_options = ["All Departments"] + [str(d) for d in dept_pool]
    dept_sel = st.sidebar.selectbox("Department selector", dept_options, index=1 if len(dept_options) > 1 else 0)

    min_d, max_d = df_raw["Date"].min().date(), df_raw["Date"].max().date()
    d_start, d_end = st.sidebar.slider(
        "Date range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
    )
    date_start = pd.Timestamp(d_start)
    date_end = pd.Timestamp(d_end)

    holiday_toggle = st.sidebar.toggle("Holiday toggle", value=True)
    promo_toggle = st.sidebar.toggle("Promotion toggle", value=False)
    scenario_mode = st.sidebar.radio("Scenario mode", ["Normal Mode", "Holiday Mode", "High Demand Mode"], index=0)

    # What-if controls in sidebar (most used parameters)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 What-if Inputs")
    demand_increase_pct = st.sidebar.slider("Demand increase (%)", min_value=-20, max_value=50, value=10, step=1)
    inventory_increase_pct = st.sidebar.slider("Inventory increase (%)", min_value=-20, max_value=50, value=15, step=1)
    sim_promo = st.sidebar.toggle("Promotion in simulation", value=False)
    sim_season = st.sidebar.selectbox("Season selector", ["Normal", "Holiday", "High Demand"], index=0)

    # -------- Data according to filters --------
    filtered_df = build_filtered(df_raw, store_sel, dept_sel, date_start, date_end)
    history_w, forecast_w = series_forecast(
        model=model,
        full_df=filtered_df if len(filtered_df) > 0 else df_raw,
        store_sel=store_sel,
        dept_sel=dept_sel,
        holiday_toggle=holiday_toggle,
        promotion_toggle=promo_toggle,
        mode=scenario_mode,
        demand_uplift_pct=0.0,
        horizon_weeks=8,
    )

    if len(history_w) == 0:
        st.error("No data found for selected filters. Please adjust Store/Department/Date range.")
        return

    base_forecast_sum = float(forecast_w["Predicted"].sum()) if len(forecast_w) > 0 else 0.0
    recent_actual_sum = float(history_w["Weekly_Sales"].tail(8).sum())
    demand_trend_pct = ((base_forecast_sum / recent_actual_sum - 1) * 100.0) if recent_actual_sum > 0 else 0.0

    base_supply = max(1.0, float(history_w["Weekly_Sales"].tail(8).mean()) * 8 * 1.08)
    current_risk = risk_level(base_forecast_sum, base_supply)
    expected_revenue = base_forecast_sum * 1.24

    # Simulation outputs
    sim_mode = {"Normal": "Normal Mode", "Holiday": "Holiday Mode", "High Demand": "High Demand Mode"}[sim_season]
    _, sim_forecast = series_forecast(
        model=model,
        full_df=filtered_df if len(filtered_df) > 0 else df_raw,
        store_sel=store_sel,
        dept_sel=dept_sel,
        holiday_toggle=holiday_toggle,
        promotion_toggle=sim_promo,
        mode=sim_mode,
        demand_uplift_pct=float(demand_increase_pct),
        horizon_weeks=8,
    )
    sim_demand = float(sim_forecast["Predicted"].sum()) if len(sim_forecast) > 0 else base_forecast_sum
    sim_supply = base_supply * (1.0 + inventory_increase_pct / 100.0)
    sim_risk = risk_level(sim_demand, sim_supply)

    # Recommendation engine state
    if "rec_action" not in st.session_state:
        st.session_state.rec_action = "Optimize Inventory"
    if "show_alert_1" not in st.session_state:
        st.session_state.show_alert_1 = False
    if "show_alert_2" not in st.session_state:
        st.session_state.show_alert_2 = False
    if "show_alert_3" not in st.session_state:
        st.session_state.show_alert_3 = False
    if "show_ai_why" not in st.session_state:
        st.session_state.show_ai_why = False

    # -------- HERO --------
    st.markdown(
        """
        <div class='hero'>
          <div class='hero-title'>Forecasting the Unpredictable</div>
          <div class='hero-subtitle'>Nike Inventory Intelligence System</div>
          <div class='subtle' style='margin-top:8px;'>AI-driven demand forecasting with real-time decision intelligence</div>
          <div class='divider'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------- KPI --------
    st.markdown("<div class='delay-1'>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-label'>Forecast Accuracy</div>
              <div class='kpi-value'>{metrics.r2:.3f}</div>
              <div class='subtle'>R² benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        c = risk_color(current_risk)
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-label'>Stock Risk</div>
              <div class='kpi-value' style='color:{c}'>{current_risk}</div>
              <div class='subtle'>Based on demand vs supply</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-label'>Expected Revenue</div>
              <div class='kpi-value'>{money(expected_revenue)}</div>
              <div class='subtle'>Forecasted period estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        tc = "#36D399" if demand_trend_pct >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-label'>Demand Trend</div>
              <div class='kpi-value' style='color:{tc}'>{pct(demand_trend_pct)}</div>
              <div class='subtle'>Vs recent 8-week actuals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Charts --------
    st.markdown("<div class='section-space delay-2'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Interactive Forecast & Drilldown Charts</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.8, 1.2], gap="large")
    with c1:
        fig = trend_chart(history_w, forecast_w, metrics.mae)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Store comparison chart respecting selected Dept/date filters
        bar_src = build_filtered(df_raw, "All Stores", dept_sel, date_start, date_end)
        bar_df = bar_src.groupby("Store", as_index=False)["Weekly_Sales"].sum().sort_values("Weekly_Sales", ascending=False).head(12)
        if len(bar_df) > 0:
            bar_df["StoreLabel"] = "Store " + bar_df["Store"].astype(str)
            selected_label = f"Store {store_sel}" if store_sel != "All Stores" else None
            bar_df["Color"] = np.where(bar_df["StoreLabel"] == selected_label, "#36D399", "rgba(255,255,255,0.62)")
            fig_bar = go.Figure(
                go.Bar(x=bar_df["StoreLabel"], y=bar_df["Weekly_Sales"], marker_color=bar_df["Color"], hovertemplate="%{x}<br>Sales=%{y:,.0f}<extra></extra>")
            )
            fig_bar.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Store", yaxis_title="Sales")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap (month x department)
        heat_src = build_filtered(df_raw, store_sel, "All Departments", date_start, date_end).copy()
        if len(heat_src) > 0:
            heat_src["Month"] = heat_src["Date"].dt.to_period("M").astype(str)
            pivot = heat_src.pivot_table(index="Dept", columns="Month", values="Weekly_Sales", aggfunc="sum", fill_value=0)
            fig_heat = px.imshow(pivot, color_continuous_scale="Greens", aspect="auto")
            fig_heat.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), coloraxis_colorbar_title="Sales")
            st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Dynamic AI Insights --------
    st.markdown("<div class='section-space delay-3'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Dynamic AI Insights</div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3, gap="large")

    holiday_avg = float(filtered_df[filtered_df["IsHoliday"] == 1]["Weekly_Sales"].mean()) if len(filtered_df) > 0 else 0
    non_holiday_avg = float(filtered_df[filtered_df["IsHoliday"] == 0]["Weekly_Sales"].mean()) if len(filtered_df) > 0 else 0
    holiday_lift = ((holiday_avg / non_holiday_avg - 1) * 100.0) if non_holiday_avg > 0 else 0.0

    stockout_two_weeks = float(forecast_w["Predicted"].head(2).sum()) > (base_supply / 4.0)

    with i1:
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>📌 Demand Signal</div>
              <div style='font-weight:850; margin-top:7px;'>Demand in current selection is {pct(demand_trend_pct)}</div>
              <div class='subtle' style='margin-top:8px;'>Selected scope: <b>{store_sel}</b> • <b>{dept_sel}</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>📌 Seasonality</div>
              <div style='font-weight:850; margin-top:7px;'>Holiday lift: {holiday_lift:+.1f}%</div>
              <div class='subtle' style='margin-top:8px;'>Department shows {'seasonal spike' if holiday_lift > 8 else 'stable seasonality'}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with i3:
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>📌 Near-Term Risk</div>
              <div style='font-weight:850; margin-top:7px;'>Stockout risk next 2 weeks: {'Yes' if stockout_two_weeks else 'No'}</div>
              <div class='subtle' style='margin-top:8px;'>Based on short-horizon forecast vs modeled coverage.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- What-if Simulation --------
    st.markdown("<div class='section-space delay-3'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧪 What-if Simulation Panel</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>Updated Forecast (8 weeks)</div>
              <div class='kpi-value'>{money(sim_demand)}</div>
              <div class='subtle'>Scenario: {sim_season} • Promo: {'On' if sim_promo else 'Off'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        rc = risk_color(sim_risk)
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>Risk Level</div>
              <div class='kpi-value' style='color:{rc}'>{sim_risk}</div>
              <div class='subtle'>Using simulated demand and inventory increase.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        sim_rec = "Increase inventory and rebalance stores" if sim_risk == "High" else ("Tight monitoring and selective transfer" if sim_risk == "Medium" else "Maintain plan with lightweight optimization")
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-label'>Updated Recommendation</div>
              <div style='font-weight:850; margin-top:8px'>{sim_rec}</div>
              <div class='subtle' style='margin-top:8px'>Demand Δ: {demand_increase_pct:+d}% • Inventory Δ: {inventory_increase_pct:+d}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Smart Alerts with View Details --------
    st.markdown("<div class='section-space delay-4'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚠️ Smart Alert System</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3, gap="large")
    alert_high_demand = demand_trend_pct >= 15
    alert_overstock = base_supply > base_forecast_sum * 1.30
    alert_low_stock = base_supply < base_forecast_sum * 0.90

    with a1:
        tone = "#EF4444" if alert_high_demand else "#36D399"
        st.markdown(f"<div class='card'><div class='chip' style='background:{tone}; color:#0b0b0b;'>🚀 HIGH DEMAND</div><div style='margin-top:8px; font-weight:800;'>{'Triggered' if alert_high_demand else 'Stable'}</div><div class='subtle' style='margin-top:6px;'>Demand acceleration alert.</div></div>", unsafe_allow_html=True)
        if st.button("View Details", key="alert_hd"):
            st.session_state.show_alert_1 = not st.session_state.show_alert_1
        if st.session_state.show_alert_1:
            st.info(f"Why: Trend is {pct(demand_trend_pct)}. Action: prioritize fast replenishment for selected scope.")

    with a2:
        tone = "#EF4444" if alert_overstock else "#36D399"
        st.markdown(f"<div class='card'><div class='chip' style='background:{tone}; color:#0b0b0b;'>📦 OVERSTOCK</div><div style='margin-top:8px; font-weight:800;'>{'Triggered' if alert_overstock else 'Balanced'}</div><div class='subtle' style='margin-top:6px;'>Excess inventory risk.</div></div>", unsafe_allow_html=True)
        if st.button("View Details", key="alert_os"):
            st.session_state.show_alert_2 = not st.session_state.show_alert_2
        if st.session_state.show_alert_2:
            st.info(f"Why: Supply is {base_supply/base_forecast_sum:.2f}x of forecast. Action: rebalance and reduce inbound allocations.")

    with a3:
        tone = "#EF4444" if alert_low_stock else "#36D399"
        st.markdown(f"<div class='card'><div class='chip' style='background:{tone}; color:#0b0b0b;'>⚠️ LOW STOCK</div><div style='margin-top:8px; font-weight:800;'>{'Triggered' if alert_low_stock else 'Healthy'}</div><div class='subtle' style='margin-top:6px;'>Coverage warning alert.</div></div>", unsafe_allow_html=True)
        if st.button("View Details", key="alert_ls"):
            st.session_state.show_alert_3 = not st.session_state.show_alert_3
        if st.session_state.show_alert_3:
            st.info(f"Why: Supply coverage is {base_supply/base_forecast_sum:.2f}x. Action: increase inventory by 15-25% for high-priority nodes.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Recommendation Engine Panel --------
    st.markdown("<div class='section-space delay-4'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Recommendation Engine</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Optimize Inventory", use_container_width=True):
            st.session_state.rec_action = "Optimize Inventory"
    with r2:
        if st.button("Balance Across Stores", use_container_width=True):
            st.session_state.rec_action = "Balance Across Stores"
    with r3:
        if st.button("Reduce Overstock", use_container_width=True):
            st.session_state.rec_action = "Reduce Overstock"

    rec_text = {
        "Optimize Inventory": "Increase inventory by 18% for high-velocity nodes and maintain safety stock near forecast uncertainty band.",
        "Balance Across Stores": "Shift 10-15% stock from low-demand stores to top-demand stores based on department-level demand intensity.",
        "Reduce Overstock": "Reduce inbound allocation by 20% for low-turn segments and trigger tactical markdown strategy.",
    }[st.session_state.rec_action]

    st.markdown(
        f"""
        <div class='card recommend-border'>
          <div class='kpi-label'>Selected Action</div>
          <div style='font-size:1.1rem; font-weight:900; margin-top:7px;'>{st.session_state.rec_action}</div>
          <div class='subtle' style='margin-top:7px'>{rec_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Decision Summary --------
    suggested_inventory_change = 20 if sim_risk == "High" else (10 if sim_risk == "Medium" else 3)
    expected_impact_pct = (max(0.0, demand_trend_pct) * 0.35) + (8 if st.session_state.rec_action == "Balance Across Stores" else 5)
    summary = f"""
    <div class='section-space'>
      <div class='section-title'>💡 Decision Summary Panel</div>
      <div class='card'>
        <div style='display:flex; gap:18px; flex-wrap:wrap;'>
          <div><div class='kpi-label'>Suggested Inventory Change</div><div style='font-size:1.7rem; font-weight:900;'>+{suggested_inventory_change}%</div></div>
          <div><div class='kpi-label'>Risk Level</div><div style='font-size:1.7rem; font-weight:900; color:{risk_color(sim_risk)}'>{sim_risk}</div></div>
          <div><div class='kpi-label'>Expected Business Impact</div><div style='font-size:1.7rem; font-weight:900;'>+{expected_impact_pct:.1f}%</div></div>
        </div>
        <div class='subtle' style='margin-top:10px;'>Decision aligns forecast, stock risk, and action choice into one executive view.</div>
      </div>
    </div>
    """
    st.markdown(summary, unsafe_allow_html=True)

    # -------- Explain AI --------
    c_explain1, c_explain2 = st.columns([1, 2], gap="large")
    with c_explain1:
        if st.button("👉 Why this prediction?"):
            st.session_state.show_ai_why = not st.session_state.show_ai_why
        st.markdown(
            "<div class='subtle'>Click to inspect feature influence, seasonality impact, and trend explanation.</div>",
            unsafe_allow_html=True,
        )
    with c_explain2:
        if st.session_state.show_ai_why:
            top_inf = influence_df.head(8).copy()
            fig_imp = px.bar(top_inf.sort_values("Influence"), x="Influence", y="Feature", orientation="h", color="Influence", color_continuous_scale="Greens")
            fig_imp.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False, xaxis_title="Influence score (proxy)")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown(
                f"""
                <div class='card'>
                  <div class='kpi-label'>AI Explanation</div>
                  <div class='subtle' style='margin-top:8px;'>
                    Forecast is driven mostly by lag features, rolling demand baseline, and calendar seasonality.
                    Current scenario mode (<b>{scenario_mode}</b>) and promotion settings modify the baseline projection in real time.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider'></div><div class='subtle' style='text-align:center; font-weight:800;'>Built for Hackathon | Team XYZ</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Nike Inventory Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# UI Theme (Nike-inspired minimal)
# -----------------------------
_THEME_CSS = """
<style>
  html, body, [class*="stApp"] {
    background-color: #0e0e0e;
    color: #ffffff;
  }
  .subtle {
    color: rgba(255,255,255,0.68);
    font-size: 0.96rem;
  }
  .section-head {
    font-size: 1.05rem;
    font-weight: 800;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.01em;
  }
  .card {
    background: #1a1a1a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow:
      0 10px 30px rgba(0,0,0,0.35),
      0 0 0 1px rgba(255,255,255,0.03) inset;
  }
  .card-soft {
    background: rgba(26,26,26,0.70);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
  }
  .kpi-value {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.05;
  }
  .kpi-label {
    color: rgba(255,255,255,0.68);
    font-size: 0.92rem;
    font-weight: 750;
    letter-spacing: -0.01em;
  }
  .chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.86rem;
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(6px);
  }
  .divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.12), rgba(255,255,255,0.0));
    margin: 14px auto;
  }
  .hero {
    text-align: center;
    padding: 22px 0 12px 0;
  }
  .gradient-text {
    background: linear-gradient(90deg, #ffffff 0%, #cfcfcf 45%, #8a8a8a 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .hero-subtitle {
    font-size: 1.05rem;
    font-weight: 850;
    letter-spacing: -0.02em;
  }
  .fade-in {
    animation: fadeInUp 0.65s ease both;
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .glow {
    box-shadow:
      0 12px 40px rgba(0,0,0,0.45),
      0 0 0 1px rgba(255,255,255,0.03) inset,
      0 0 18px rgba(54,211,153,0.10);
  }
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)


DATA_PATH = os.path.join(os.path.dirname(__file__), "train.csv")


# -----------------------------
# Data loading & feature building
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at `{path}`. Expected `train.csv` in the project root."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    # Dataset uses strings TRUE/FALSE; normalize to int
    df["IsHoliday"] = df["IsHoliday"].astype(str).str.upper().eq("TRUE").astype(int)
    return df


@st.cache_data(show_spinner=True)
def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a global tabular forecasting dataset with lag + rolling stats.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()

    g = df.groupby(["Store", "Dept"], sort=False)["Weekly_Sales"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)
    # rolling stats over the previous 4 weeks (shift(1) avoids leakage)
    df["rolling_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["rolling_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())

    df["month"] = df["Date"].dt.month
    # week of year can be fractional in some calendars; enforce int for stability
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year
    df["dow"] = df["Date"].dt.dayofweek

    feature_cols = [
        "Store",
        "Dept",
        "IsHoliday",
        "month",
        "weekofyear",
        "year",
        "dow",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_4",
        "rolling_std_4",
    ]
    df_feat = df.dropna(subset=feature_cols + ["Weekly_Sales"])[["Date"] + feature_cols + ["Weekly_Sales"]]

    # Replace any remaining numerical NaNs with safe defaults
    df_feat = df_feat.fillna(0)
    return df_feat


@dataclass(frozen=True)
class ModelMetrics:
    r2: float
    mae: float


@st.cache_resource(show_spinner=True)
def train_forecasting_model(df_feat: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, ModelMetrics]:
    """
    Train a global forecaster across all Store/Dept series.
    We keep evaluation simple: last ~20% of dates as holdout.
    """
    df_feat = df_feat.sort_values("Date")
    cutoff = df_feat["Date"].quantile(0.8)

    train_df = df_feat[df_feat["Date"] < cutoff]
    val_df = df_feat[df_feat["Date"] >= cutoff]

    feature_cols = [
        "Store",
        "Dept",
        "IsHoliday",
        "month",
        "weekofyear",
        "year",
        "dow",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_4",
        "rolling_std_4",
    ]

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=250,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["Weekly_Sales"])

    pred = model.predict(val_df[feature_cols])
    r2 = float(r2_score(val_df["Weekly_Sales"], pred))
    mae = float(mean_absolute_error(val_df["Weekly_Sales"], pred))

    return model, ModelMetrics(r2=r2, mae=mae)


def forecast_next_weeks(
    model: HistGradientBoostingRegressor,
    history: pd.DataFrame,
    store: int,
    dept: int,
    horizon_weeks: int,
    holiday_toggle: bool,
) -> pd.DataFrame:
    """
    Recursive multi-step forecast:
    - uses last 3 lags
    - uses rolling mean/std of previous 4 weeks
    """
    history = history.sort_values("Date").copy()
    if len(history) < 8:
        raise ValueError("Not enough historical data for the selected Store/Dept to forecast.")

    history_sales = history["Weekly_Sales"].astype(float).tolist()
    last_date = history["Date"].max()

    def make_features(d: pd.Timestamp, sales_ext: list[float], is_holiday: int) -> dict:
        # sales_ext includes both actual history and previous predictions
        lag_1 = sales_ext[-1]
        lag_2 = sales_ext[-2]
        lag_3 = sales_ext[-3]
        recent_4 = sales_ext[-4:]
        rolling_mean_4 = float(np.mean(recent_4))
        rolling_std_4 = float(np.std(recent_4, ddof=0))

        return {
            "Store": store,
            "Dept": dept,
            "IsHoliday": is_holiday,
            "month": int(d.month),
            "weekofyear": int(d.isocalendar().week),
            "year": int(d.year),
            "dow": int(d.dayofweek),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4,
        }

    future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, horizon_weeks + 1)]
    is_holiday_value = 1 if holiday_toggle else 0

    preds: list[float] = []
    for d in future_dates:
        feats = make_features(d, history_sales + preds, is_holiday_value)
        X = pd.DataFrame([feats])
        yhat = float(model.predict(X)[0])
        yhat = max(0.0, yhat)  # guard against negative sales
        preds.append(yhat)

    out = pd.DataFrame({"Date": future_dates, "Predicted_Weekly_Sales": preds})
    return out


def month_string(period: pd.Period) -> str:
    return str(period)


def safe_pct_change(new: float, old: float) -> float:
    if old == 0 or np.isclose(old, 0.0):
        return 0.0
    return (new - old) / old * 100.0


def stock_status(demand: float, supply: float) -> tuple[str, str]:
    """
    Returns (label, color_hex).
    """
    if supply >= demand * 1.10:
        return "Good coverage", "#36D399"  # green
    if supply >= demand * 0.90:
        return "Watch (near balance)", "#FBBF24"  # yellow
    return "Low stock risk", "#EF4444"  # red


def confidence_level(global_mae: float, series_mean: float) -> float:
    """
    Confidence heuristic:
    - higher MAE relative to average sales => lower confidence
    """
    if series_mean <= 0:
        return 0.0
    rel = global_mae / series_mean
    # map to 0..1 with gentle clipping
    return float(max(0.05, min(0.95, 1.0 / (1.0 + rel))))


def compute_series_insights(df: pd.DataFrame, store: int, dept: int) -> dict:
    """
    Create executive-style insight cards using simple, explainable aggregations.
    """
    series = df[(df["Store"] == store) & (df["Dept"] == dept)].copy()
    if series.empty:
        return {}

    holiday_sales = series.loc[series["IsHoliday"] == 1, "Weekly_Sales"].mean()
    non_holiday_sales = series.loc[series["IsHoliday"] == 0, "Weekly_Sales"].mean()
    holiday_ratio = (holiday_sales / non_holiday_sales) if (non_holiday_sales and not np.isclose(non_holiday_sales, 0.0)) else np.nan

    store_avg = (
        df[df["Dept"] == dept]
        .groupby("Store")["Weekly_Sales"]
        .mean()
        .sort_values(ascending=False)
    )
    overall_avg_dept = df[df["Dept"] == dept]["Weekly_Sales"].mean()
    selected_store_mean = store_avg.loc[store] if store in store_avg.index else np.nan
    store_uplift = selected_store_mean / overall_avg_dept if overall_avg_dept else np.nan

    dept_avg = df.groupby("Dept")["Weekly_Sales"].mean()
    overall_avg_all = df["Weekly_Sales"].mean()
    selected_dept_mean = dept_avg.loc[dept] if dept in dept_avg.index else np.nan
    dept_uplift = selected_dept_mean / overall_avg_all if overall_avg_all else np.nan

    insights: dict[str, dict] = {}
    if pd.notna(holiday_ratio) and holiday_ratio >= 1.2:
        insights["holiday_spike"] = {
            "title": "Demand spikes during holidays",
            "detail": f"Holiday weeks average {holiday_ratio:.1f}x higher sales than non-holiday weeks for Store {store}, Dept {dept}.",
        }
    else:
        insights["holiday_spike"] = {
            "title": "Holiday demand is stable",
            "detail": f"Holiday weeks are only {holiday_ratio:.1f}x vs non-holiday for Store {store}, Dept {dept}.",
        }

    if pd.notna(store_uplift) and store_uplift >= 1.10:
        insights["store_strength"] = {
            "title": f"Store {store} consistently outperforms",
            "detail": f"This store's average demand is {store_uplift:.1f}x the overall dept baseline.",
        }
    else:
        insights["store_strength"] = {
            "title": f"Store {store} runs near baseline",
            "detail": "No major store-level outperformance detected for this product.",
        }

    if pd.notna(dept_uplift) and dept_uplift <= 0.92:
        insights["dept_underperforming"] = {
            "title": "Product category underperforming",
            "detail": f"Dept {dept} averages {dept_uplift:.2f}x of overall category demand - optimize allocation and promotions.",
        }
    else:
        insights["dept_underperforming"] = {
            "title": "Product category performing normally",
            "detail": "Dept demand is within the typical range across the Nike dataset.",
        }

    return insights


def plot_trend(history: pd.DataFrame, forecast: pd.DataFrame, mae: float) -> go.Figure:
    """
    Past + forecast with a heuristic confidence region.
    """
    history = history.sort_values("Date")
    forecast = forecast.sort_values("Date")

    x_past = history["Date"]
    y_past = history["Weekly_Sales"]

    x_f = forecast["Date"]
    y_f = forecast["Predicted_Weekly_Sales"].astype(float)

    # Confidence band heuristic:
    # - start near MAE magnitude
    # - widen slightly with each week ahead
    steps = np.arange(1, len(y_f) + 1, dtype=float)
    band = float(mae) * 1.25 * np.sqrt(steps / max(1.0, steps[-1]))
    y_upper = (y_f + band).astype(float)
    y_lower = np.maximum(0.0, (y_f - band)).astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_past,
            y=y_past,
            mode="lines",
            name="Past sales",
            line=dict(color="rgba(255,255,255,0.55)", width=2),
            hovertemplate="Past<br>%{x|%Y-%m-%d}<br>Sales=%{y:.2f}<extra></extra>",
        )
    )

    # Confidence region (forecast only)
    fig.add_trace(
        go.Scatter(
            x=x_f.tolist() + x_f[::-1].tolist(),
            y=np.concatenate([y_upper.values, y_lower.values[::-1]]),
            fill="toself",
            name="Confidence region",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(255,255,255,0.10)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_f,
            y=y_f,
            mode="lines",
            name="Forecast",
            line=dict(color="#ffffff", width=3, shape="spline"),
            hovertemplate="Forecast<br>%{x|%Y-%m-%d}<br>Sales=%{y:.2f}<extra></extra>",
        )
    )

    # Highlight final forecast point
    if len(x_f) > 0:
        fig.add_trace(
            go.Scatter(
                x=[x_f.iloc[-1]],
                y=[y_f.iloc[-1]],
                mode="markers",
                name="Latest forecast",
                marker=dict(size=12, color="#36D399", line=dict(color="#0e0e0e", width=2)),
                hovertemplate="Latest forecast<br>%{x|%Y-%m-%d}<br>Sales=%{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Week"),
        yaxis=dict(title="Sales (Weekly_Sales)"),
        transition_duration=300,
    )
    return fig


def plot_monthly_heatmap(months: list[str], store_labels: list[str], values: np.ndarray) -> go.Figure:
    """
    Heatmap of monthly sales (rows=stores, cols=months).
    """
    fig = go.Figure(
        data=go.Heatmap(
            x=months,
            y=store_labels,
            z=values,
            colorscale="Greens",
            colorbar=dict(title="Sales", tickformat=",.0f"),
            hovertemplate="Store=%{y}<br>Month=%{x}<br>Sales=%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        transition_duration=300,
        xaxis=dict(title="Month"),
        yaxis=dict(title="Store"),
    )
    return fig


def plot_store_comparison(store_labels: list[str], predicted_values: list[float], selected_store: str) -> go.Figure:
    """
    Bar chart comparing predicted monthly demand across stores.
    """
    colors = ["#36D399" if lbl == selected_store else "rgba(255,255,255,0.55)" for lbl in store_labels]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=store_labels,
            y=predicted_values,
            marker_color=colors,
            hovertemplate="Store=%{x}<br>Predicted monthly demand=%{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        transition_duration=300,
        xaxis=dict(title="Store"),
        yaxis=dict(title="Predicted monthly demand"),
        bargap=0.25,
    )
    return fig


def plot_inventory_distribution(online_supply: float, retail_supply: float, online_demand: float, retail_demand: float) -> go.Figure:
    labels = ["Online (DTC)", "Retail Store"]
    supply = [online_supply, retail_supply]
    demand = [online_demand, retail_demand]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Supply (Inventory)", x=labels, y=supply, marker_color="rgba(255,255,255,0.95)"))
    fig.add_trace(go.Bar(name="Demand (Forecast)", x=labels, y=demand, marker_color="rgba(54,211,153,0.85)"))
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Units (modeled)"),
    )
    return fig


def plot_mismatch(online_supply: float, retail_supply: float, online_demand: float, retail_demand: float) -> go.Figure:
    labels = ["Online (DTC)", "Retail Store"]
    mismatch = [online_supply - online_demand, retail_supply - retail_demand]  # + means excess

    colors = ["#36D399" if v >= 0 else "#EF4444" for v in mismatch]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=mismatch,
            marker_color=colors,
            name="Supply - Demand",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Excess inventory (+) / Shortage (-)"),
    )
    return fig


def format_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def format_int(x: float) -> str:
    try:
        return f"{int(round(x)):,.0f}"
    except Exception:
        return str(x)


def main():
    df_raw = load_raw_dataset(DATA_PATH)
    df_feat = build_training_frame(df_raw)
    with st.spinner("Training AI forecasting engine..."):
        model, metrics = train_forecasting_model(df_feat)

    stores = sorted(df_raw["Store"].unique().tolist())
    depts = sorted(df_raw["Dept"].unique().tolist())

    st.sidebar.header("🎛️ Decision Controls")
    store = st.sidebar.selectbox("Store selection", stores, index=0)
    dept = st.sidebar.selectbox("Product selection (Dept)", depts, index=0)
    horizon_weeks = st.sidebar.slider("Forecast window (weeks)", min_value=4, max_value=20, value=12, step=2)
    holiday_toggle = st.sidebar.toggle("Holiday toggle (forecast)", value=True)

    # Pull the selected series from raw dataset
    history = df_raw[(df_raw["Store"] == store) & (df_raw["Dept"] == dept)].copy().sort_values("Date")

    # Month selector is based on the selected series' own forecast horizon
    history_last_date = history["Date"].max() if not history.empty else df_raw["Date"].max()
    future_dates_for_menu = [history_last_date + pd.Timedelta(weeks=i) for i in range(1, horizon_weeks + 1)]
    future_months = sorted({month_string(d.to_period("M")) for d in future_dates_for_menu})
    target_month = st.sidebar.selectbox("Month (forecast)", future_months, index=0) if future_months else None

    st.markdown(
        """
        <div class="hero fade-in">
          <div style="font-size:46px; font-weight:950; line-height:1.05; margin-bottom:10px;" class="gradient-text">
            Forecasting the Unpredictable
          </div>
          <div class="hero-subtitle" style="margin-bottom:10px;">Nike Inventory Intelligence System</div>
          <div class="subtle" style="max-width:920px; margin:0 auto;">
            AI-powered demand forecasting, smart inventory alignment, and real-time decision intelligence
          </div>
          <div class="divider" style="width:86%; margin-top:18px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history.empty or len(history) < 10:
        st.error("No sufficient history for this Store/Dept selection. Try a different combination.")
        return

    # Forecast
    with st.spinner("Running AI forecast (past + recursive future)..."):
        forecast = forecast_next_weeks(
            model=model,
            history=history,
            store=store,
            dept=dept,
            horizon_weeks=horizon_weeks,
            holiday_toggle=holiday_toggle,
        )
    forecast["Month"] = forecast["Date"].dt.to_period("M").astype(str)

    if target_month is None:
        st.warning("No forecast months available for this horizon.")
        return

    target_forecast = forecast[forecast["Month"] == target_month].copy()
    demand_forecast = float(target_forecast["Predicted_Weekly_Sales"].sum())

    # Previous month actual for demand change %
    target_period = pd.Period(target_month)
    prev_period = target_period - 1
    prev_month_actual = float(
        history[history["Date"].dt.to_period("M") == prev_period]["Weekly_Sales"].sum()
    )
    demand_change_pct = safe_pct_change(demand_forecast, prev_month_actual)

    # Inventory model (modeled supply)
    weeks_in_target_month = max(1, len(target_forecast))
    recent_avg_weekly = float(history["Weekly_Sales"].tail(8).mean())
    # supply is modeled as "reorder + safety" inventory arriving for that month
    safety_factor = 1.15
    supply_month = float(recent_avg_weekly * safety_factor * weeks_in_target_month)

    status_label, status_color = stock_status(demand=demand_forecast, supply=supply_month)
    stock_status_chip = f"<span class='chip' style='background:{status_color}; color:#0B0B0B;'>{status_label}</span>"

    # -----------------------------
    # AI Insights engine (why)
    # -----------------------------
    # Holiday lift (how demand changes during IsHoliday weeks)
    holiday_mean = float(history.loc[history["IsHoliday"] == 1, "Weekly_Sales"].mean())
    non_holiday_mean = float(history.loc[history["IsHoliday"] == 0, "Weekly_Sales"].mean())
    holiday_lift_pct = (
        (holiday_mean / non_holiday_mean - 1.0) * 100.0
        if non_holiday_mean > 0 and not np.isnan(non_holiday_mean)
        else 0.0
    )

    # Store outperformance (store rank inside selected Dept)
    store_dept_mean = df_raw[df_raw["Dept"] == dept].groupby("Store")["Weekly_Sales"].mean()
    overall_dept_mean = float(df_raw[df_raw["Dept"] == dept]["Weekly_Sales"].mean())
    selected_store_mean_dept = float(store_dept_mean.loc[store]) if store in store_dept_mean.index else overall_dept_mean
    store_outperformance_pct = (
        (selected_store_mean_dept / overall_dept_mean - 1.0) * 100.0
        if overall_dept_mean > 0
        else 0.0
    )
    store_rank = int(store_dept_mean.rank(ascending=False, method="min").loc[store]) if store in store_dept_mean.index else 999

    # Dept trend (declining vs rising demand over recent weeks)
    dept_weekly_avg = df_raw[df_raw["Dept"] == dept].groupby("Date")["Weekly_Sales"].mean().sort_index()
    tail16 = dept_weekly_avg.tail(16)
    if len(tail16) >= 16:
        recent_avg = float(tail16.tail(8).mean())
        prev_avg = float(tail16.head(8).mean())
    else:
        recent_avg = float(tail16.tail(8).mean())
        prev_avg = float(dept_weekly_avg.drop(tail16.index).tail(8).mean()) if len(dept_weekly_avg) > len(tail16) else float(recent_avg)

    dept_trend_pct = ((recent_avg / prev_avg - 1.0) * 100.0) if prev_avg > 0 else 0.0

    # Build bullet-card insights
    insight_cards: list[dict] = []
    if holiday_lift_pct >= 5.0:
        insight_cards.append(
            {
                "icon": "🚀",
                "title": f"Demand increases by {holiday_lift_pct:.0f}% during holiday weeks",
                "bullets": [
                    f"Holiday-week avg: {format_money(holiday_mean)} vs non-holiday: {format_money(non_holiday_mean)}.",
                    f"Implication for {target_month}: expand coverage when IsHoliday is active.",
                ],
            }
        )
    else:
        insight_cards.append(
            {
                "icon": "🧭",
                "title": "Holiday demand is relatively stable",
                "bullets": [
                    f"Holiday lift: {holiday_lift_pct:.0f}% vs non-holiday demand.",
                    "Implication: focus shifts on store and category trend instead of holiday spikes.",
                ],
            }
        )

    if store_outperformance_pct >= 8.0 and store_rank <= 5:
        insight_cards.append(
            {
                "icon": "🏁",
                "title": f"Store {store} consistently outperforms others",
                "bullets": [
                    f"Store avg vs Dept baseline: {store_outperformance_pct:+.1f}%.",
                    f"Ranking inside Dept: #{store_rank} (higher = stronger demand).",
                ],
            }
        )
    else:
        insight_cards.append(
            {
                "icon": "📍",
                "title": f"Store {store} runs near its Dept baseline",
                "bullets": [
                    f"Store avg vs Dept baseline: {store_outperformance_pct:+.1f}%.",
                    "Implication: use targeted rebalancing only if risk alerts trigger.",
                ],
            }
        )

    if dept_trend_pct <= -5.0:
        insight_cards.append(
            {
                "icon": "📉",
                "title": f"Product category (Dept {dept}) shows declining trend",
                "bullets": [
                    f"Recent vs prior avg: {dept_trend_pct:+.1f}%.",
                    "Implication: reduce excess inventory pressure and optimize allocations.",
                ],
            }
        )
    else:
        insight_cards.append(
            {
                "icon": "📈",
                "title": f"Product category (Dept {dept}) demand is stabilizing",
                "bullets": [
                    f"Recent vs prior avg: {dept_trend_pct:+.1f}%.",
                    "Implication: maintain planned stock but watch alerts closely.",
                ],
            }
        )

    # -----------------------------
    # Smart Alerts (risk)
    # -----------------------------
    low_stock = supply_month < demand_forecast * 0.90
    overstock = supply_month > demand_forecast * 1.35
    demand_surge = False
    if prev_month_actual > 0:
        demand_surge = demand_change_pct >= 15.0
    else:
        # fallback: compare against recent mean
        demand_surge = demand_forecast >= recent_avg_weekly * weeks_in_target_month * 1.20

    alert_cards: list[dict] = []
    # Low stock risk
    if low_stock:
        alert_cards.append(
            {
                "tone": "red",
                "icon": "⚠️",
                "title": "Low stock risk",
                "detail": f"Modeled supply ({format_money(supply_month)}) is below 90% of forecast demand ({format_money(demand_forecast)}).",
            }
        )
    elif supply_month < demand_forecast * 1.00:
        alert_cards.append(
            {
                "tone": "yellow",
                "icon": "🟡",
                "title": "Low stock watch",
                "detail": f"Supply is close to forecast. Tight coverage for {target_month}.",
            }
        )
    else:
        alert_cards.append(
            {
                "tone": "green",
                "icon": "✅",
                "title": "Coverage looks healthy",
                "detail": f"Supply covers demand with room for demand variability.",
            }
        )

    # Overstock risk
    if overstock:
        alert_cards.append(
            {
                "tone": "red",
                "icon": "🚫",
                "title": "Overstock risk",
                "detail": f"Supply exceeds forecast by more than 35% for Dept {dept}.",
            }
        )
    elif supply_month > demand_forecast * 1.15:
        alert_cards.append(
            {
                "tone": "yellow",
                "icon": "⚠️",
                "title": "Overstock watch",
                "detail": "Consider tighter replenishment or faster sell-through actions.",
            }
        )
    else:
        alert_cards.append(
            {
                "tone": "green",
                "icon": "🧊",
                "title": "Inventory levels balanced",
                "detail": "Modeled supply is within a safe coverage band.",
            }
        )

    # Demand surge
    if demand_surge:
        alert_cards.append(
            {
                "tone": "red",
                "icon": "🚀",
                "title": "Demand surge detected",
                "detail": f"Forecast demand is up {demand_change_pct:+.1f}% vs prior month — prioritize allocation.",
            }
        )
    else:
        alert_cards.append(
            {
                "tone": "green",
                "icon": "🌿",
                "title": "No major surge detected",
                "detail": "Demand movement looks consistent with the baseline trend.",
            }
        )

    # -----------------------------
    # Action Recommendations (what to do)
    # -----------------------------
    actions: list[dict] = []
    if low_stock:
        actions.append(
            {
                "tone": "green",
                "icon": "📦",
                "title": f"Increase inventory for Store {store} by 20%",
                "detail": "Target the selected month with incremental replenishment to prevent lost sales.",
            }
        )
    else:
        actions.append(
            {
                "tone": "yellow",
                "icon": "🛡️",
                "title": f"Maintain coverage for Store {store}",
                "detail": "Supply is within a manageable band. Keep safety stock steady and watch for category drift.",
            }
        )

    if overstock:
        actions.append(
            {
                "tone": "yellow",
                "icon": "🧹",
                "title": f"Reduce stock for Dept {dept} by 25%",
                "detail": "Rebalance allocation to reduce markdown/write-down risk while demand remains below peak.",
            }
        )
    else:
        actions.append(
            {
                "tone": "green",
                "icon": "📈",
                "title": f"Protect sell-through for Dept {dept}",
                "detail": "Avoid over-corrections. Maintain planned replenishment with alert-driven adjustments.",
            }
        )

    # Shift inventory between stores (use historical store propensity within selected Dept)
    if len(store_dept_mean) >= 2:
        sorted_stores = store_dept_mean.sort_values(ascending=False)
        best_store = int(sorted_stores.index[0])
        worst_store = int(sorted_stores.index[-1])
        if best_store == store and len(sorted_stores) >= 2:
            best_store = int(sorted_stores.index[1])
        if worst_store == store and len(sorted_stores) >= 2:
            worst_store = int(sorted_stores.index[-2])

        best_mean = float(store_dept_mean.loc[best_store])
        worst_mean = float(store_dept_mean.loc[worst_store]) if worst_store in store_dept_mean.index else max(1.0, best_mean * 0.8)
        demand_spread_ratio = (best_mean / worst_mean) if worst_mean > 0 else 1.0

        if best_store != worst_store and demand_spread_ratio >= 1.05:
            shift_pct = 15 if (demand_surge or low_stock) else 10
            actions.append(
                {
                    "tone": "green" if shift_pct >= 15 else "yellow",
                    "icon": "🔁",
                    "title": f"Shift {shift_pct}% of inventory from Store {worst_store} to Store {best_store}",
                    "detail": "Move units toward stores with consistently higher demand propensity for this Dept.",
                }
            )

    if len(actions) < 3:
        actions.append(
            {
                "tone": "yellow",
                "icon": "🎯",
                "title": "Keep allocation stable (alert-driven adjustments)",
                "detail": "Forecast and modeled supply are aligned. Only intervene if alerts turn red.",
            }
        )

    # Inventory Alignment (multi-channel simulation)
    # Modeled online share: dependent on dept (proxy for digital propensity)
    max_dept = max(depts) if depts else 1
    dept_norm = (dept / max_dept) if max_dept else 0.5
    online_demand_share = float(np.clip(0.30 + 0.10 * dept_norm, 0.22, 0.48))
    retail_demand_share = 1.0 - online_demand_share

    # Initial inventory split (what we have today)
    online_supply_share = 0.35
    retail_supply_share = 1.0 - online_supply_share

    online_demand = demand_forecast * online_demand_share
    retail_demand = demand_forecast * retail_demand_share
    online_supply = supply_month * online_supply_share
    retail_supply = supply_month * retail_supply_share

    # Rebalancing suggestion between channels
    online_shortage = max(0.0, online_demand - online_supply)
    retail_surplus = max(0.0, retail_supply - retail_demand)
    transfer = min(online_shortage, retail_surplus)
    rebalancing_needed = transfer > 0.0

    # Compute "after" metrics for Customer Impact
    supply_total_adj = supply_month
    online_supply_after = online_supply
    retail_supply_after = retail_supply
    if rebalancing_needed:
        online_supply_after += transfer
        retail_supply_after -= transfer

    # Preserve the allocation coming out of the channel rebalancing step
    # (so scaling total supply later doesn't erase the transfer effect).
    online_share_after_rebalance = (online_supply_after / supply_month) if supply_month > 0 else online_supply_share

    # Adjust total supply based on recommendations (simplified)
    if low_stock:
        supply_total_adj = supply_month * 1.20
        online_supply_after = supply_total_adj * float(online_share_after_rebalance)
        retail_supply_after = supply_total_adj - online_supply_after
    elif overstock:
        supply_total_adj = supply_month * 0.75
        online_supply_after = supply_total_adj * float(online_share_after_rebalance)
        retail_supply_after = supply_total_adj - online_supply_after

    baseline_availability = float(np.clip(supply_month / demand_forecast, 0.0, 1.0)) if demand_forecast > 0 else 0.0
    new_availability = float(np.clip(supply_total_adj / demand_forecast, 0.0, 1.0)) if demand_forecast > 0 else 0.0
    out_of_stock_risk_baseline = 1.0 - baseline_availability
    out_of_stock_risk_new = 1.0 - new_availability
    out_of_stock_reduction_pct = (
        (out_of_stock_risk_baseline - out_of_stock_risk_new) / out_of_stock_risk_baseline * 100.0
        if out_of_stock_risk_baseline > 0
        else 0.0
    )

    # Faster delivery potential: approximate as how well online supply covers online demand
    online_fill_baseline = float(np.clip(online_supply / online_demand, 0.0, 1.0)) if online_demand > 0 else 0.0
    online_fill_after = float(np.clip(online_supply_after / online_demand, 0.0, 1.0)) if online_demand > 0 else 0.0
    faster_delivery_potential_pct = float((online_fill_after - online_fill_baseline) * 100.0)
    faster_delivery_potential_pct = max(-50.0, min(40.0, faster_delivery_potential_pct))

    customer_impact = {
        "stock_availability_improved_pct": max(-100.0, min(100.0, (new_availability - baseline_availability) * 100.0)),
        "reduced_out_of_stock_risk_pct": max(0.0, out_of_stock_reduction_pct),
        "faster_delivery_potential_pct": faster_delivery_potential_pct,
    }

    # -----------------------------
    # Premium executive visuals data
    # -----------------------------
    # Map stores to a relative "demand index" for this Dept using historical averages.
    # This lets us compare stores month-over-month without expensive per-store recursive forecasting.
    selected_store_mean_dept = float(store_dept_mean.loc[store]) if store in store_dept_mean.index else float(store_dept_mean.mean())
    if selected_store_mean_dept <= 0:
        selected_store_mean_dept = 1.0
    store_demand_scale = store_dept_mean / selected_store_mean_dept  # selected store => 1.0

    # KPI indicator colors:
    # - Predicted demand "high" if above prior month baseline by >=10%
    # - "low" if <=95% of baseline
    predicted_rel = demand_forecast / prev_month_actual if prev_month_actual > 0 else 1.0
    predicted_color = "#36D399" if predicted_rel >= 1.10 else ("#FBBF24" if predicted_rel >= 0.95 else "#EF4444")
    trend_color = "#36D399" if demand_change_pct >= 10 else ("#FBBF24" if demand_change_pct > -10 else "#EF4444")

    # Monthly heatmap: last 12 months (actual) + target forecast month (predicted) for top stores.
    history_last_month = history["Date"].max().to_period("M")
    past_months = [history_last_month - i for i in range(11, -1, -1)]
    target_period = pd.Period(target_month)
    heatmap_months = [str(p) for p in past_months] + [str(target_period)]

    top_n_stores = 6
    top_store_ids = store_demand_scale.sort_values(ascending=False).head(top_n_stores).index.tolist()
    if store not in top_store_ids and len(store_demand_scale) > 0:
        # Ensure selected store is visible even if outside the top-N.
        top_store_ids[-1] = store

    # Precompute monthly actual totals for heatmap stores.
    df_heat = df_raw[(df_raw["Dept"] == dept) & (df_raw["Store"].isin(top_store_ids))].copy()
    df_heat["month"] = df_heat["Date"].dt.to_period("M")
    monthly_actual = df_heat.groupby(["Store", "month"])["Weekly_Sales"].sum()

    # Build matrix: rows=stores, cols=months
    heatmap_values = np.zeros((len(top_store_ids), len(heatmap_months)), dtype=float)
    for i, s_id in enumerate(top_store_ids):
        # past months
        for j, m in enumerate(past_months):
            heatmap_values[i, j] = float(monthly_actual.get((s_id, m), 0.0))
        # forecast month scaled from selected store
        heatmap_values[i, len(past_months)] = float(demand_forecast * float(store_demand_scale.loc[s_id]) if s_id in store_demand_scale.index else demand_forecast)

    heatmap_store_labels = [f"Store {s}" for s in top_store_ids]
    heatmap_fig = plot_monthly_heatmap(months=heatmap_months, store_labels=heatmap_store_labels, values=heatmap_values)

    # Store comparison bar chart for target month (predicted monthly demand).
    predicted_by_store = (demand_forecast * store_demand_scale).dropna()
    # Choose top stores but always keep selected store present.
    top_k = 10 if len(predicted_by_store) > 10 else len(predicted_by_store)
    top_store_pred_ids = predicted_by_store.sort_values(ascending=False).head(top_k).index.tolist()
    if store not in top_store_pred_ids:
        top_store_pred_ids[-1] = store
    # Sort for nicer visuals
    top_store_pred_ids = sorted(top_store_pred_ids, key=lambda x: float(predicted_by_store.get(x, 0.0)), reverse=True)

    store_bar_labels = [f"Store {s}" for s in top_store_pred_ids]
    store_bar_vals = [float(predicted_by_store.get(s, 0.0)) for s in top_store_pred_ids]
    selected_store_label = f"Store {store}"
    store_bar_fig = plot_store_comparison(
        store_labels=store_bar_labels,
        predicted_values=store_bar_vals,
        selected_store=selected_store_label,
    )

    # -----------------------------
    # Render UI
    # -----------------------------
    # Section A: Demand Overview
    st.markdown("<div class='section-head'>📊 Demand Overview (What is happening)</div>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3, gap="large")

    with kpi_col1:
        st.markdown(
            f"""
            <div class='card glow fade-in' style='animation-delay:0.05s'>
              <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                <div class='kpi-label'>📈 Predicted Demand</div>
                <span class='chip' style='background:{predicted_color}; color:#0e0e0e;'>
                  {'HIGH' if predicted_rel >= 1.10 else ('MED' if predicted_rel >= 0.95 else 'LOW')}
                </span>
              </div>
              <div class='kpi-value' style='margin-top:6px'>{format_money(demand_forecast)}</div>
              <div class='subtle' style='margin-top:6px'>Forecast for <b>{target_month}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class='card glow fade-in' style='animation-delay:0.15s'>
              <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                <div class='kpi-label'>📊 Demand Trend %</div>
                <span class='chip' style='background:{trend_color}; color:#0e0e0e;'>
                  {'RISING' if demand_change_pct >= 10 else ('STEADY' if demand_change_pct > -10 else 'DECLINING')}
                </span>
              </div>
              <div class='kpi-value' style='margin-top:6px; color:{trend_color}'>{demand_change_pct:+.1f}%</div>
              <div class='subtle' style='margin-top:6px'>Vs prior month baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class='card glow fade-in' style='animation-delay:0.25s'>
              <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                <div class='kpi-label'>⚠️ Stock Status</div>
                <span class='chip' style='background:{status_color}; color:#0e0e0e;'>{status_label}</span>
              </div>
              <div class='subtle' style='margin-top:10px'>
                Supply: <b>{format_money(supply_month)}</b> • Demand: <b>{format_money(demand_forecast)}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    trend_fig = plot_trend(history=history, forecast=forecast, mae=metrics.mae)
    st.plotly_chart(trend_fig, use_container_width=True)

    # Premium executive extras: monthly heatmap + store comparison
    heat_col1, heat_col2 = st.columns([1, 1], gap="large")
    with heat_col1:
        st.markdown("<div class='subtle' style='font-weight:800; margin:6px 0 8px 2px;'>Monthly trend heatmap</div>", unsafe_allow_html=True)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with heat_col2:
        st.markdown("<div class='subtle' style='font-weight:800; margin:6px 0 8px 2px;'>Store comparison (predicted)</div>", unsafe_allow_html=True)
        st.plotly_chart(store_bar_fig, use_container_width=True)

    # Section B: AI Insights
    st.markdown("<div class='section-head'>🧠 AI Insights (Why it is happening)</div>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            card = insight_cards[0]
            st.markdown(
                f"""
                <div class='card glow fade-in' style='animation-delay:0.05s'>
                  <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                    <div class='kpi-label'>Insight</div>
                    <span class='chip' style='background:rgba(255,255,255,0.08); color:#ffffff;'>{card['icon']}</span>
                  </div>
                  <div style='font-weight:900; margin-top:8px; line-height:1.15'>{card['title']}</div>
                  <ul style='margin:10px 0 0 18px; padding:0; color:rgba(255,255,255,0.82);'>
                    {''.join([f"<li>✔ {b}</li>" for b in card['bullets']])}
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            card = insight_cards[1]
            st.markdown(
                f"""
                <div class='card glow fade-in' style='animation-delay:0.12s'>
                  <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                    <div class='kpi-label'>Insight</div>
                    <span class='chip' style='background:rgba(255,255,255,0.08); color:#ffffff;'>{card['icon']}</span>
                  </div>
                  <div style='font-weight:900; margin-top:8px; line-height:1.15'>{card['title']}</div>
                  <ul style='margin:10px 0 0 18px; padding:0; color:rgba(255,255,255,0.82);'>
                    {''.join([f"<li>✔ {b}</li>" for b in card['bullets']])}
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            card = insight_cards[2]
            st.markdown(
                f"""
                <div class='card glow fade-in' style='animation-delay:0.19s'>
                  <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                    <div class='kpi-label'>Insight</div>
                    <span class='chip' style='background:rgba(255,255,255,0.08); color:#ffffff;'>{card['icon']}</span>
                  </div>
                  <div style='font-weight:900; margin-top:8px; line-height:1.15'>{card['title']}</div>
                  <ul style='margin:10px 0 0 18px; padding:0; color:rgba(255,255,255,0.82);'>
                    {''.join([f"<li>✔ {b}</li>" for b in card['bullets']])}
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section C: Smart Alerts
    st.markdown("<div class='section-head'>⚠️ Smart Alerts (Risk Detection)</div>", unsafe_allow_html=True)
    alert_colors = {"red": "#EF4444", "yellow": "#FBBF24", "green": "#36D399"}
    with st.container():
        col1, col2, col3 = st.columns(3, gap="large")
        for idx, col in enumerate([col1, col2, col3]):
            card = alert_cards[idx]
            bg = alert_colors[card["tone"]]
            col.markdown(
                f"""
                <div class='card glow fade-in' style='animation-delay:{0.05 * (idx + 1)}s'>
                  <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                    <div class='kpi-label'>Alert</div>
                    <span class='chip' style='background:{bg}; color:#0e0e0e;'>{card['icon']} {card['tone'].upper()}</span>
                  </div>
                  <div style='font-weight:950; margin-top:8px'>{card['title']}</div>
                  <div class='subtle' style='margin-top:8px; line-height:1.25'>{card['detail']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section D: Action Recommendations (MOST IMPORTANT)
    st.markdown("<div class='section-head'>🎯 Action Recommendations (Decision + Impact)</div>", unsafe_allow_html=True)
    with st.container():
        rec_col1, rec_col2 = st.columns([2, 1])
        with rec_col1:
            tone_colors = {"green": "#36D399", "yellow": "#FBBF24", "red": "#EF4444"}
            for idx, action in enumerate(actions):
                bg = tone_colors.get(action["tone"], "#FBBF24")
                st.markdown(
                    f"""
                    <div class='card glow fade-in' style='animation-delay:{0.06 * (idx + 1)}s'>
                      <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
                        <div class='kpi-label'>Decision</div>
                        <span class='chip' style='background:{bg}; color:#0e0e0e;'>{action['icon']} {action['tone'].upper()}</span>
                      </div>
                      <div style='font-weight:950; margin-top:8px; line-height:1.15'>{action['title']}</div>
                      <div class='subtle' style='margin-top:8px; line-height:1.25'>{action['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with rec_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-label'>Executive decision context</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:8px; font-weight:950'>Forecast → Inventory Plan</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='subtle' style='margin-top:8px'>Target month: <b>{target_month}</b> • Forecast demand: <b>{format_money(demand_forecast)}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='subtle' style='margin-top:6px'>Demand trend: <b style='color:{trend_color}'>{demand_change_pct:+.1f}%</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='subtle' style='margin-top:6px'>Modeled supply: <b>{format_money(supply_month)}</b> • Status: <b>{status_label}</b></div>",
                unsafe_allow_html=True,
            )
            if rebalancing_needed:
                st.markdown("<div style='margin-top:12px; font-weight:950'>Channel optimization</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='subtle'>Estimated transfer to reduce mismatch: <b>{format_int(transfer)}</b> units.</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # 🔐 Trust & Transparency Panel
    # -----------------------------
    st.markdown("<div class='section-head'>🔐 Trust & Transparency Panel</div>", unsafe_allow_html=True)
    with st.container():
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(
                f"""
                <div class='card'>
                  <div class='kpi-label'>Model Accuracy</div>
                  <div class='kpi-value'>{metrics.r2:.3f}</div>
                  <div class='subtle'>R² (holdout across recent dates)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                f"""
                <div class='card'>
                  <div class='kpi-label'>Mean Absolute Error</div>
                  <div class='kpi-value'>{format_money(metrics.mae)}</div>
                  <div class='subtle'>Lower MAE means tighter forecasts.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with t3:
            series_mean = float(history["Weekly_Sales"].tail(8).mean())
            conf = confidence_level(global_mae=metrics.mae, series_mean=series_mean)
            pct = int(round(conf * 100))
            st.markdown(
                f"""
                <div class='card'>
                  <div class='kpi-label'>Confidence Level</div>
                  <div class='kpi-value'>{pct}%</div>
                  <div class='subtle'>Higher when recent demand is consistent with model error.</div>
                  <div style='margin-top:10px'>
                    <div class='subtle'>0% (low) → 100% (high)</div>
                    <div style="background:rgba(255,255,255,0.10); border-radius:999px; height:10px; overflow:hidden; margin-top:6px;">
                      <div style="width:{pct}%; background:linear-gradient(90deg,#36D399,#FFFFFF); height:10px;"></div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='card'>"
        "<div class='kpi-label'>Simple explanation (executive-friendly)</div>"
        "<div style='margin-top:8px' class='subtle'>"
        "Prediction is based on past sales patterns (lags), seasonality (week/month/year), holiday impact, and store/product trends."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Inventory Alignment (Multi-channel Simulation)
    # -----------------------------
    st.markdown("<div class='section-head'>📦 Inventory Alignment (Multi-channel Simulation)</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.plotly_chart(
            plot_inventory_distribution(
                online_supply=online_supply_after,
                retail_supply=retail_supply_after,
                online_demand=online_demand,
                retail_demand=retail_demand,
            ),
            use_container_width=True,
        )
    with s2:
        st.plotly_chart(
            plot_mismatch(
                online_supply=online_supply_after,
                retail_supply=retail_supply_after,
                online_demand=online_demand,
                retail_demand=retail_demand,
            ),
            use_container_width=True,
        )

    # Channel recommendation text
    if rebalancing_needed:
        st.success(
            f"Rebalancing suggestion: shift modeled inventory toward the channel with unmet demand (Online). "
            f"Estimated transfer: {format_int(transfer)} units."
        )
    else:
        st.info("Rebalancing suggestion: current channel split is already close to demand for the selected month.")

    # -----------------------------
    # Business Impact
    # -----------------------------
    baseline_excess_ratio = float(max(0.0, supply_month - demand_forecast) / supply_month) if supply_month > 0 else 0.0
    new_excess_ratio = float(max(0.0, supply_total_adj - demand_forecast) / supply_total_adj) if supply_total_adj > 0 else 0.0
    inventory_efficiency_improved_pct = (
        (baseline_excess_ratio - new_excess_ratio) / baseline_excess_ratio * 100.0
        if baseline_excess_ratio > 0
        else 0.0
    )
    # Satisfaction proxy: higher availability + reduced OOS risk + improved online fill
    satisfaction_improve_pct = (
        0.55 * max(0.0, customer_impact["stock_availability_improved_pct"])
        + 0.45 * customer_impact["reduced_out_of_stock_risk_pct"]
    )
    satisfaction_improve_pct = float(max(0.0, min(100.0, satisfaction_improve_pct)))

    st.markdown("<div class='section-head'>😊 Business Impact</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class='card glow fade-in'>
              <div class='kpi-label'>Reduce stockouts</div>
              <div class='kpi-value' style='color:#36D399'>{customer_impact['reduced_out_of_stock_risk_pct']:.1f}%</div>
              <div class='subtle' style='margin-top:6px'>Lower modeled out-of-stock risk for {target_month}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='card glow fade-in' style='animation-delay:0.08s'>
              <div class='kpi-label'>Improve efficiency</div>
              <div class='kpi-value' style='color:{'#36D399' if inventory_efficiency_improved_pct >= 0 else '#EF4444'}'>{inventory_efficiency_improved_pct:.1f}%</div>
              <div class='subtle' style='margin-top:6px'>Less excess supply pressure vs demand.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class='card glow fade-in' style='animation-delay:0.16s'>
              <div class='kpi-label'>Enhance satisfaction</div>
              <div class='kpi-value' style='color:#ffffff'>{satisfaction_improve_pct:.1f}%</div>
              <div class='subtle' style='margin-top:6px'>Availability-driven satisfaction lift proxy.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Executive summary line (high-signal)
    st.markdown(
        f"""
        <div class='card'>
          <div class='kpi-label'>Executive summary</div>
          <div class='subtle' style='margin-top:8px'>
            For <b>Store {store}</b> • <b>Dept {dept}</b>, forecasted demand for <b>{target_month}</b> is <b>{format_money(demand_forecast)}</b>. "
            "Modeled supply is <b>{format_money(supply_month)}</b> ({status_label}). The dashboard recommends inventory actions that aim to improve availability and reduce out-of-stock risk.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        """
        <div class='divider'></div>
        <div class='subtle' style='text-align:center; font-weight:800;'>
          Built for Hackathon | Team XYZ
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
