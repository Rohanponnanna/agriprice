"""
Real-Time Crop Price Forecasting for Agri-MSMEs
NDVI + Weather + Mandi Price → XGBoost + ARIMA Ensemble → WhatsApp Alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time, os, json, warnings
warnings.filterwarnings("ignore")

from utils.data_fetcher import fetch_weather_data, fetch_mandi_prices, fetch_ndvi_data
from utils.feature_engineering import build_features
from models.ensemble import CropPriceForecast
from utils.whatsapp import send_whatsapp_alert

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPrice Forecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main palette */
    :root {
        --green: #1D9E75;
        --green-light: #E1F5EE;
        --amber: #BA7517;
        --amber-light: #FAEEDA;
        --red: #E24B4A;
        --red-light: #FCEBEB;
        --blue: #185FA5;
        --blue-light: #E6F1FB;
        --text: #2C2C2A;
        --muted: #5F5E5A;
        --border: rgba(136,135,128,0.2);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F9F8F5;
        border-right: 1px solid var(--border);
    }

    /* Metric cards */
    .metric-card {
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: var(--text);
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 13px;
        margin-top: 3px;
    }
    .delta-up   { color: var(--green); }
    .delta-down { color: var(--red); }

    /* Signal badge */
    .signal-buy  { background: var(--green-light); color: #085041;
                    border: 1px solid #9FE1CB; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }
    .signal-sell { background: var(--red-light);   color: #791F1F;
                    border: 1px solid #F7C1C1; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }
    .signal-hold { background: var(--amber-light); color: #633806;
                    border: 1px solid #FAC775; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }

    /* Section header */
    .section-header {
        font-size: 15px;
        font-weight: 500;
        color: var(--text);
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* Status pills */
    .pill-green { background: var(--green-light); color: #085041;
                  border-radius: 4px; padding: 2px 8px; font-size: 11px; }
    .pill-amber { background: var(--amber-light); color: #633806;
                  border-radius: 4px; padding: 2px 8px; font-size: 11px; }

    /* Table styling */
    .forecast-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .forecast-table th {
        background: #F1EFE8; font-weight: 500; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.04em;
        padding: 8px 12px; text-align: left; color: var(--muted);
    }
    .forecast-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }

    /* WhatsApp button */
    .wa-btn {
        background: #25D366; color: #fff; border: none; border-radius: 8px;
        padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer;
        display: inline-flex; align-items: center; gap: 8px;
    }

    /* Hide streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
CROPS = {
    "Tomato": {"emoji": "🍅", "unit": "₹/Quintal", "season": "Year-round"},
    "Onion":  {"emoji": "🧅", "unit": "₹/Quintal", "season": "Oct–Mar"},
    "Potato": {"emoji": "🥔", "unit": "₹/Quintal", "season": "Nov–Apr"},
    "Wheat":  {"emoji": "🌾", "unit": "₹/Quintal", "season": "Mar–Apr"},
    "Rice":   {"emoji": "🌾", "unit": "₹/Quintal", "season": "Oct–Nov"},
    "Cotton": {"emoji": "🌱", "unit": "₹/Quintal", "season": "Oct–Jan"},
    "Soybean":{"emoji": "🌿", "unit": "₹/Quintal", "season": "Sep–Nov"},
    "Maize":  {"emoji": "🌽", "unit": "₹/Quintal", "season": "Sep–Oct"},
}

MANDIS = {
    "Karnataka": ["Bangalore (APMC)", "Hubballi", "Mysuru", "Davanagere", "Belgaum"],
    "Maharashtra": ["Mumbai (Vashi)", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Punjab":      ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain"],
    "Uttar Pradesh":  ["Lucknow", "Agra", "Varanasi", "Kanpur", "Meerut"],
    "Andhra Pradesh": ["Visakhapatnam", "Guntur", "Vijayawada", "Kurnool", "Tirupati"],
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriPrice Forecast")
    st.markdown("<p style='color:#5F5E5A;font-size:13px;margin-top:-8px'>Satellite + Mandi + Weather Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Select Crop & Location**")
    crop = st.selectbox("Crop", list(CROPS.keys()),
                        format_func=lambda x: f"{CROPS[x]['emoji']} {x}")
    state = st.selectbox("State", list(MANDIS.keys()))
    mandi = st.selectbox("APMC Mandi", MANDIS[state])

    st.divider()
    st.markdown("**Forecast Settings**")
    forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
    confidence   = st.select_slider("Confidence interval", ["70%", "80%", "90%", "95%"], value="80%")
    show_history = st.slider("Historical data (days)", 14, 90, 30)

    st.divider()
    st.markdown("**WhatsApp Alerts**")
    phone = st.text_input("Farmer phone (+91XXXXXXXXXX)", placeholder="+919876543210")

    st.divider()
    st.markdown("**API Keys**")
    with st.expander("Configure (optional for demo)"):
        ow_key   = st.text_input("OpenWeather API key", type="password",
                                  value=os.getenv("OPENWEATHER_KEY",""))
        wa_token = st.text_input("WhatsApp token",      type="password",
                                  value=os.getenv("WHATSAPP_TOKEN",""))
        wa_phone_id = st.text_input("WA Phone ID",      type="password",
                                     value=os.getenv("WHATSAPP_PHONE_ID",""))
        if st.button("Save keys"):
            os.environ["OPENWEATHER_KEY"]     = ow_key
            os.environ["WHATSAPP_TOKEN"]      = wa_token
            os.environ["WHATSAPP_PHONE_ID"]   = wa_phone_id
            st.success("Saved for this session")

    run = st.button("🔍 Run Forecast", use_container_width=True, type="primary")


# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown(f"## {CROPS[crop]['emoji']} {crop} Price Forecast — {mandi}")
st.markdown(
    f"<span class='pill-green'>● Live</span> &nbsp;"
    f"<span style='color:#5F5E5A;font-size:13px'>{mandi} &nbsp;·&nbsp; {state} &nbsp;·&nbsp; {forecast_days}-day horizon &nbsp;·&nbsp; {confidence} confidence</span>",
    unsafe_allow_html=True,
)
st.write("")

if not run:
    # Landing state
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(
            "👈  Configure your crop, mandi, and location in the sidebar, then click **Run Forecast** to fetch live data and generate a 7-day price prediction.",
            icon="ℹ️",
        )
    with col_b:
        st.markdown("""
        **Data sources used**
        - 🛰 ISRO Bhuvan NDVI (crop health)
        - 🌦 OpenWeather API (climate)
        - 📊 Agmarknet APMC (mandi prices)
        - 🤖 XGBoost + ARIMA ensemble
        """)
    st.stop()


# ─── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("Fetching satellite, weather, and mandi data…"):
    progress = st.progress(0, text="Connecting to ISRO Bhuvan NDVI…")
    time.sleep(0.4)

    ndvi_df     = fetch_ndvi_data(crop, state, days=show_history)
    progress.progress(30, text="Fetching OpenWeather climate data…")
    time.sleep(0.3)

    weather_df  = fetch_weather_data(state, days=show_history)
    progress.progress(55, text="Scraping Agmarknet mandi prices…")
    time.sleep(0.4)

    mandi_df    = fetch_mandi_prices(crop, mandi, state, days=show_history)
    progress.progress(75, text="Building feature matrix…")
    time.sleep(0.2)

    features_df = build_features(ndvi_df, weather_df, mandi_df)
    progress.progress(90, text="Running ensemble model…")
    time.sleep(0.3)

    model    = CropPriceForecast()
    results  = model.forecast(features_df, horizon=forecast_days, ci=int(confidence.strip("%")))
    progress.progress(100, text="Done.")
    time.sleep(0.2)
    progress.empty()


# ─── Unpack results ───────────────────────────────────────────────────────────
forecast   = results["forecast"]       # list of dicts: date, price, low, high
history    = results["history"]        # list of dicts: date, price
metrics    = results["metrics"]        # dict: rmse, mae, mape
model_info = results["model_info"]     # dict: arima_weight, xgb_weight
signal     = results["signal"]         # BUY / SELL / HOLD
signal_reason = results["signal_reason"]

last_price = history[-1]["price"] if history else forecast[0]["price"]
next_price = forecast[0]["price"]
week_high  = max(f["high"] for f in forecast)
week_low   = min(f["low"]  for f in forecast)
pct_change = (forecast[-1]["price"] - last_price) / last_price * 100

# ─── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

for col, label, value, delta, delta_pos in [
    (c1, "Today's Price",   f"₹{last_price:,.0f}",  f"+{pct_change:.1f}% projected", pct_change >= 0),
    (c2, "Tomorrow",        f"₹{next_price:,.0f}",  f"{'▲' if next_price>last_price else '▼'} ₹{abs(next_price-last_price):.0f}", next_price >= last_price),
    (c3, f"{forecast_days}-Day High", f"₹{week_high:,.0f}", "Upper band",  True),
    (c4, f"{forecast_days}-Day Low",  f"₹{week_low:,.0f}",  "Lower band",  False),
    (c5, "Forecast MAPE",   f"{metrics['mape']:.1f}%", f"RMSE ₹{metrics['rmse']:.0f}", metrics['mape'] < 8),
]:
    d_class = "delta-up" if delta_pos else "delta-down"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {d_class}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ─── Signal banner ────────────────────────────────────────────────────────────
sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]
sig_icon  = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸"}[signal]
st.markdown(
    f"**Market signal:** &nbsp;<span class='{sig_class}'>{sig_icon} {signal}</span>"
    f"&nbsp; <span style='color:#5F5E5A;font-size:13px'>{signal_reason}</span>",
    unsafe_allow_html=True,
)
st.write("")


# ─── Main forecast chart ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "🌦 Feature Analysis", "🤖 Model Details"])

with tab1:
    fig = go.Figure()

    hist_dates  = [h["date"] for h in history]
    hist_prices = [h["price"] for h in history]
    fc_dates    = [f["date"] for f in forecast]
    fc_prices   = [f["price"] for f in forecast]
    fc_low      = [f["low"]   for f in forecast]
    fc_high     = [f["high"]  for f in forecast]

    # CI band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_high  + fc_low[::-1],
        fill="toself",
        fillcolor="rgba(29, 158, 117, 0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{confidence} Confidence band",
        hoverinfo="skip",
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode="lines",
        line=dict(color="#888780", width=1.5),
        name="Historical price",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_prices,
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2.5, dash="dot"),
        marker=dict(size=7, color="#1D9E75", line=dict(color="#fff", width=2)),
        name="Forecast (ensemble)",
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}/Quintal<extra></extra>",
    ))

    # Divider line — using add_shape + add_annotation to avoid
    # Plotly version bug where add_vline with annotation_text
    # triggers an internal _mean() TypeError on string x-axis values.
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=today_str,
        x1=today_str,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#B4B2A9", dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=today_str,
        y=1,
        xref="x",
        yref="paper",
        text="Today",
        showarrow=False,
        font=dict(size=11, color="#5F5E5A"),
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.7)",
    )

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(orientation="h", y=-0.12, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#E8E6DF", showgrid=True, tickfont=dict(size=11)),
        yaxis=dict(
            gridcolor="#E8E6DF", showgrid=True, tickfont=dict(size=11),
            tickprefix="₹", tickformat=",",
            title=dict(text="Price (₹/Quintal)", font=dict(size=12)),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("<div class='section-header'>7-Day price outlook</div>", unsafe_allow_html=True)
    rows = ""
    for i, f in enumerate(forecast):
        chg = f["price"] - last_price
        chg_str = f"{'▲' if chg>0 else '▼'} ₹{abs(chg):.0f} ({abs(chg/last_price*100):.1f}%)"
        chg_color = "#085041" if chg >= 0 else "#791F1F"
        day_label = "Tomorrow" if i == 0 else f["date"]
        rows += f"""
        <tr>
          <td><strong>{day_label}</strong></td>
          <td>₹{f['price']:,.0f}</td>
          <td style="color:#888780">₹{f['low']:,.0f} – ₹{f['high']:,.0f}</td>
          <td style="color:{chg_color};font-weight:500">{chg_str}</td>
        </tr>"""
    st.markdown(f"""
    <table class="forecast-table">
      <thead><tr>
        <th>Date</th><th>Forecast ₹/Q</th>
        <th>{confidence} Range</th><th>vs Today</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)


with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        # NDVI chart
        st.markdown("<div class='section-header'>Crop health (NDVI index)</div>", unsafe_allow_html=True)
        fig_ndvi = go.Figure()
        fig_ndvi.add_trace(go.Scatter(
            x=ndvi_df["date"].tolist(), y=ndvi_df["ndvi"].tolist(),
            fill="tozeroy", mode="lines",
            line=dict(color="#1D9E75", width=2),
            fillcolor="rgba(29,158,117,0.12)",
            name="NDVI",
        ))
        fig_ndvi.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E8E6DF"), yaxis=dict(gridcolor="#E8E6DF", range=[0,1]),
            showlegend=False,
        )
        st.plotly_chart(fig_ndvi, use_container_width=True)

        # Rainfall chart
        st.markdown("<div class='section-header'>Rainfall (mm)</div>", unsafe_allow_html=True)
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=weather_df["date"].tolist(), y=weather_df["rainfall"].tolist(),
            marker_color="#378ADD", opacity=0.75, name="Rainfall",
        ))
        fig_rain.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E8E6DF"), yaxis=dict(gridcolor="#E8E6DF"),
            showlegend=False,
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with col_right:
        # Feature importance
        st.markdown("<div class='section-header'>Top predictive features (XGBoost)</div>", unsafe_allow_html=True)
        feat_imp = results.get("feature_importance", {})
        fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color="#7F77DD",
        ))
        fig_fi.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E8E6DF", title="Importance score", titlefont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Temperature chart
        st.markdown("<div class='section-header'>Temperature trend (°C)</div>", unsafe_allow_html=True)
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=weather_df["date"].tolist(), y=weather_df["temp_max"].tolist(),
            mode="lines", line=dict(color="#D85A30", width=1.5), name="Max",
        ))
        fig_temp.add_trace(go.Scatter(
            x=weather_df["date"].tolist(), y=weather_df["temp_min"].tolist(),
            mode="lines", line=dict(color="#378ADD", width=1.5),
            fill="tonexty", fillcolor="rgba(186,117,23,0.07)", name="Min",
        ))
        fig_temp.update_layout(
            height=160, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E8E6DF"),
            yaxis=dict(gridcolor="#E8E6DF", ticksuffix="°"),
            legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
        )
        st.plotly_chart(fig_temp, use_container_width=True)


with tab3:
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<div class='section-header'>Ensemble blend weights</div>", unsafe_allow_html=True)
        fig_blend = go.Figure(go.Pie(
            labels=["XGBoost", "ARIMA"],
            values=[model_info["xgb_weight"] * 100, model_info["arima_weight"] * 100],
            hole=0.55,
            marker=dict(colors=["#534AB7", "#0F6E56"]),
            textfont=dict(size=12),
        ))
        fig_blend.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=12)),
            annotations=[dict(text="Blend", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=13, color="#444441"))],
        )
        st.plotly_chart(fig_blend, use_container_width=True)

        st.markdown("<div class='section-header'>Model performance metrics</div>", unsafe_allow_html=True)
        perf = {
            "RMSE (₹/Quintal)": f"₹{metrics['rmse']:.2f}",
            "MAE  (₹/Quintal)": f"₹{metrics['mae']:.2f}",
            "MAPE (%)": f"{metrics['mape']:.2f}%",
            "R² Score": f"{metrics['r2']:.3f}",
            "Training samples": str(metrics.get("train_n", 720)),
            "Validation split": "80 / 20",
        }
        for k, v in perf.items():
            cols = st.columns([2, 1])
            cols[0].markdown(f"<span style='font-size:13px;color:#5F5E5A'>{k}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span style='font-size:13px;font-weight:500'>{v}</span>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='section-header'>ARIMA diagnostics</div>", unsafe_allow_html=True)
        arima_resid = results.get("arima_residuals", [])
        fig_res = go.Figure(go.Scatter(
            x=list(range(len(arima_resid))), y=arima_resid,
            mode="lines", line=dict(color="#888780", width=1),
        ))
        fig_res.add_hline(y=0, line_dash="dot", line_color="#B4B2A9")
        fig_res.update_layout(
            height=180, margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="ARIMA residuals", font=dict(size=12), x=0),
            xaxis=dict(gridcolor="#E8E6DF", title="Index", titlefont=dict(size=11)),
            yaxis=dict(gridcolor="#E8E6DF", title="Residual ₹", titlefont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("<div class='section-header'>XGBoost training history</div>", unsafe_allow_html=True)
        xgb_train = results.get("xgb_train_history", {})
        fig_xgb = go.Figure()
        if xgb_train:
            fig_xgb.add_trace(go.Scatter(y=xgb_train.get("train",[]), mode="lines",
                                          line=dict(color="#534AB7", width=1.5), name="Train"))
            fig_xgb.add_trace(go.Scatter(y=xgb_train.get("val",[]),   mode="lines",
                                          line=dict(color="#D4537E", width=1.5), name="Val"))
        fig_xgb.update_layout(
            height=160, margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="RMSE per boosting round", font=dict(size=12), x=0),
            xaxis=dict(gridcolor="#E8E6DF"),
            yaxis=dict(gridcolor="#E8E6DF"),
            legend=dict(font=dict(size=11), orientation="h", y=-0.35),
        )
        st.plotly_chart(fig_xgb, use_container_width=True)


# ─── WhatsApp section ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📱 Send WhatsApp Alert")

wa_col1, wa_col2 = st.columns([2, 1])

with wa_col1:
    preview_msg = f"""🌾 *{crop} Price Alert — {mandi}*

📅 7-Day Forecast ({forecast[0]['date']} → {forecast[-1]['date']})

💰 *Today:* ₹{last_price:,.0f}/Quintal
📈 *7-Day High:* ₹{week_high:,.0f}/Quintal  
📉 *7-Day Low:* ₹{week_low:,.0f}/Quintal

🎯 *Signal: {signal}* — {signal_reason}

📊 Source: NDVI + APMC + OpenWeather
🤖 Model: XGBoost + ARIMA Ensemble (MAPE: {metrics['mape']:.1f}%)

_AgriPrice Forecast System — Powered by Anthropic AI_"""

    st.text_area("Message preview", preview_msg, height=230, disabled=True)

with wa_col2:
    st.markdown("**Cooperative list**")
    farmer_numbers = st.text_area(
        "Phone numbers (one per line)",
        placeholder="+919876543210\n+919812345678\n+919898989898",
        height=130,
    )
    batch_send = st.checkbox("Send to all cooperative members")

    if st.button("📲 Send WhatsApp Alert", use_container_width=True):
        numbers = []
        if phone:
            numbers.append(phone)
        if batch_send and farmer_numbers:
            numbers.extend([n.strip() for n in farmer_numbers.strip().split("\n") if n.strip()])

        if not numbers:
            st.warning("Please enter at least one phone number.")
        elif not os.getenv("WHATSAPP_TOKEN"):
            st.info("WhatsApp token not configured. In demo mode, the message above is what would be sent.", icon="ℹ️")
        else:
            with st.spinner(f"Sending to {len(numbers)} farmer(s)…"):
                results_wa = send_whatsapp_alert(numbers, preview_msg)
            ok  = sum(1 for r in results_wa if r.get("success"))
            err = len(results_wa) - ok
            if ok:
                st.success(f"✅ Alert sent to {ok} farmer(s).")
            if err:
                st.error(f"❌ Failed for {err} number(s). Check logs.")


# ─── Download ─────────────────────────────────────────────────────────────────
st.divider()
dl_col1, dl_col2, _ = st.columns([1, 1, 3])

forecast_df = pd.DataFrame(forecast)
with dl_col1:
    st.download_button(
        "⬇️ Download forecast CSV",
        forecast_df.to_csv(index=False).encode(),
        f"{crop}_{mandi}_forecast_{datetime.now():%Y%m%d}.csv",
        "text/csv",
        use_container_width=True,
    )
with dl_col2:
    combined = {
        "crop": crop, "mandi": mandi, "state": state,
        "generated_at": datetime.now().isoformat(),
        "signal": signal, "metrics": metrics,
        "forecast": forecast,
    }
    st.download_button(
        "⬇️ Download full JSON",
        json.dumps(combined, indent=2, default=str).encode(),
        f"{crop}_{mandi}_full_{datetime.now():%Y%m%d}.json",
        "application/json",
        use_container_width=True,
    )

# Footer
st.markdown(
    "<div style='text-align:center;color:#B4B2A9;font-size:11px;margin-top:2rem'>"
    "AgriPrice Forecast · ISRO Bhuvan NDVI · Agmarknet APMC · OpenWeather · "
    "XGBoost + ARIMA Ensemble · Meta WhatsApp Cloud API"
    "</div>",
    unsafe_allow_html=True,
)
