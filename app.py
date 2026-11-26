import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="xGRAPM Matchup Simulator",
    page_icon="⚽",
    layout="wide",
)

@st.cache_resource
def load_bundle():
    v2_path = "models/xgrapm_inference_v2.joblib" if os.path.exists("models/xgrapm_inference_v2.joblib") else "xgrapm_inference_v2.joblib"
    v2 = joblib.load(v2_path)
    v1 = joblib.load("xgrapm_inference.joblib") if os.path.exists("xgrapm_inference.joblib") else {}
    return v2, v1

v2_data, v1_data = load_bundle()
ratings = v2_data["player_ratings"]
match_outcome_clf = v2_data.get("match_outcome_clf")
context_weights = v1_data.get("context_weights", {
    "home_advantage_coef": 0.24,
    "manpower_coef": 0.65,
})

players = [
    {
        "player_id": int(float(pid)),
        "player_name": d.get("player_name", f"Player {pid}"),
        "position": d.get("position", "MID"),
        "minutes_played": float(d.get("minutes_played", 0.0)),
        "beta_Micro_OnBall": float(d.get("beta_Micro_OnBall", 0.0)),
        "delta_Macro_OffBall": float(d.get("delta_Macro_OffBall", 0.0)),
        "beta_Total_xGRAPM": float(d.get("beta_Total_xGRAPM", 0.0)),
    }
    for pid, d in ratings.items()
    if float(d.get("minutes_played", 0.0)) >= 300
]
players.sort(key=lambda x: x["beta_Total_xGRAPM"], reverse=True)
id_to_player = {p["player_id"]: p for p in players}

def match_defaults(player_list: list[dict], queries: list[str]) -> list[str]:
    selected = []
    for q in queries:
        q_lower = q.lower()
        for p in player_list:
            label = f"{p['player_name']} ({p['position']} | ID: {p['player_id']})"
            if q_lower in p["player_name"].lower() and label not in selected:
                selected.append(label)
                break
    return selected

st.title("⚽ xGRAPM Football Matchup & Lineup Impact Simulator")
st.markdown(
    "Decomposes Big 5 player value into **On-Ball Micro Priors ($\\beta_{\\text{Micro}}$)** "
    "and **Off-Ball Macro Residuals ($\\delta_{\\text{Macro}}$)** to forecast net **npxG / 90** differentials and Stage 3 match odds."
)

label_to_id = {f"{p['player_name']} ({p['position']} | ID: {p['player_id']})": p["player_id"] for p in players}
all_labels = list(label_to_id.keys())

default_home = match_defaults(
    players,
    [
        "Messi", "Suárez Díaz", "Neymar", "Iniesta", "Busquets",
        "Rakitić", "Jordi Alba", "Piqué", "Daniel Alves", "Mascherano", "ter Stegen",
    ],
)
default_away = match_defaults(
    players,
    [
        "Ibrahimović", "Cavani", "Di María", "Verratti", "Matuidi",
        "Thiago Motta", "Thiago Emiliano da Silva", "David Luiz", "Maxwell", "Aurier", "Trapp",
    ],
)

with st.sidebar:
    st.header("Match Context")
    is_home_venue = st.toggle("Home Venue Advantage", value=True)
    home_red_cards = st.number_input("Home Red Cards", min_value=0, max_value=4, value=0, step=1)
    away_red_cards = st.number_input("Away Red Cards", min_value=0, max_value=4, value=0, step=1)
    st.caption(f"Active Pitch Manpower: **{11 - int(home_red_cards)}v{11 - int(away_red_cards)}**")
    st.divider()
    st.caption("Model Artifact: `models/xgrapm_inference_v2.joblib`")
    st.caption(f"Player Catalog: **{len(players):,}** Big 5 players (≥300 mins)")

col_home, col_away = st.columns(2)
with col_home:
    home_selected = st.multiselect(
        "Home Starting XI (11 players)",
        options=all_labels,
        default=default_home,
        max_selections=11,
    )
with col_away:
    away_options = [lbl for lbl in all_labels if lbl not in home_selected]
    away_default_clean = [lbl for lbl in default_away if lbl not in home_selected]
    away_selected = st.multiselect(
        "Away Starting XI (11 players)",
        options=away_options,
        default=away_default_clean,
        max_selections=11,
    )

if len(home_selected) != len(away_selected) or len(home_selected) == 0:
    st.info(
        f"Select equal lineup sizes for both teams (currently **{len(home_selected)}** Home vs. "
        f"**{len(away_selected)}** Away) so the shared baseline cancels cleanly."
    )
    st.stop()

home_players = [id_to_player[label_to_id[lbl]] for lbl in home_selected]
away_players = [id_to_player[label_to_id[lbl]] for lbl in away_selected]

micro_diff = sum(p["beta_Micro_OnBall"] for p in home_players) - sum(p["beta_Micro_OnBall"] for p in away_players)
macro_diff = sum(p["delta_Macro_OffBall"] for p in home_players) - sum(p["delta_Macro_OffBall"] for p in away_players)
net_lineup_diff = micro_diff + macro_diff

home_adv = float(context_weights.get("home_advantage_coef", 0.24)) if is_home_venue else 0.0
manpower_adv = (int(away_red_cards) - int(home_red_cards)) * float(context_weights.get("manpower_coef", 0.65))
context_adj = round(home_adv + manpower_adv, 4)
net_diff = round(net_lineup_diff + context_adj, 4)

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Net npxG / 90", f"{net_diff:+.3f}", delta="Home Favored" if net_diff > 0 else "Away Favored")
m2.metric("Lineup Skill Diff (Home − Away)", f"{net_lineup_diff:+.3f}")
m3.metric("On-Ball Micro Diff", f"{micro_diff:+.3f}")
m4.metric("Off-Ball Macro + Context", f"{macro_diff + context_adj:+.3f}")

if match_outcome_clf is not None:
    probs = match_outcome_clf.predict_proba(np.array([[net_diff]]))[0]
    p1, p2, p3 = st.columns(3)
    p1.metric("Stage 3 Home Win Prob", f"{probs[2]:.1%}")
    p2.metric("Stage 3 Draw Prob", f"{probs[1]:.1%}")
    p3.metric("Stage 3 Away Win Prob", f"{probs[0]:.1%}")

home_df = pd.DataFrame(home_players).assign(Side="Home")
away_df = pd.DataFrame(away_players).assign(Side="Away")
combined_df = pd.concat([home_df, away_df], ignore_index=True)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="On-Ball Micro Prior (β_Micro)",
        y=combined_df["player_name"],
        x=combined_df["beta_Micro_OnBall"],
        orientation="h",
        marker_color="#1f77b4",
    )
)
fig.add_trace(
    go.Bar(
        name="Off-Ball Macro Residual (δ_Macro)",
        y=combined_df["player_name"],
        x=combined_df["delta_Macro_OffBall"],
        orientation="h",
        marker_color="#2ca02c",
    )
)
fig.update_layout(
    title="Player-Level Micro vs. Macro Decomposition (npxG / 90)",
    barmode="relative",
    height=max(420, len(combined_df) * 28),
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="npxG / 90 Rating",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

t1, t2 = st.columns(2)
with t1:
    st.subheader("Home Players")
    st.dataframe(
        home_df[["player_name", "position", "beta_Micro_OnBall", "delta_Macro_OffBall", "beta_Total_xGRAPM"]],
        use_container_width=True,
        hide_index=True,
    )
with t2:
    st.subheader("Away Players")
    st.dataframe(
        away_df[["player_name", "position", "beta_Micro_OnBall", "delta_Macro_OffBall", "beta_Total_xGRAPM"]],
        use_container_width=True,
        hide_index=True,
    )
