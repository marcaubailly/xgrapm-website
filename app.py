import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="xGRAPM Matchup Simulator",
    page_icon="⚽",
    layout="wide",
)

# Resolve Cloud Run API URL (works both locally and on Streamlit Community Cloud)
DEFAULT_API_URL = "https://xgrapm-api-753801721214.us-central1.run.app"
try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
except Exception:
    API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


@st.cache_data(ttl=3600)
def fetch_players(api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/players", timeout=30)
    resp.raise_for_status()
    return resp.json()["players"]


def run_prediction(
    api_url: str,
    home_ids: list[int],
    away_ids: list[int],
    is_home_venue: bool,
    home_red_cards: int,
    away_red_cards: int,
) -> dict:
    payload = {
        "home_player_ids": home_ids,
        "away_player_ids": away_ids,
        "is_home_venue": is_home_venue,
        "home_red_cards": home_red_cards,
        "away_red_cards": away_red_cards,
    }
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def match_defaults(players: list[dict], queries: list[str]) -> list[str]:
    selected = []
    for q in queries:
        q_lower = q.lower()
        for p in players:
            label = f"{p['player_name']} (ID: {p['player_id']})"
            if q_lower in p["player_name"].lower() and label not in selected:
                selected.append(label)
                break
    return selected


st.title("⚽ xGRAPM Football Matchup & Lineup Impact Simulator")
st.markdown(
    "Decomposes Big 5 player value into **On-Ball Micro Priors ($\\beta_{\\text{Micro}}$)** "
    "and **Off-Ball Macro Residuals ($\\delta_{\\text{Macro}}$)** to forecast net **npxG / 90** differentials."
)

try:
    players = fetch_players(API_URL)
except Exception as exc:
    st.error(f"Could not connect to Cloud Run API at `{API_URL}`: {exc}")
    st.stop()

label_to_id = {f"{p['player_name']} (ID: {p['player_id']})": p["player_id"] for p in players}
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
    st.divider()
    st.caption(f"Connected API:\n`{API_URL}`")
    st.caption(f"Player Catalog: **{len(players):,}** Big 5 players")

col_home, col_away = st.columns(2)
with col_home:
    home_selected = st.multiselect(
        "Home Lineup (up to 11 players)",
        options=all_labels,
        default=default_home,
        max_selections=11,
    )
with col_away:
    away_selected = st.multiselect(
        "Away Lineup (up to 11 players)",
        options=all_labels,
        default=default_away,
        max_selections=11,
    )

if not home_selected or not away_selected:
    st.warning("Select at least one player for both Home and Away lineups.")
    st.stop()

home_ids = [label_to_id[lbl] for lbl in home_selected]
away_ids = [label_to_id[lbl] for lbl in away_selected]

with st.spinner("Computing xGRAPM matchup differential..."):
    result = run_prediction(
        API_URL,
        home_ids,
        away_ids,
        is_home_venue,
        int(home_red_cards),
        int(away_red_cards),
    )

comp = result["components"]
net_diff = result["predicted_npxg_diff_90"]
context_adj = round(comp["home_advantage_effect_90"] + comp["manpower_effect_90"], 4)

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Predicted Net npxG / 90",
    f"{net_diff:+.3f}",
    delta="Home Favored" if net_diff > 0 else "Away Favored",
)
m2.metric("Lineup Skill Diff (Home − Away)", f"{comp['net_lineup_diff_90']:+.3f}")
m3.metric("On-Ball Micro Diff", f"{comp['micro_diff_90']:+.3f}")
m4.metric("Off-Ball Macro + Context", f"{comp['macro_diff_90'] + context_adj:+.3f}")

# Player Decomposition Chart
home_df = pd.DataFrame(result["home_players"]).assign(Side="Home")
away_df = pd.DataFrame(result["away_players"]).assign(Side="Away")
combined_df = pd.concat([home_df, away_df], ignore_index=True)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="On-Ball Micro Prior (β_Micro)",
        y=combined_df["player_name"],
        x=combined_df["beta_micro_on_ball"],
        orientation="h",
        marker_color="#1f77b4",
    )
)
fig.add_trace(
    go.Bar(
        name="Off-Ball Macro Residual (δ_Macro)",
        y=combined_df["player_name"],
        x=combined_df["delta_macro_off_ball"],
        orientation="h",
        marker_color="#2ca02c",
    )
)
fig.update_layout(
    title="Player-Level Micro vs. Macro Decomposition (npxG / 90)",
    barmode="relative",
    height=max(420, len(combined_df) * 28),
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="npxG / 90 Contribution",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

t1, t2 = st.columns(2)
with t1:
    st.subheader("Home Players")
    st.dataframe(
        home_df[["player_name", "beta_micro_on_ball", "delta_macro_off_ball", "beta_total_xgrapm"]],
        use_container_width=True,
        hide_index=True,
    )
with t2:
    st.subheader("Away Players")
    st.dataframe(
        away_df[["player_name", "beta_micro_on_ball", "delta_macro_off_ball", "beta_total_xgrapm"]],
        use_container_width=True,
        hide_index=True,
    )
