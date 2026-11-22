import os, math, unicodedata, joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="xGRAPM Football Analytics — 2015/16 Big 5 Leagues")
model_data = joblib.load("xgrapm_inference.joblib")
ratings = model_data.get("player_ratings", {})
context = model_data.get("context_weights", {})
top_actions = model_data.get("player_top_actions", {})
threat_grid = model_data.get("threat_grid", [])
tactical_zones = model_data.get("tactical_zones", [])
threat_grid_contexts = model_data.get("threat_grid_contexts", {"all": threat_grid})
tactical_zones_contexts = model_data.get("tactical_zones_contexts", {"all": tactical_zones})

def num(val, default=0.0):
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def txt(val, default="Unknown"):
    return val.strip() if isinstance(val, str) and val.strip() and val.strip().lower() != "nan" else default

def norm_str(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower()) if unicodedata.category(c) != "Mn")

def get_player_record(pid: int):
    return ratings.get(pid) or ratings.get(str(pid))

BALLON_DOR_2016 = [
    {"bd_rank": 1, "name": "Cristiano Ronaldo", "club": "Real Madrid", "keys": ["cristiano ronaldo"]},
    {"bd_rank": 2, "name": "Lionel Messi", "club": "Barcelona", "keys": ["lionel", "messi"]},
    {"bd_rank": 3, "name": "Antoine Griezmann", "club": "Atlético Madrid", "keys": ["griezmann"]},
    {"bd_rank": 4, "name": "Luis Suárez", "club": "Barcelona", "keys": ["luis", "suarez"]},
    {"bd_rank": 5, "name": "Neymar", "club": "Barcelona", "keys": ["neymar"]},
    {"bd_rank": 6, "name": "Gareth Bale", "club": "Real Madrid", "keys": ["gareth", "bale"]},
    {"bd_rank": 7, "name": "Riyad Mahrez", "club": "Leicester City", "keys": ["mahrez"]},
    {"bd_rank": 8, "name": "Jamie Vardy", "club": "Leicester City", "keys": ["vardy"]},
    {"bd_rank": 9, "name": "Gianluigi Buffon", "club": "Juventus", "keys": ["buffon"]},
    {"bd_rank": 9, "name": "Pepe", "club": "Real Madrid", "keys": ["kepler laveran", "ferreira"]},
    {"bd_rank": 11, "name": "Pierre-Emerick Aubameyang", "club": "Borussia Dortmund", "keys": ["aubameyang"]},
    {"bd_rank": 13, "name": "Zlatan Ibrahimović", "club": "Paris Saint-Germain", "keys": ["ibrahimovic"]},
    {"bd_rank": 14, "name": "Paul Pogba", "club": "Juventus", "keys": ["pogba"]},
    {"bd_rank": 14, "name": "Arturo Vidal", "club": "Bayern Munich", "keys": ["arturo", "vidal"]},
    {"bd_rank": 16, "name": "Robert Lewandowski", "club": "Bayern Munich", "keys": ["lewandowski"]},
    {"bd_rank": 17, "name": "Toni Kroos", "club": "Real Madrid", "keys": ["toni kroos"]},
    {"bd_rank": 17, "name": "Luka Modrić", "club": "Real Madrid", "keys": ["modric"]},
    {"bd_rank": 17, "name": "Dimitri Payet", "club": "West Ham United", "keys": ["payet"]},
    {"bd_rank": 20, "name": "Sergio Agüero", "club": "Manchester City", "keys": ["sergio", "aguero"]},
    {"bd_rank": 20, "name": "Kevin De Bruyne", "club": "Manchester City", "keys": ["de bruyne"]},
    {"bd_rank": 20, "name": "Paulo Dybala", "club": "Juventus", "keys": ["dybala"]},
    {"bd_rank": 20, "name": "Diego Godín", "club": "Atlético Madrid", "keys": ["godin"]},
    {"bd_rank": 20, "name": "Gonzalo Higuaín", "club": "Napoli", "keys": ["higuain"]},
    {"bd_rank": 20, "name": "Andrés Iniesta", "club": "Barcelona", "keys": ["iniesta"]},
    {"bd_rank": 20, "name": "Koke", "club": "Atlético Madrid", "keys": ["resurreccion"]},
    {"bd_rank": 20, "name": "Hugo Lloris", "club": "Tottenham Hotspur", "keys": ["lloris"]},
    {"bd_rank": 20, "name": "Thomas Müller", "club": "Bayern Munich", "keys": ["thomas", "muller"]},
    {"bd_rank": 20, "name": "Manuel Neuer", "club": "Bayern Munich", "keys": ["neuer"]},
    {"bd_rank": 20, "name": "Sergio Ramos", "club": "Real Madrid", "keys": ["sergio ramos"]},
]

BALLON_DOR_2015 = [
    {"bd_rank": 1, "name": "Lionel Messi", "club": "Barcelona", "keys": ["lionel", "messi"]},
    {"bd_rank": 2, "name": "Cristiano Ronaldo", "club": "Real Madrid", "keys": ["cristiano ronaldo"]},
    {"bd_rank": 3, "name": "Neymar", "club": "Barcelona", "keys": ["neymar"]},
    {"bd_rank": 4, "name": "Robert Lewandowski", "club": "Bayern Munich", "keys": ["lewandowski"]},
    {"bd_rank": 5, "name": "Luis Suárez", "club": "Barcelona", "keys": ["luis", "suarez"]},
    {"bd_rank": 6, "name": "Thomas Müller", "club": "Bayern Munich", "keys": ["thomas", "muller"]},
    {"bd_rank": 7, "name": "Manuel Neuer", "club": "Bayern Munich", "keys": ["neuer"]},
    {"bd_rank": 8, "name": "Eden Hazard", "club": "Chelsea", "keys": ["eden hazard"]},
    {"bd_rank": 9, "name": "Andrés Iniesta", "club": "Barcelona", "keys": ["iniesta"]},
    {"bd_rank": 10, "name": "Alexis Sánchez", "club": "Arsenal", "keys": ["alexis", "sanchez"]},
    {"bd_rank": 11, "name": "Zlatan Ibrahimović", "club": "Paris Saint-Germain", "keys": ["ibrahimovic"]},
    {"bd_rank": 12, "name": "Yaya Touré", "club": "Manchester City", "keys": ["yaya toure"]},
    {"bd_rank": 13, "name": "Sergio Agüero", "club": "Manchester City", "keys": ["sergio", "aguero"]},
    {"bd_rank": 14, "name": "Javier Mascherano", "club": "Barcelona", "keys": ["mascherano"]},
    {"bd_rank": 15, "name": "Paul Pogba", "club": "Juventus", "keys": ["pogba"]},
    {"bd_rank": 16, "name": "Gareth Bale", "club": "Real Madrid", "keys": ["gareth", "bale"]},
    {"bd_rank": 17, "name": "Arturo Vidal", "club": "Bayern Munich", "keys": ["arturo", "vidal"]},
    {"bd_rank": 18, "name": "Kevin De Bruyne", "club": "Manchester City", "keys": ["de bruyne"]},
    {"bd_rank": 19, "name": "James Rodríguez", "club": "Real Madrid", "keys": ["james", "rodriguez"]},
    {"bd_rank": 20, "name": "Karim Benzema", "club": "Real Madrid", "keys": ["benzema"]},
    {"bd_rank": 21, "name": "Toni Kroos", "club": "Real Madrid", "keys": ["toni kroos"]},
    {"bd_rank": 22, "name": "Arjen Robben", "club": "Bayern Munich", "keys": ["robben"]},
    {"bd_rank": 23, "name": "Ivan Rakitić", "club": "Barcelona", "keys": ["rakitic"]},
]

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/api/players")
def get_players(min_minutes: int = 900, pos: str = "ALL", sort_by: str = "total"):
    out = []
    for pid, d in ratings.items():
        try:
            pid_int = int(float(pid))
        except (TypeError, ValueError):
            continue
        mins = num(d.get("minutes_played"), 0.0)
        role = txt(d.get("position"), "MID")
        if mins >= min_minutes and (pos == "ALL" or role == pos):
            micro = num(d.get("beta_Micro_OnBall"))
            micro_off = num(d.get("beta_Micro_Off"), micro)
            micro_def = num(d.get("beta_Micro_Def"), 0.0)
            macro = num(d.get("delta_Macro_OffBall"))
            total = num(d.get("beta_Total_xGRAPM"))
            acts_90 = num(d.get("actions_per_90"), 55.0)
            eff_100 = num(d.get("threat_per_100"), (total / max(acts_90, 1.0)) * 100.0)
            role_pct = num(d.get("role_pct"), 50.0)
            out.append({
                "player_id": pid_int,
                "player_name": txt(d.get("player_name"), f"Player {pid_int}"),
                "position": role,
                "minutes_played": int(round(mins)),
                "actions_per_90": round(acts_90, 1),
                "beta_Micro_OnBall": round(micro, 4),
                "beta_Micro_Off": round(micro_off, 4),
                "beta_Micro_Def": round(micro_def, 4),
                "delta_Macro_OffBall": round(macro, 4),
                "beta_Total_xGRAPM": round(total, 4),
                "threat_per_100": round(eff_100, 3),
                "role_pct": round(role_pct, 1),
            })

    sort_key_map = {
        "total": lambda x: x["beta_Total_xGRAPM"],
        "role_pct": lambda x: (x["role_pct"], x["beta_Total_xGRAPM"]),
        "efficiency": lambda x: x["threat_per_100"],
        "micro_off": lambda x: x["beta_Micro_Off"],
        "micro_def": lambda x: x["beta_Micro_Def"],
        "macro": lambda x: x["delta_Macro_OffBall"],
    }
    out.sort(key=sort_key_map.get(sort_by, sort_key_map["total"]), reverse=True)
    return out

@app.get("/api/player/{player_id}")
def get_player_details(player_id: int):
    d = get_player_record(player_id)
    if not d:
        raise HTTPException(status_code=404, detail="Player not found")
    acts = top_actions.get(player_id) or top_actions.get(str(player_id)) or top_actions.get(f"{player_id}_15/16") or []
    micro = num(d.get("beta_Micro_OnBall"))
    total = num(d.get("beta_Total_xGRAPM"))
    acts_90 = num(d.get("actions_per_90"), 55.0)
    return {
        "ratings": {
            "player_name": txt(d.get("player_name"), f"Player {player_id}"),
            "position": txt(d.get("position"), "MID"),
            "minutes_played": int(round(num(d.get("minutes_played"), 0.0))),
            "actions_per_90": round(acts_90, 1),
            "beta_Micro_OnBall": round(micro, 4),
            "beta_Micro_Off": round(num(d.get("beta_Micro_Off"), micro), 4),
            "beta_Micro_Def": round(num(d.get("beta_Micro_Def"), 0.0), 4),
            "delta_Macro_OffBall": round(num(d.get("delta_Macro_OffBall")), 4),
            "beta_Total_xGRAPM": round(total, 4),
            "threat_per_100": round(num(d.get("threat_per_100"), (total / max(acts_90, 1.0)) * 100.0), 3),
            "role_pct": round(num(d.get("role_pct"), 50.0), 1),
        },
        "top_actions": acts,
    }

@app.get("/api/ballon-dor")
def compare_ballon_dor(year: int = 2016, min_minutes: int = 900):
    pool = []
    for pid, d in ratings.items():
        mins = num(d.get("minutes_played"), 0.0)
        if mins < min_minutes:
            continue
        name = txt(d.get("player_name"))
        total = num(d.get("beta_Total_xGRAPM"))
        acts_90 = num(d.get("actions_per_90"), 55.0)
        pool.append({
            "player_id": int(float(pid)),
            "player_name": name,
            "norm_name": norm_str(name),
            "position": txt(d.get("position"), "MID"),
            "mins_1516": int(round(mins)),
            "micro_1516": round(num(d.get("beta_Micro_OnBall")), 4),
            "macro_1516": round(num(d.get("delta_Macro_OffBall")), 4),
            "total_1516": round(total, 4),
            "threat_per_100": round(num(d.get("threat_per_100"), (total / max(acts_90, 1.0)) * 100.0), 3),
            "role_pct": round(num(d.get("role_pct"), 50.0), 1),
        })

    pool.sort(key=lambda x: x["total_1516"], reverse=True)
    for idx, p in enumerate(pool):
        p["league_rank"] = idx + 1

    nominees = BALLON_DOR_2016 if year == 2016 else BALLON_DOR_2015
    matched_ids = set()
    comparison = []

    for nom in nominees:
        keys = [norm_str(k) for k in nom["keys"]]
        match = next((p for p in pool if all(k in p["norm_name"] for k in keys)), None)
        if match:
            matched_ids.add(match["player_id"])
            comparison.append({
                "bd_rank": nom["bd_rank"],
                "name": nom["name"],
                "club": nom["club"],
                "position": match["position"],
                "mins_1516": match["mins_1516"],
                "micro_1516": match["micro_1516"],
                "macro_1516": match["macro_1516"],
                "total_1516": match["total_1516"],
                "threat_per_100": match["threat_per_100"],
                "role_pct": match["role_pct"],
                "league_rank": match["league_rank"],
            })

    by_model = sorted(comparison, key=lambda x: x["total_1516"], reverse=True)
    for idx, item in enumerate(by_model):
        item["model_rank"] = idx + 1
        item["rank_diff"] = item["bd_rank"] - item["model_rank"]

    snubs = [
        {
            "model_rank": p["league_rank"],
            "player_name": p["player_name"],
            "position": p["position"],
            "role_pct": p["role_pct"],
            "threat_per_100": p["threat_per_100"],
            "total_1516": p["total_1516"],
        }
        for p in pool[:25] if p["player_id"] not in matched_ids
    ][:8]

    return {"year": year, "pool_size": len(pool), "comparison": comparison, "snubs": snubs}

class ContextSimRequest(BaseModel):
    team_gap_npxg: float = 0.0
    is_home_venue: bool = True
    score_diff: int = 0
    minute: int = 0
    manpower_diff: int = 0

def poisson_match_wdl(lam_h: float, lam_a: float, goals_h: int = 0, goals_a: int = 0, max_g: int = 9):
    pmf_h = [math.exp(-lam_h) * (lam_h ** k) / math.factorial(k) for k in range(max_g + 1)]
    pmf_a = [math.exp(-lam_a) * (lam_a ** k) / math.factorial(k) for k in range(max_g + 1)]
    wp = dp = lp = 0.0
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = pmf_h[i] * pmf_a[j]
            fh, fa = goals_h + i, goals_a + j
            if fh > fa:
                wp += p
            elif fh == fa:
                dp += p
            else:
                lp += p
    tot = max(wp + dp + lp, 1e-9)
    return round(wp / tot * 100, 1), round(dp / tot * 100, 1), round(lp / tot * 100, 1)

def eval_context_scenario(team_gap: float, is_home: bool, score_diff: int, minute: int, manpower_diff: int):
    home_coef = num(context.get("home_advantage_coef"))
    lead_coef = num(context.get("is_leading_coef"))
    trail_coef = num(context.get("is_trailing_coef"))
    man_coef = num(context.get("manpower_coef"))

    venue_adj = home_coef if is_home else 0.0
    if score_diff > 0:
        scale = 1.0 if score_diff == 1 else 1.35
        state_adj = lead_coef * scale
    elif score_diff < 0:
        scale = 1.0 if score_diff == -1 else 1.35
        state_adj = trail_coef * scale
    else:
        state_adj = 0.0

    man_adj = manpower_diff * man_coef
    net_rate = team_gap + venue_adj + state_adj + man_adj

    rem_frac = max(0.05, (90 - max(0, min(85, minute))) / 90.0)
    lam_h = max(0.08, (1.35 + 0.5 * net_rate) * rem_frac)
    lam_a = max(0.08, (1.35 - 0.5 * net_rate) * rem_frac)

    gh = max(0, score_diff)
    ga = max(0, -score_diff)
    wp, dp, lp = poisson_match_wdl(lam_h, lam_a, gh, ga)

    return {
        "venue_adj": round(venue_adj, 4),
        "state_adj": round(state_adj, 4),
        "manpower_adj": round(man_adj, 4),
        "net_npxg_90": round(net_rate, 4),
        "rem_xg_home": round(lam_h, 2),
        "rem_xg_away": round(lam_a, 2),
        "win_prob": wp,
        "draw_prob": dp,
        "loss_prob": lp,
    }

@app.post("/api/context-sim")
def simulate_context(req: ContextSimRequest):
    current = eval_context_scenario(req.team_gap_npxg, req.is_home_venue, req.score_diff, req.minute, req.manpower_diff)
    benchmarks = [
        {"label": "Away & -1 Behind", **eval_context_scenario(req.team_gap_npxg, False, -1, req.minute, 0)},
        {"label": "Away & Tied", **eval_context_scenario(req.team_gap_npxg, False, 0, req.minute, 0)},
        {"label": "Home & Tied", **eval_context_scenario(req.team_gap_npxg, True, 0, req.minute, 0)},
        {"label": "Home & +1 Ahead", **eval_context_scenario(req.team_gap_npxg, True, 1, req.minute, 0)},
        {"label": "Home & 11v10", **eval_context_scenario(req.team_gap_npxg, True, 0, req.minute, 1)},
    ]
    raw_weights = {k: round(num(v), 4) for k, v in context.items()}
    return {"weights": raw_weights, "current": current, "benchmarks": benchmarks}

@app.get("/api/tactics")
def get_tactics():
    return {
        "threat_grid": threat_grid,
        "tactical_zones": tactical_zones,
        "threat_grid_contexts": threat_grid_contexts,
        "tactical_zones_contexts": tactical_zones_contexts,
    }
