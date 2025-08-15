# app.py
from __future__ import annotations
import math
from typing import Dict, Tuple, Iterable, List

import pandas as pd
import numpy as np
import streamlit as st

# ----------------------- UI CONFIG -------------------------------------------
st.set_page_config(page_title="Smarter Power Ranking (NFL Elo)", layout="wide")

# ----------------------- PARAMS (tunable) ------------------------------------
YEARS: Iterable[int] = range(2012, 2026)   # app exposes 2012+ in UI; includes 2025 preseason
BASE_RATING       = 1500.0
LEAGUE_MEAN       = 1500.0
SEASON_REGRESSION = 0.40
K_BASE            = 20.0
K_PLAYOFF_MULT    = 1.2
HOME_FIELD_ELO    = 40.0
RDIFF_SCALE       = 400.0
TIE_VALUE         = 0.5
NEUTRAL_IF_SUPER_BOWL = True

# Market blend knobs (app will use moneylines when available; otherwise Elo-only)
MARKET_WEIGHT = 0.30
MARKET_SIGMA  = 13.5
SPREAD_SIGN   = 1

# ---------- Light branding & team logos ----------
TEAM_ALIASES = {
    "LA": "LAR",   # Rams
    "STL": "LAR",  # old Rams
    "SD": "LAC",   # old Chargers
    "OAK": "LV",   # old Raiders
    "WSH": "WAS",  # Commanders
    "JAC": "JAX",  # Jaguars
}

def canon(team: str) -> str:
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return ""
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)
def inject_brand_css():
    st.markdown("""
    <style>
      /* Base background */
      .stApp { background: linear-gradient(180deg,#0f1115 0%, #0b0d12 100%); }

      /* Force strong, non-faded text everywhere */
      [data-testid="stAppViewContainer"], .stMarkdown, .stMarkdown p, .stMarkdown li,
      .stMarkdown ul, .stMarkdown ol, .stMarkdown blockquote {
        color: #e6eaf0 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #e6eaf0 !important; /* iOS */
      }
      h1, h2, h3, h4 {
        color: #ffffff !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #ffffff !important; /* iOS */
        letter-spacing: .2px;
      }

      /* Streamlit header/sidebar transparency tweaks */
      [data-testid="stHeader"] { background: transparent !important; backdrop-filter: none !important; }
      [data-testid="stSidebar"] { background: #0d1117 !important; }

      /* Dataframes slightly tighter */
      div[data-testid="stDataFrame"] div[role="row"] { font-size: 14px; }

      /* Buttons */
      .stButton>button {
        border-radius: 10px; padding: .5rem 1rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,.10); color:#e6eaf0;
      }

      /* iPhone/small screens: remove any chance of dim/overlay feel */
      @media (max-width: 768px) {
        .stApp, [data-testid="stAppViewContainer"], .stMarkdown, h1,h2,h3,h4,p,li {
          filter: none !important;
          opacity: 1 !important;
        }
        .block-container { padding: 1rem .75rem !important; }
      }
    </style>
    """, unsafe_allow_html=True)

TEAM_LOGO_URL = {
    "ARI":"https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "ATL":"https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "BAL":"https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "BUF":"https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "CAR":"https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "CHI":"https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "CIN":"https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "CLE":"https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "DAL":"https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "DEN":"https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "DET":"https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "GB":"https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU":"https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND":"https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX":"https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC":"https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LV":"https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "LAC":"https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LAR":"https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "MIA":"https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN":"https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE":"https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO":"https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG":"https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ":"https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI":"https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT":"https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SF":"https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "SEA":"https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "TB":"https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN":"https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS":"https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png",
    "WSH":"https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png",
}

def team_logo_url(team: str) -> str:
    return TEAM_LOGO_URL.get(canon(team), "")

def team_conf(team: str) -> str:
    return TEAM_TO_CONF_DIV.get(canon(team), ("",""))[0]

def team_div(team: str) -> str:
    return TEAM_TO_CONF_DIV.get(canon(team), ("",""))[1]


# ============================================================================
# DATA LOAD
# ============================================================================
@st.cache_data(show_spinner=True)
def load_schedules(years: Iterable[int]) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        st.error(
            "nfl_data_py is required. On your server, install with:\n\n"
            "    pip install --upgrade nfl_data_py\n\n"
            f"Import error: {e}"
        )
        st.stop()

    raw = nfl.import_schedules(list(years))

    # Keep only finished games for historical Elo
    df = raw[raw["home_score"].notna() & raw["away_score"].notna()].copy()

    # Normalize + CANONICALIZE
    df["date"]        = pd.to_datetime(df["gameday"])
    df["season"]      = df["season"].astype(int)
    df["week"]        = df["week"].astype(int)
    df["home_team"]   = df["home_team"].astype(str).map(canon)
    df["away_team"]   = df["away_team"].astype(str).map(canon)
    df["home_score"]  = df["home_score"].astype(int)
    df["away_score"]  = df["away_score"].astype(int)
    df["game_type"]   = df["game_type"].astype(str)
    df["is_playoff"]  = (df["game_type"] != "REG").astype(int)

    # Moneylines (market blend optional)
    for col in ("home_moneyline", "away_moneyline"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA

    # Optional spread -> normalize to home_spread
    df["home_spread"] = (
        pd.to_numeric(df["spread_line"], errors="coerce") if "spread_line" in df.columns else pd.NA
    )

    # Rest days
    df.rename(columns={"home_rest": "home_rest_days", "away_rest": "away_rest_days"}, inplace=True)

    # Neutral-site (and force SB neutral if configured)
    if "location" in df.columns:
        loc_is_neutral = df["location"].astype(str).str.contains("neutral", case=False, na=False)
    else:
        loc_is_neutral = pd.Series(False, index=df.index)
    sb_neutral = df["game_type"].str.upper().eq("SB") if NEUTRAL_IF_SUPER_BOWL else pd.Series(False, index=df.index)
    df["neutral_site"] = (loc_is_neutral | sb_neutral).astype(int)

    df = df.sort_values(["date", "season", "week"]).reset_index(drop=True)
    return df[[
        "date","season","week","game_type","is_playoff",
        "home_team","away_team","home_score","away_score",
        "home_rest_days","away_rest_days","neutral_site",
        "home_moneyline","away_moneyline","home_spread"
    ]]

@st.cache_data(show_spinner=True)
def load_future_schedule_2025() -> pd.DataFrame:
    """Load 2025 regular-season schedule (no scores). Used for the simulator UI."""
    try:
        import nfl_data_py as nfl
    except Exception:
        return pd.DataFrame()
    df = nfl.import_schedules([2025]).copy()
    df["season"] = df["season"].astype(int)
    df["week"]   = pd.to_numeric(df["week"], errors="coerce")
    df = df[(df["game_type"] == "REG") & df["week"].notna()]
    df["week"]   = df["week"].astype(int)
    df = df[["season","week","gameday","home_team","away_team","location"]].sort_values(["week","gameday"])
    df["home_team"] = df["home_team"].map(canon)
    df["away_team"] = df["away_team"].map(canon)
    # Mark neutral site if provided
    if "location" in df.columns:
        df["neutral_site"] = df["location"].astype(str).str.contains("neutral", case=False, na=False).astype(int)
    else:
        df["neutral_site"] = 0
    return df.reset_index(drop=True)

# ============================================================================
# ELO HELPERS
# ============================================================================
def mov_multiplier(mov: float, elo_diff_no_bumps: float) -> float:
    if mov <= 0:
        return 1.0
    return math.log(mov + 1.0) * (2.2 / ((abs(elo_diff_no_bumps) / 1000.0) + 2.2))

def rest_elo_bump(rest_days) -> float:
    if pd.isna(rest_days):
        return 0.0
    try:
        r = float(rest_days)
    except Exception:
        return 0.0
    if r <= 3:
        return -15.0
    if r >= 8:
        return +25.0
    return 0.0

def expected_prob(elo_team: float, elo_opp: float) -> float:
    dr = (elo_team - elo_opp) / RDIFF_SCALE
    return 1.0 / (1.0 + 10.0 ** (-dr))

def regress_to_mean(prev: Dict[str, float], mean: float, frac: float) -> Dict[str, float]:
    return {t: mean * frac + r * (1.0 - frac) for t, r in prev.items()}

def outcome_points(home_pts: int, away_pts: int) -> Tuple[float, float, int]:
    if home_pts > away_pts: return 1.0, 0.0, home_pts - away_pts
    if home_pts < away_pts: return 0.0, 1.0, away_pts - home_pts
    return TIE_VALUE, TIE_VALUE, 0

# Market helpers (moneyline preferred, spread fallback)
from math import erf, sqrt
def prob_from_american_odds(odds) -> float | None:
    if pd.isna(odds): return None
    try: o = float(odds)
    except Exception: return None
    if o < 0:  return (-o) / ((-o) + 100.0)
    else:      return 100.0 / (o + 100.0)

def devig_two_way(p_home_raw: float | None, p_away_raw: float | None):
    if p_home_raw is None or p_away_raw is None: return None, None
    s = p_home_raw + p_away_raw
    if s <= 0: return None, None
    return p_home_raw / s, p_away_raw / s

def market_home_winprob_from_moneylines(home_ml, away_ml) -> float | None:
    ph = prob_from_american_odds(home_ml)
    pa = prob_from_american_odds(away_ml)
    ph_fair, pa_fair = devig_two_way(ph, pa)
    return ph_fair

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def market_home_winprob_from_spread(home_spread: float, sigma_pts: float, sign: int) -> float | None:
    if pd.isna(home_spread): return None
    try: s = float(home_spread)
    except Exception: return None
    return _norm_cdf((sign * s) / sigma_pts)

def blend_probs(p_elo: float, p_mkt: float | None, w: float) -> float:
    return p_elo if p_mkt is None else (1.0 - w) * p_elo + w * p_mkt

def infer_spread_sign(df: pd.DataFrame) -> int:
    if "home_spread" not in df.columns: return 1
    sub = df.dropna(subset=["home_spread"]).copy()
    if len(sub) < 200: return 1
    home_win = (sub["home_score"] > sub["away_score"]).astype(int)
    corr = sub["home_spread"].corr(home_win)
    return 1 if corr > 0 else -1

# ============================================================================
# CORE ELO + SNAPSHOTS
# ============================================================================
@st.cache_data(show_spinner=True)
def compute_elo_with_snapshots(
    schedules: pd.DataFrame,
    market_weight: float = MARKET_WEIGHT,
    market_sigma: float = MARKET_SIGMA,
    spread_sign: int = SPREAD_SIGN
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      game_log: per-game updates
      weekly_snapshots: ratings per (season, week, team) after all games of that week (only teams that played)
      preseason_snapshots: ratings per (season, team) AFTER regression, before Week 1 (week=0)
    """
    ratings: Dict[str, float] = {}
    last_season = None
    out_rows: List[dict] = []
    preseason_rows: List[dict] = []

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = BASE_RATING
        return ratings[team]

    games = schedules.sort_values(["date","season","week"]).reset_index(drop=True)

    for _, g in games.iterrows():
        season = int(g["season"])
        week   = int(g["week"])
        date   = pd.to_datetime(g["date"]).date()
        h, a   = str(g["home_team"]), str(g["away_team"])
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        neutral   = bool(g["neutral_site"])
        is_playoff = bool(g["is_playoff"])

        # New season? regression + preseason snapshot
        if last_season is None:
            last_season = season
            if len(ratings):
                ratings.update(regress_to_mean(ratings, LEAGUE_MEAN, SEASON_REGRESSION))
            for t, r in ratings.items():
                preseason_rows.append({"season": season, "team": t, "rating": r})
        elif season != last_season:
            ratings.update(regress_to_mean(ratings, LEAGUE_MEAN, SEASON_REGRESSION))
            last_season = season
            for t, r in ratings.items():
                preseason_rows.append({"season": season, "team": t, "rating": r})

        # Pre-game ratings
        home_pre = get_rating(h)
        away_pre = get_rating(a)

        # Bumps
        h_rest_bump = rest_elo_bump(g.get("home_rest_days"))
        a_rest_bump = rest_elo_bump(g.get("away_rest_days"))
        hfa = 0.0 if neutral else HOME_FIELD_ELO

        home_adj = home_pre + hfa + h_rest_bump
        away_adj = away_pre + a_rest_bump

        # Expectations (market blend if available)
        E_home_elo = expected_prob(home_adj, away_adj)
        E_home_mkt = market_home_winprob_from_moneylines(g.get("home_moneyline"), g.get("away_moneyline"))
        if E_home_mkt is None and not pd.isna(g.get("home_spread", pd.NA)):
            E_home_mkt = market_home_winprob_from_spread(g.get("home_spread"), MARKET_SIGMA, spread_sign)
        E_home = blend_probs(E_home_elo, E_home_mkt, market_weight)
        E_away = 1.0 - E_home

        # Outcome & K
        S_home, S_away, mov = outcome_points(hs, as_)
        K = K_BASE * (K_PLAYOFF_MULT if is_playoff else 1.0)
        elo_diff_for_mov = (home_pre + (0.0 if neutral else HOME_FIELD_ELO)) - away_pre
        mult = mov_multiplier(mov, elo_diff_for_mov)
        K_adj = K * mult

        home_post = home_pre + K_adj * (S_home - E_home)
        away_post = away_pre + K_adj * (S_away - E_away)

        ratings[h] = home_post
        ratings[a] = away_post

        out_rows.append({
            "date": date.isoformat(),
            "season": season, "week": week,
            "home_team": h, "away_team": a,
            "home_score": hs, "away_score": as_,
            "neutral_site": int(neutral), "is_playoff": int(is_playoff),
            "home_rating_pre": round(home_pre, 2), "away_rating_pre": round(away_pre, 2),
            "home_rating_pregame": round(home_adj, 2), "away_rating_pregame": round(away_adj, 2),
            "E_home_elo": round(E_home_elo, 4),
            "E_home_mkt": (round(E_home_mkt, 4) if E_home_mkt is not None else None),
            "E_home": round(E_home, 4), "E_away": round(E_away, 4),
            "mov": mov, "K_adj": round(K_adj, 4),
            "home_rating_post": round(home_post, 2), "away_rating_post": round(away_post, 2),
        })

    # Per-week snapshots (teams that played)
    game_log = pd.DataFrame(out_rows)
    wk = pd.concat([
        game_log[["season","week","home_team","home_rating_post"]].rename(columns={"home_team":"team","home_rating_post":"rating"}),
        game_log[["season","week","away_team","away_rating_post"]].rename(columns={"away_team":"team","away_rating_post":"rating"}),
    ], ignore_index=True)
    wk = (
        wk.sort_values(["season","week"])
          .groupby(["season","week","team"], as_index=False)
          .last()
          .sort_values(["season","week","rating"], ascending=[True,True,False])
          .reset_index(drop=True)
    )

    preseason = pd.DataFrame(preseason_rows).drop_duplicates(["season","team"], keep="last") if preseason_rows else pd.DataFrame(columns=["season","team","rating"])
    return game_log, wk, preseason

# --- make a "full" weekly table that carries ratings forward so all 32 teams show every week
def build_full_weekly(wk: pd.DataFrame, preseason: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s, sub in wk.groupby("season"):
        teams = sorted(sub["team"].unique())
        weeks = sorted(sub["week"].unique())
        if not weeks:
            continue
        wmin, wmax = min(weeks), max(weeks)
        grid = pd.MultiIndex.from_product([range(wmin, wmax+1), teams], names=["week","team"]).to_frame(index=False)
        grid["season"] = s
        merged = grid.merge(sub[["season","week","team","rating"]], on=["season","week","team"], how="left")
        merged = merged.sort_values(["team","week"])
        merged["rating"] = merged.groupby("team")["rating"].ffill()
        if len(preseason):
            pre = preseason.query("season == @s")[["team","rating"]].rename(columns={"rating":"pre"})
            merged = merged.merge(pre, on="team", how="left")
            merged["rating"] = merged["rating"].fillna(merged["pre"])
            merged = merged.drop(columns=["pre"])
        merged = merged[["season","week","team","rating"]]
        out.append(merged)
    if not out:
        return pd.DataFrame(columns=["season","week","team","rating"])
    full = pd.concat(out, ignore_index=True)
    full = full.sort_values(["season","week","rating"], ascending=[True,True,False]).reset_index(drop=True)
    return full

# ============================================================================
# SIMULATION HELPERS
# ============================================================================
def elo_update_one(
    ratings: Dict[str, float],
    home_team: str, away_team: str,
    home_win: bool,
    neutral_site: bool = False,
    mov_points: int = 3,
    is_playoff: bool = False,
) -> None:
    """Apply one simulated game update in-place on ratings dict.
    If mov_points == 0, treat as a tie (S_home = S_away = 0.5) and ignore home_win.
    """
    home_team, away_team = canon(home_team), canon(away_team)

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = BASE_RATING
        return ratings[team]

    home_pre = get_rating(home_team)
    away_pre = get_rating(away_team)

    hfa = 0.0 if neutral_site else HOME_FIELD_ELO
    home_adj = home_pre + hfa
    away_adj = away_pre

    E_home = expected_prob(home_adj, away_adj)

    if mov_points == 0:
        S_home = 0.5
    else:
        S_home = 1.0 if home_win else 0.0
    S_away = 1.0 - S_home

    K = K_BASE * (K_PLAYOFF_MULT if is_playoff else 1.0)
    elo_diff_for_mov = (home_pre + (0.0 if neutral_site else HOME_FIELD_ELO)) - away_pre
    mult = mov_multiplier(mov_points, elo_diff_for_mov)
    K_adj = K * mult

    home_post = home_pre + K_adj * (S_home - E_home)
    away_post = away_pre + K_adj * (S_away - (1.0 - E_home))

    ratings[home_team] = home_post
    ratings[away_team] = away_post

# ============================================================================
# LOAD + PRECOMPUTE
# ============================================================================
schedules = load_schedules(YEARS)
SPREAD_SIGN = infer_spread_sign(schedules)

game_log, weekly_snapshots, preseason_snapshots = compute_elo_with_snapshots(
    schedules, MARKET_WEIGHT, MARKET_SIGMA, SPREAD_SIGN
)

weekly_full = build_full_weekly(weekly_snapshots, preseason_snapshots)  # <-- always 32 teams per week
future_2025 = load_future_schedule_2025()  # may be empty if dataset not present

# For the simulator, build preseason 2025 ratings (make sure we have all 32 teams)
def build_preseason_2025(weekly: pd.DataFrame, preseason: pd.DataFrame, future_sched: pd.DataFrame) -> Dict[str, float]:
    teams_2025 = sorted(pd.unique(pd.concat([future_sched["home_team"], future_sched["away_team"]], ignore_index=True))) if len(future_sched) else []
    # try preseason in table
    p25 = preseason.query("season == 2025")
    if len(p25):
        base = dict(zip(p25["team"], p25["rating"]))
    else:
        # regress from last finished season
        last_season = int(weekly["season"].max())
        last_week   = int(weekly.query("season == @last_season")["week"].max())
        end = weekly.query("season == @last_season and week == @last_week")
        base = dict(zip(end["team"], end["rating"]))
        base = regress_to_mean(base, LEAGUE_MEAN, SEASON_REGRESSION)
    if teams_2025:
        # ensure every 2025 team has a value
        for t in teams_2025:
            if t not in base:
                base[t] = BASE_RATING
    base = {canon(k): float(v) for k, v in base.items()}
    if teams_2025:
        teams_2025 = [canon(t) for t in teams_2025]
        for t in teams_2025:
            base.setdefault(t, BASE_RATING)
    return base

PRESEASON_2025 = build_preseason_2025(weekly_full, preseason_snapshots, future_2025)

# ============================================================================
# SMALL UTILS
# ============================================================================
def label_weeks_for_season(season: int) -> dict[int,str]:
    """Return a dict week->label with playoff tags for the last four weeks."""
    weeks = sorted(weekly_full.query("season == @season")["week"].unique().tolist())
    labels = {w: f"Week {w}" for w in weeks}
    if len(weeks) >= 4:
        labels[weeks[-1]] = f"Week {weeks[-1]} (Super Bowl)"
        labels[weeks[-2]] = f"Week {weeks[-2]} (Conference Championship)"
        labels[weeks[-3]] = f"Week {weeks[-3]} (Divisional Round)"
        labels[weeks[-4]] = f"Week {weeks[-4]} (Wild Card)"
    return labels

def render_rank_table(df: pd.DataFrame, title: str):
    df = df.copy()
    df["team"] = df["team"].map(canon)
    df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    df["SmartScore"] = df["rating"].round(0).astype(int)

    # CSS (crisp text & images)
    st.markdown("""
    <style>
      .tbl { width:100%; border-collapse:separate; border-spacing:0 8px; }
      .tbl th { font-size:13px; color:#cfd7e0; text-align:left; padding:4px 10px; }
      .tbl td { padding:10px 12px; background:#0b1220; border:1px solid #2a3344; }
      .tbl td:first-child { border-radius:12px 0 0 12px; }
      .tbl td:last-child  { border-radius:0 12px 12px 0; }
      .logo-img { vertical-align:middle; image-rendering:-webkit-optimize-contrast; }
      .teamcell { display:flex; gap:10px; align-items:center; font-weight:600; color:#e5e7eb; }
      .rk, .score { width:80px; color:#e5e7eb; font-weight:700; }
      .rk { text-align:center; }
      .score { text-align:right; }
      .hdr-wrap { margin: 6px 0 4px; }
    </style>
    """, unsafe_allow_html=True)

    # Build rows
    rows = []
    for _, r in df.iterrows():
        team = r["team"]
        rows.append(
            f"<tr>"
            f"<td class='rk'>{int(r['Rank'])}</td>"
            f"<td><div class='teamcell'>{_logo_img_tag(team, 22)}<span>{team}</span></div></td>"
            f"<td class='score'>{int(r['SmartScore'])}</td>"
            f"</tr>"
        )

    st.subheader(title)
    st.markdown(
        "<table class='tbl'>"
        "<thead><tr><th class='rk'>Rank</th><th>Team</th><th class='score'>Smart Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True
    )



# ============================================================================
# DIVISIONS / STANDINGS / SEEDING HELPERS
# ============================================================================

# Team -> (Conference, Division). Includes both WAS and WSH just in case.
TEAM_TO_CONF_DIV = {
    # AFC East
    "BUF": ("AFC","East"), "MIA": ("AFC","East"), "NE": ("AFC","East"), "NYJ": ("AFC","East"),
    # AFC North
    "BAL": ("AFC","North"), "CIN": ("AFC","North"), "CLE": ("AFC","North"), "PIT": ("AFC","North"),
    # AFC South
    "HOU": ("AFC","South"), "IND": ("AFC","South"), "JAX": ("AFC","South"), "TEN": ("AFC","South"),
    # AFC West
    "DEN": ("AFC","West"), "KC": ("AFC","West"), "LAC": ("AFC","West"), "LV": ("AFC","West"),
    # NFC East
    "DAL": ("NFC","East"), "NYG": ("NFC","East"), "PHI": ("NFC","East"), "WAS": ("NFC","East"), "WSH": ("NFC","East"),
    # NFC North
    "CHI": ("NFC","North"), "DET": ("NFC","North"), "GB": ("NFC","North"), "MIN": ("NFC","North"),
    # NFC South
    "ATL": ("NFC","South"), "CAR": ("NFC","South"), "NO": ("NFC","South"), "TB": ("NFC","South"),
    # NFC West
    "ARI": ("NFC","West"), "LAR": ("NFC","West"), "SEA": ("NFC","West"), "SF": ("NFC","West"),
}


def _ensure_sim_results():
    if "sim_results" not in st.session_state:
        # each row: dict(season, week, home, away, winner, tie, mov, neutral, is_playoff, round)
        st.session_state.sim_results = []

def _results_df():
    _ensure_sim_results()
    return pd.DataFrame(st.session_state.sim_results)

def _win_tie_to_pct(w: int, l: int, t: int) -> float:
    denom = w + l + t
    return 0.0 if denom == 0 else (w + 0.5*t) / denom

def _head_to_head_pct(results: pd.DataFrame, team: str, others: set[str]) -> float | None:
    if results.empty: return None
    mask = ((results["home"] == team) & (results["away"].isin(others))) | \
           ((results["away"] == team) & (results["home"].isin(others)))
    sub = results.loc[mask]
    if sub.empty: return None
    w = ((sub["winner"] == "HOME") & (sub["home"] == team) |
         (sub["winner"] == "AWAY") & (sub["away"] == team)).sum()
    l = ((sub["winner"] == "HOME") & (sub["away"] == team) |
         (sub["winner"] == "AWAY") & (sub["home"] == team)).sum()
    t = (sub["winner"] == "TIE").sum()
    return _win_tie_to_pct(int(w), int(l), int(t))

def _common_games_pct(results: pd.DataFrame, team: str, common_opps: set[str]) -> float | None:
    if results.empty or len(common_opps) < 4: return None
    mask = ((results["home"] == team) & (results["away"].isin(common_opps))) | \
           ((results["away"] == team) & (results["home"].isin(common_opps)))
    sub = results.loc[mask]
    if sub.empty: return None
    w = ((sub["winner"] == "HOME") & (sub["home"] == team) |
         (sub["winner"] == "AWAY") & (sub["away"] == team)).sum()
    l = ((sub["winner"] == "HOME") & (sub["away"] == team) |
         (sub["winner"] == "AWAY") & (sub["home"] == team)).sum()
    t = (sub["winner"] == "TIE").sum()
    return _win_tie_to_pct(int(w), int(l), int(t))

def _opp_record_excluding(results: pd.DataFrame, exclude_team: str) -> dict[str, tuple[int,int,int]]:
    """Return W,L,T for every team with all games *excluding* those vs exclude_team."""
    rec = {t: [0,0,0] for t in pd.unique(pd.concat([results["home"], results["away"]]))}
    for _,r in results.iterrows():
        if r["home"] == exclude_team or r["away"] == exclude_team:
            continue
        if r["winner"] == "TIE":
            rec[r["home"]][2] += 1; rec[r["away"]][2] += 1
        elif r["winner"] == "HOME":
            rec[r["home"]][0] += 1; rec[r["away"]][1] += 1
        elif r["winner"] == "AWAY":
            rec[r["away"]][0] += 1; rec[r["home"]][1] += 1
    return {k: tuple(v) for k,v in rec.items()}

def render_seeds_pretty(seeds_all: dict[str, dict[int, str]]) -> None:
    st.markdown("""
    <style>
      .seedtbl { width:100%; border-collapse:separate; border-spacing:0 6px; }
      .seedtbl th { color:#cfd7e0; text-align:left; font-size:13px; padding:4px 8px; }
      .seedtbl td { background:#0b1220; border:1px solid #2a3344; padding:8px 10px; }
      .seedtbl td:first-child { width:54px; text-align:center; font-weight:700; color:#e5e7eb; border-radius:12px 0 0 12px; }
      .seedtbl td:last-child { border-radius:0 12px 12px 0; }
      .seed-row { display:flex; align-items:center; gap:10px; color:#e5e7eb; font-weight:600; }
      .logo-img { image-rendering:-webkit-optimize-contrast; }
    </style>
    """, unsafe_allow_html=True)

    cA, cN = st.columns(2)
    for conf, col in zip(["AFC", "NFC"], [cA, cN]):
        with col:
            st.markdown(f"#### {conf} Seeds")
            seeds = seeds_all[conf]
            order = sorted(seeds.keys())
            rows = []
            for s in order:
                t = seeds[s]
                rows.append(
                    f"<tr><td>#{s}</td><td>"
                    f"<div class='seed-row'>{_logo_img_tag(t, 22)}<span>{t}</span></div>"
                    f"</td></tr>"
                )
            st.markdown(
                "<table class='seedtbl'><thead><tr><th>Seed</th><th>Team</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>",
                unsafe_allow_html=True
            )


def build_wc_pairs() -> list[tuple[int, int]]:
    # higher seed hosts (lower number = higher seed)
    return [(2, 7), (3, 6), (4, 5)]

def reseed_divisional_from_survivors(survivor_seeds: list[int]) -> list[tuple[int, int]]:
    """survivor_seeds includes the #1 seed; return (home_seed, away_seed) pairings."""
    s = sorted(survivor_seeds)
    assert 1 in s and len(s) == 4
    low = min([x for x in s if x != 1])
    others = [x for x in s if x not in (1, low)]
    hi_other, lo_other = max(others), min(others)
    return [(1, low), (hi_other, lo_other)]

def _pct_from_triple(wlt: tuple[int,int,int]) -> float:
    return _win_tie_to_pct(*wlt)





def compute_team_records(teams: list[str], results: pd.DataFrame) -> pd.DataFrame:
    teams = [canon(t) for t in teams]  # ensure canonical list

    if results.empty:
        base = []
        for t in teams:
            base.append({"team": t, "conf": team_conf(t), "div": team_div(t),
                         "W":0,"L":0,"T":0,"PCT":0.0,
                         "DivW":0,"DivL":0,"DivT":0,"DivPct":0.0,
                         "ConfW":0,"ConfL":0,"ConfT":0,"ConfPct":0.0,
                         "NP":0,"NPC":0,"SOS":0.0,"SOV":0.0,
                         "Opponents": set(), "BeatenOpps": set()})
        return pd.DataFrame(base)

    reg = results[results["round"]=="REG"].copy()
    reg["home"] = reg["home"].map(canon)
    reg["away"] = reg["away"].map(canon)
    base = {t: {
        "team": t, "conf": team_conf(t), "div": team_div(t),
        "W":0,"L":0,"T":0, "DivW":0,"DivL":0,"DivT":0, "ConfW":0,"ConfL":0,"ConfT":0,
        "NP":0, "NPC":0, "Opponents": set(), "BeatenOpps": set()
    } for t in teams}

    for _,r in reg.iterrows():
        h,a = r["home"], r["away"]
        mov = int(r["mov"])
        same_conf = team_conf(h) == team_conf(a)
        same_div  = team_div(h) == team_div(a) and same_conf
        base[h]["Opponents"].add(a); base[a]["Opponents"].add(h)

        if r["winner"] == "TIE":
            base[h]["T"] += 1; base[a]["T"] += 1
            if same_conf: base[h]["ConfT"] += 1; base[a]["ConfT"] += 1
            if same_div:  base[h]["DivT"]  += 1; base[a]["DivT"]  += 1
            # NP stays same for tie (mov may be 0)
        elif r["winner"] == "HOME":
            base[h]["W"] += 1; base[a]["L"] += 1
            base[h]["NP"] += mov; base[a]["NP"] -= mov
            if same_conf:
                base[h]["ConfW"] += 1; base[a]["ConfL"] += 1
                base[h]["NPC"] += mov; base[a]["NPC"] -= mov
            if same_div:
                base[h]["DivW"] += 1; base[a]["DivL"] += 1
            base[h]["BeatenOpps"].add(a)
        elif r["winner"] == "AWAY":
            base[a]["W"] += 1; base[h]["L"] += 1
            base[a]["NP"] += mov; base[h]["NP"] -= mov
            if same_conf:
                base[a]["ConfW"] += 1; base[h]["ConfL"] += 1
                base[a]["NPC"] += mov; base[h]["NPC"] -= mov
            if same_div:
                base[a]["DivW"] += 1; base[h]["DivL"] += 1
            base[a]["BeatenOpps"].add(h)

    rows = []
    for t, d in base.items():
        d["PCT"] = _win_tie_to_pct(d["W"], d["L"], d["T"])
        d["DivPct"]  = _win_tie_to_pct(d["DivW"], d["DivL"], d["DivT"])
        d["ConfPct"] = _win_tie_to_pct(d["ConfW"], d["ConfL"], d["ConfT"])
        rows.append(d)
    df = pd.DataFrame(rows)

    # Strength of schedule (SOS) & strength of victory (SOV)
    for idx, r in df.iterrows():
        team = r["team"]
        opps = r["Opponents"]
        beaten = r["BeatenOpps"]
        if len(opps) == 0:
            df.at[idx, "SOS"] = 0.0
        else:
            opp_rec = _opp_record_excluding(reg, team)
            w=l=t=0
            for o in opps:
                if o in opp_rec:
                    ow,ol,ot = opp_rec[o]; w+=ow; l+=ol; t+=ot
            df.at[idx, "SOS"] = _win_tie_to_pct(w,l,t)
        if len(beaten) == 0:
            df.at[idx, "SOV"] = 0.0
        else:
            opp_rec = _opp_record_excluding(reg, team)
            w=l=t=0
            for o in beaten:
                if o in opp_rec:
                    ow,ol,ot = opp_rec[o]; w+=ow; l+=ol; t+=ot
            df.at[idx, "SOV"] = _win_tie_to_pct(w,l,t)
    return df

def _rank_with_criteria(tied: list[str], rec: pd.DataFrame, results: pd.DataFrame, criteria: list[str], scope: str) -> list[str]:
    """
    Generic tiebreak sorter. criteria is a list of keys:
    'H2H', 'DIV', 'COMMON', 'CONF', 'SOV', 'SOS', 'NPC', 'NP'
    scope: 'division' or 'conference' (used for COMMON opponent sets)
    """
    group = tied[:]
    if len(group) <= 1:
        return group

    # Precompute common-opponent sets if needed
    common_opps: set[str] | None = None
    if "COMMON" in criteria:
        opp_sets = []
        for t in group:
            row = rec.loc[rec["team"]==t].iloc[0]
            opp_sets.append(set(row["Opponents"]))
        if opp_sets:
            common_opps = set.intersection(*opp_sets)

    def metric(t: str, key: str) -> float | None:
        row = rec.loc[rec["team"]==t].iloc[0]
        if key == "H2H":
            return _head_to_head_pct(results, t, set(group) - {t})
        if key == "DIV":
            return float(row["DivPct"])
        if key == "CONF":
            return float(row["ConfPct"])
        if key == "COMMON":
            return _common_games_pct(results, t, common_opps or set())
        if key == "SOV":
            return float(row["SOV"])
        if key == "SOS":
            return float(row["SOS"])
        if key == "NPC":
            return float(row["NPC"])
        if key == "NP":
            return float(row["NP"])
        return None

    remaining = group
    for key in criteria:
        # compute numeric metric; None -> treat as -inf so it won't separate
        vals = {t: (metric(t, key) if metric(t, key) is not None else -10.0) for t in remaining}
        # sort by metric desc
        sorted_group = sorted(remaining, key=lambda t: (vals[t], t), reverse=True)
        # split by metric
        tiers: list[list[str]] = []
        cur = [sorted_group[0]]
        for a,b in zip(sorted_group, sorted_group[1:]+[None]):
            if b is None: break
            if abs(vals[a] - vals[b]) < 1e-12:
                cur.append(b)
            else:
                tiers.append(cur); cur = [b]
        tiers.append(cur)
        # if all ties unresolved, continue to next criterion
        if all(len(tier) > 1 for tier in tiers):
            remaining = [t for tier in tiers for t in tier]
            continue
        # rebuild list with resolved tiers first
        new_order = []
        for tier in tiers:
            if len(tier) == 1:
                new_order.extend(tier)
            else:
                # keep the tied tier together for next criterion
                new_order.extend(tier)
        remaining = new_order
        # if fully resolved
        if len(set(vals.values())) == len(vals):
            break

    # final fallback = alphabetical (stable)
    return sorted(remaining, key=lambda t: (remaining.index(t), t))

def rank_division(teams: list[str], rec: pd.DataFrame, results: pd.DataFrame) -> list[str]:
    # Start by win% desc, then apply NFL division tiebreakers
    teams_sorted = sorted(teams, key=lambda t: (rec.loc[rec["team"]==t,"PCT"].iloc[0], t), reverse=True)
    # Partition by identical win%
    from itertools import groupby
    out = []
    for pct, grp in groupby(teams_sorted, key=lambda t: rec.loc[rec["team"]==t,"PCT"].iloc[0]):
        tie_group = list(grp)
        if len(tie_group) == 1:
            out.extend(tie_group)
        else:
            # Official order (close to NFL): H2H, DIV, COMMON (>=4), CONF, SOV, SOS, NP
            criteria = ["H2H","DIV","COMMON","CONF","SOV","SOS","NP"]
            out.extend(_rank_with_criteria(tie_group, rec, results, criteria, scope="division"))
    return out

def rank_conference(teams: list[str], rec: pd.DataFrame, results: pd.DataFrame) -> list[str]:
    # Sort by overall win%, then break ties via wild-card procedure
    teams_sorted = sorted(teams, key=lambda t: (rec.loc[rec["team"]==t,"PCT"].iloc[0], t), reverse=True)
    from itertools import groupby
    out = []
    for pct, grp in groupby(teams_sorted, key=lambda t: rec.loc[rec["team"]==t,"PCT"].iloc[0]):
        tie_group = list(grp)
        if len(tie_group) == 1:
            out.extend(tie_group)
        else:
            # Wild-card/NFL conference seeding tiebreak (approx): H2H, CONF, COMMON, SOV, SOS, NPC, NP
            criteria = ["H2H","CONF","COMMON","SOV","SOS","NPC","NP"]
            out.extend(_rank_with_criteria(tie_group, rec, results, criteria, scope="conference"))
    return out

def compute_conference_seeds(rec: pd.DataFrame, results: pd.DataFrame, conference: str) -> dict[int,str]:
    conf_teams = rec.loc[rec["conf"]==conference, "team"].tolist()
    # Division winners (rank within each division)
    seeds = {}
    div_winners = []
    for division in ["East","North","South","West"]:
        div_teams = rec[(rec["conf"]==conference)&(rec["div"]==division)]["team"].tolist()
        order = rank_division(div_teams, rec, results)
        div_winners.append(order[0])
    # Seed 1-4 by conference ranking among the four winners
    winners_order = rank_conference(div_winners, rec, results)
    for i, t in enumerate(winners_order, start=1):
        seeds[i] = t
    # Wild cards: rank remaining conf teams and take top 3 for seeds 5-7
    remaining = [t for t in conf_teams if t not in winners_order]
    wc_order = rank_conference(remaining, rec, results)[:3]
    for j, t in enumerate(wc_order, start=5):
        seeds[j] = t
    return seeds

def build_all_seeds(rec: pd.DataFrame, results: pd.DataFrame) -> dict[str, dict[int,str]]:
    return {
        "AFC": compute_conference_seeds(rec, results, "AFC"),
        "NFC": compute_conference_seeds(rec, results, "NFC")
    }

def bracket_for_round(seeds_conf: dict[int,str], round_name: str) -> list[tuple[int,int]]:
    """
    Return list of (higher_seed, lower_seed) pairings for a given round.
    Wild Card: (2 v 7), (3 v 6), (4 v 5) – seed 1 bye
    Divisional: reseed, 1 v lowest, other two play
    Conference: winners of divisional (higher vs lower)
    Super Bowl handled outside (AFC champ vs NFC champ; neutral)
    """
    if round_name == "WC":
        return [(2,7),(3,6),(4,5)]
    # For later rounds we'll compute based on survivors at runtime
    return []

def reseed_divisional(survivors: list[int]) -> list[tuple[int,int]]:
    # survivors include 1 plus winners' seeds
    # Lowest remaining plays 1; other two are paired high vs low
    remaining = sorted(survivors)
    assert 1 in remaining
    low = min([s for s in remaining if s != 1])
    others = [s for s in remaining if s not in (1, low)]
    hi_other, lo_other = max(others), min(others)
    return [(1, low), (hi_other, lo_other)]

# ========= SIMPLE BRACKET TABLES (no matplotlib) =========
def render_bracket_tables(seeds_all: dict[str, dict[int, str]], results_df: pd.DataFrame) -> None:
    """
    Bracket-as-tables with exactly 3 rows:
      - AFC (left → right):   [WILD CARD | DIVISIONAL | CONFERENCE]
      - NFC (right → left):   [CONFERENCE | DIVISIONAL | WILD CARD]
      - Super Bowl column in the middle shows matchup + champions.
    """
    import html

    # --- helpers --------------------------------------------------------------
    def conf_champ_from_results(conf: str) -> str | None:
        sub = results_df[results_df["round"] == "CONF"].copy()
        if sub.empty:
            return None
        mask = sub.apply(lambda r: team_conf(r["home"]) == conf and team_conf(r["away"]) == conf, axis=1)
        sub = sub[mask]
        if sub.empty:
            return None
        r = sub.iloc[-1]
        return r["home"] if r["winner"] == "HOME" else r["away"]

    def last_winner(round_code: str, t1: str, t2: str) -> str | None:
        sub = results_df[
            (results_df["round"] == round_code)
            & (
                ((results_df["home"] == t1) & (results_df["away"] == t2))
                | ((results_df["home"] == t2) & (results_df["away"] == t1))
            )
        ]
        if sub.empty:
            return None
        r = sub.iloc[-1]
        return r["home"] if r["winner"] == "HOME" else r["away"]

    def build_side(conf: str) -> dict:
        seeds = seeds_all[conf]
        team_to_seed = {team: s for s, team in seeds.items()}

        # Wild Card (always 3)
        wc_pairs = [(2, 7), (3, 6), (4, 5)]
        wc = []
        for hi, lo in wc_pairs:
            h, a = seeds[hi], seeds[lo]
            wc.append((h, a, last_winner("WC", h, a)))

        # Divisional (reseed if WC winners known)
        winners = [w for _, _, w in wc if isinstance(w, str)]
        surv = [1] + ([team_to_seed[w] for w in winners] if winners else [])
        if len(surv) == 4:
            s = sorted(surv)
            low = min([x for x in s if x != 1])
            others = [x for x in s if x not in (1, low)]
            div_pairs = [(1, low), (max(others), min(others))]
            div = []
            for hi, lo in div_pairs:
                h, a = seeds[hi], seeds[lo]
                div.append((h, a, last_winner("DIV", h, a)))
        else:
            div = [(seeds[1], "TBD", None), ("TBD", "TBD", None)]

        # Conference
        div_winners = [w for _, _, w in div if isinstance(w, str)]
        if len(div_winners) == 2:
            conf_pair = (div_winners[0], div_winners[1])
        else:
            conf_pair = ("TBD", "TBD")
        conf_win = last_winner("CONF", conf_pair[0], conf_pair[1]) if all(isinstance(x, str) for x in conf_pair) else None
        return {"WC": wc, "DIV": div, "CONF": (conf_pair[0], conf_pair[1], conf_win)}

    def seeds_map(conf: str) -> dict[str, int]:
        return {team: seed for seed, team in seeds_all[conf].items()}

    def match_html(conf: str, home: str | None, away: str | None, win: str | None) -> str:
        edge = "#3b82f6" if conf == "AFC" else "#10b981"
        smap = seeds_map(conf)

        def txt(x):
            return "TBD" if not isinstance(x, str) else html.escape(x)

        home_txt, away_txt = txt(home), txt(away)
        sh = smap.get(home, "?") if isinstance(home, str) else "?"
        sa = smap.get(away, "?") if isinstance(away, str) else "?"
        crown = "🏆 " if isinstance(win, str) else ""
        label = f"{crown}#{sa} {away_txt} @ #{sh} {home_txt}"
        return f'<div class="match" style="border-color:{edge}">{label}</div>'

    def three_row_table(conf: str, reverse: bool) -> str:
        side = build_side(conf)
        wc0, wc1, wc2 = side["WC"][0], side["WC"][1], side["WC"][2]
        div0, div1 = side["DIV"][0], side["DIV"][1]
        ch, ca, cw = side["CONF"]

        wc0h = match_html(conf, *wc0)
        wc1h = match_html(conf, *wc1)
        wc2h = match_html(conf, *wc2)
        div0h = match_html(conf, *div0)
        div1h = match_html(conf, *div1)
        confh = match_html(conf, ch, ca, cw)

        if reverse:
            # left→right = [CONF, DIV, WC]
            row1 = [confh, div0h, wc0h]
            row2 = ["", div1h, wc1h]
            row3 = ["", "", wc2h]
            header = "<tr><th>CONFERENCE</th><th>DIVISIONAL</th><th>WILD CARD</th></tr>"
        else:
            # left→right = [WC, DIV, CONF]
            row1 = [wc0h, div0h, confh]
            row2 = [wc1h, div1h, ""]
            row3 = [wc2h, "", ""]
            header = "<tr><th>WILD CARD</th><th>DIVISIONAL</th><th>CONFERENCE</th></tr>"

        def tr(cells):
            return "<tr>" + "".join(
                (f'<td class="box">{c}</td>' if c else '<td class="box empty"></td>') for c in cells
            ) + "</tr>"

        # Build without leading spaces (so Markdown doesn't code-block it)
        return (
            "<table class=\"brkt\">"
            f"<thead>{header}</thead>"
            "<tbody>"
            f"{tr(row1)}{tr(row2)}{tr(row3)}"
            "</tbody>"
            "</table>"
        )

    # --- CSS --------------------------------------------------------------
    st.markdown("""
<style>
  .grid-brkt {
    display: grid;
    grid-template-columns: minmax(560px,1fr) auto minmax(560px,1fr);
    grid-template-rows: 1fr auto 1fr;
    column-gap: 28px;
  }
  .grid-brkt .afc { grid-column: 1; grid-row: 1 / 4; justify-self: start; width: max-content; }
  .grid-brkt .nfc { grid-column: 3; grid-row: 1 / 4; justify-self: end;   width: max-content; }
  .grid-brkt .sb  { grid-column: 2; grid-row: 2; place-self: center; }

  .brkt { border-collapse: separate; border-spacing: 14px 10px; }
  .brkt th {
    color:#d1d5db; font-weight:700; padding:6px 2px; text-align:center;
    font-size:12px; letter-spacing:.4px; white-space:nowrap;
    -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision;
  }
  .brkt td.box { width: 280px; height: 60px; vertical-align: middle; }
  .brkt td.box.empty { background: transparent; }
  .match {
    border: 1.5px solid #334155; border-radius: 14px; background: #0b1220;
    color: #e5e7eb; font-weight: 600; text-align: center; padding: 10px 12px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-size: 14px; -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision;
  }
  .sbtable .box { width: 280px; height: 60px; }
  .grid-brkt .sb { width: 280px; }

  /* === NEW: horizontal scroll wrapper so layout never collapses === */
  .xscroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .xscroll .grid-brkt { min-width: 1240px; } /* 560 + 280 + 560 + gaps */
</style>
""", unsafe_allow_html=True
    )

    # --- SB matchup + champion labels -------------------------------------
    afc_champ = conf_champ_from_results("AFC")
    nfc_champ = conf_champ_from_results("NFC")
    sb_match_label = f"{(nfc_champ or 'TBD')} vs {(afc_champ or 'TBD')}"

    sb_game = results_df[results_df["round"] == "SB"].tail(1)
    champ = None if sb_game.empty else (sb_game.iloc[-1]["home"] if sb_game.iloc[-1]["winner"] == "HOME" else sb_game.iloc[-1]["away"])
    sb_champ_label = f"🏆 {champ}" if isinstance(champ, str) else "TBD"

    afc_html = three_row_table("AFC", reverse=False)
    nfc_html = three_row_table("NFC", reverse=True)

    # --- Layout: AFC | SB | NFC  (no leading spaces in HTML) ---------------
    st.markdown(
        (
            f"<div class=\"xscroll\">"                              # NEW wrapper
            f"<div class=\"grid-brkt\">"                           # existing grid
            f"<div class=\"afc\">"
            f"<h4 style=\"margin:0 0 6px\">AFC</h4>"
            f"{afc_html}"
            f"</div>"
            f"<div class=\"sb\">"
            f"<table class=\"brkt sbtable\">"
            f"<thead><tr><th>SUPER BOWL</th></tr></thead>"
            f"<tbody><tr><td class=\"box\">"
            f"<div class=\"match\" style=\"border-color:#94a3b8\">{sb_match_label}</div>"
            f"</td></tr></tbody>"
            f"</table>"
            f"<div style=\"height:10px\"></div>"
            f"<table class=\"brkt sbtable\">"
            f"<thead><tr><th>CHAMPIONS</th></tr></thead>"
            f"<tbody><tr><td class=\"box\">"
            f"<div class=\"match\" style=\"border-color:#eab308\">{sb_champ_label}</div>"
            f"</td></tr></tbody>"
            f"</table>"
            f"</div>"
            f"<div class=\"nfc\">"
            f"<h4 style=\"margin:0 0 6px\">NFC</h4>"
            f"{nfc_html}"
            f"</div>"
            f"</div>"                                             # end .grid-brkt
            f"</div>"                                             # end .xscroll
        ),
        unsafe_allow_html=True,
    )



def render_division_standings_table(rec: pd.DataFrame, title: str = "Standings (through current week)"):
    st.markdown(f"### {title}")
    st.markdown("""
    <style>
      .divtbl { width:100%; border-collapse:separate; border-spacing:0 6px; }
      .divtbl th { color:#cfd7e0; text-align:left; font-size:12px; padding:4px 6px; white-space:nowrap; }
      .divtbl td { background:#0b1220; border:1px solid #2a3344; padding:8px 8px; font-size:13px; color:#e5e7eb; }
      .divtbl td:first-child { border-radius:12px 0 0 12px; }
      .divtbl td:last-child  { border-radius:0 12px 12px 0; }
      .teamrow { display:flex; align-items:center; gap:8px; font-weight:600; }
      .logo-img { image-rendering:-webkit-optimize-contrast; }
      .num { text-align:right; font-variant-numeric: tabular-nums; }
    </style>
    """, unsafe_allow_html=True)

    for conf in ["AFC", "NFC"]:
        st.markdown(f"**{conf}**")
        cols = st.columns(4)
        for col, division in zip(cols, ["East", "North", "South", "West"]):
            sub = rec[(rec["conf"] == conf) & (rec["div"] == division)].copy()
            sub = sub.sort_values(["PCT", "NP", "team"], ascending=[False, False, True])

            rows = []
            for _, r in sub.iterrows():
                team = r["team"]
                wl = f"{int(r['W'])}-{int(r['L'])}-{int(r['T'])}"
                rows.append(
                    "<tr>"
                    f"<td><div class='teamrow'>{_logo_img_tag(team, 20)}<span>{team}</span></div></td>"
                    f"<td class='num'>{wl}</td>"
                    f"<td class='num'>{_win_tie_to_pct(r['W'], r['L'], r['T']):.3f}</td>"
                    f"<td class='num'>{float(r['DivPct']):.3f}</td>"
                    f"<td class='num'>{float(r['ConfPct']):.3f}</td>"
                    f"<td class='num'>{float(r['SOS']):.3f}</td>"
                    f"<td class='num'>{float(r['SOV']):.3f}</td>"
                    f"<td class='num'>{int(r['NP'])}</td>"
                    "</tr>"
                )
            with col:
                st.markdown(
                    f"<div style='margin-bottom:6px; font-weight:700'>{division}</div>"
                    "<table class='divtbl'>"
                    "<thead><tr>"
                    "<th>Team</th><th>W-L-T</th><th>Pct</th><th>Div</th><th>Conf</th>"
                    "<th>SOS</th><th>SOV</th><th>NetPts</th>"
                    "</tr></thead>"
                    f"<tbody>{''.join(rows)}</tbody></table>",
                    unsafe_allow_html=True
                )

@st.cache_resource
def _logo_http() -> "requests.Session":
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Referer": "https://www.espn.com/",
    })
    return s

@st.cache_data(show_spinner=False, ttl=60*60*24, max_entries=256)
def _download_logo_bytes(url: str) -> bytes | None:
    """Download once (per URL) with a real UA. Fallback to ESPN's alt host if needed."""
    import requests
    s = _logo_http()
    try:
        r = s.get(url, timeout=5)
        r.raise_for_status()
        return r.content
    except Exception:
        # try alternate CDN host
        try:
            alt = url.replace("a.espncdn.com", "sports-ak.espncdn.com")
            r = s.get(alt, timeout=5)
            r.raise_for_status()
            return r.content
        except Exception:
            return None

@st.cache_data(show_spinner=False)
def _fetch_logo_rgba(url: str, target_px: int = 180):
    """Return an RGBA array ~target_px wide (keeps aspect), high quality."""
    import io
    from PIL import Image, ImageOps
    import numpy as np

    raw = _download_logo_bytes(url)
    if not raw:
        return None  # caller will handle fallback

    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    im = ImageOps.contain(im, (target_px, target_px), Image.LANCZOS)
    return np.asarray(im)

@st.cache_data(show_spinner=False)
def _logo_data_uri(team: str, size_css: int = 28, dpr: float = 2.0, bg_hex: str = "#0f1115") -> str:
    """
    Returns a data: URI for a crisp PNG logo:
      - fetched via your _fetch_logo_rgba()
      - composited on a dark bg to remove white halos
      - rendered at (size_css * dpr) px but displayed at size_css for retina sharpness
    """
    import io, base64
    from PIL import Image, ImageOps, ImageColor
    url = team_logo_url(team)
    if not url:
        return ""
    try:
        px = int(size_css * dpr)
        arr = _fetch_logo_rgba(url, target_px=px)
        im = Image.fromarray(arr, mode="RGBA")
        if bg_hex:
            bg = Image.new("RGBA", im.size, ImageColor.getcolor(bg_hex, "RGBA"))
            im = Image.alpha_composite(bg, im)
        im = ImageOps.contain(im, (px, px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def _logo_img_tag(team: str, size_css: int = 24) -> str:
    """Small <img> tag using the crisp data URI."""
    uri = _logo_data_uri(team, size_css=size_css, dpr=2.0)
    alt = team or "logo"
    # Lock CSS size so browser doesn't resample again
    return (f'<img class="logo-img" src="{uri}" alt="{alt}" '
            f'width="{size_css}" height="{size_css}">')

# ============================================================================
# SIDEBAR NAV
# ============================================================================
page = st.sidebar.radio(
    "Navigate",
    ["Smarter Power Ranking", "2025 Season Simulator", "How It Works"],
    index=0
)

# ============================================================================
# PAGE: RANKING
# ============================================================================
if page == "Smarter Power Ranking":
    st.title("Smarter NFL Power Rankings")
    inject_brand_css()
    seasons_available = sorted(weekly_full["season"].unique().tolist())

    # Default to 2024 if it's available; otherwise fall back to the latest season
    default_season = 2024 if 2024 in seasons_available else max(seasons_available)
    season = st.sidebar.selectbox(
        "Season",
        seasons_available,
        index=seasons_available.index(default_season)
    )

    # Week selector (with playoff labels)
    week_labels = label_weeks_for_season(season)
    season_weeks = sorted(weekly_full.query("season == @season")["week"].unique().tolist())

    # Default to Week 22 for 2024; otherwise default to the last available week for that season.
    week_options = [0] + season_weeks
    default_week = 22 if (season == 2024 and 22 in season_weeks) else (season_weeks[-1] if season_weeks else 0)
    week = st.sidebar.selectbox(
        "Week",
        options=week_options,
        index=week_options.index(default_week),
        format_func=lambda w: "Preseason (week 0)" if w == 0 else week_labels.get(w, f"Week {w}")
    )
    # Build table (always 32 teams)
    if week == 0:
        season_teams = sorted(weekly_full.query("season == @season")["team"].unique().tolist())
        ps = preseason_snapshots.query("season == @season")[["team","rating"]].copy()
        if len(ps) == 0:
            # derive from previous season end + regression, else BASE
            prev = season - 1
            if prev in seasons_available:
                last_wk = weekly_full.query("season == @prev")["week"].max()
                end = weekly_full.query("season == @prev and week == @last_wk")
                regd = regress_to_mean(dict(zip(end["team"], end["rating"])), LEAGUE_MEAN, SEASON_REGRESSION)
            else:
                regd = {}
            ps = pd.DataFrame({"team": season_teams, "rating": [regd.get(t, BASE_RATING) for t in season_teams]})
        else:
            # ensure all teams present
            ps = pd.DataFrame({"team": season_teams}).merge(ps, on="team", how="left")
            ps["rating"] = ps["rating"].fillna(BASE_RATING)
        render_rank_table(ps, f"Ratings — {season} Preseason")
    else:
        table = weekly_full.query("season == @season and week == @week")[["team","rating"]].copy()
        render_rank_table(table, f"Ratings — {season} {week_labels.get(week, f'Week {week}')}")

# ============================================================================
# PAGE: SIMULATOR
# ============================================================================
elif page == "2025 Season Simulator":
    st.title("2025 Season Simulator")
    inject_brand_css()
    _ensure_sim_results()

    if "sim_week" not in st.session_state:
        st.session_state.sim_week = 1
    if "sim_ratings" not in st.session_state:
        st.session_state.sim_ratings = PRESEASON_2025.copy()
    if "playoff_stage" not in st.session_state:
        st.session_state.playoff_stage = None  # "WC","DIV","CONF","SB","DONE"
    if "playoff_data" not in st.session_state:
        st.session_state.playoff_data = {}     # store seeds and round winners

    # Controls
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Reset to Preseason"):
            st.session_state.sim_week = 1
            st.session_state.sim_ratings = PRESEASON_2025.copy()
            st.session_state.sim_results = []
            st.session_state.playoff_stage = None
            st.session_state.playoff_data = {}
            st.rerun()
    with c2:
        if st.button("Advance without changes"):
            st.session_state.sim_week += 1

    # Current ratings table (all 32; whole numbers; new headers; no index)
    cur = pd.DataFrame({
        "team": list(st.session_state.sim_ratings.keys()),
        "rating": list(st.session_state.sim_ratings.values())
    })
    render_rank_table(cur, f"Current Smarter Power Ranking (pre-Week {st.session_state.sim_week})")

    if len(future_2025) == 0:
        st.warning("2025 schedule not available from nflverse right now. You can still reset and view preseason ratings.")
        st.stop()

    max_week_2025 = int(future_2025["week"].max())
    st.info(f"Simulating regular season. Weeks available: 1 – {max_week_2025}")

    week = st.session_state.sim_week

    # ---------- REGULAR SEASON FLOW ----------
    if week <= max_week_2025:
        wk_games = future_2025.query("week == @week").copy()
        st.subheader(f"Week {week} — Make Your Picks")

        # Favorite detector (Elo + HFA)
        def favorite_is_home(row) -> bool:
            h = row["home_team"]; a = row["away_team"]; neutral = bool(row.get("neutral_site", 0))
            r_h = st.session_state.sim_ratings.get(h, BASE_RATING)
            r_a = st.session_state.sim_ratings.get(a, BASE_RATING)
            hfa = 0.0 if neutral else HOME_FIELD_ELO
            return (r_h + hfa) >= r_a

        # ---- Set to Favorites (before form): write defaults into session_state ----
        set_faves = st.button("Set to Favorites", key=f"set_faves_week_{week}")
        if set_faves:
            for i, row in wk_games.iterrows():
                home, away = row["home_team"], row["away_team"]
                default_home = favorite_is_home(row)
                st.session_state[f"pick_{week}_{i}"] = home if default_home else away
                st.session_state[f"mov_{week}_{i}"] = st.session_state.get(f"mov_{week}_{i}", 3)
            st.rerun()  # radios will pick up these values as their default

        # ------------------------------- Picks form -------------------------------
        with st.form(f"week_{week}_form", clear_on_submit=False):
            for i, row in wk_games.iterrows():
                home = canon(row["home_team"]); away = canon(row["away_team"])
                neutral = bool(row.get("neutral_site", 0))

                # decide default winner using Elo favorite
                default_home = favorite_is_home(row)

                # keys for this game
                pick_key = f"pick_{week}_{i}"
                mov_key  = f"mov_{week}_{i}"

                # set defaults ONCE in session_state
                st.session_state.setdefault(pick_key, home if default_home else away)
                st.session_state.setdefault(mov_key, 3)

                # UI (read from session_state via key only — no 'value' or 'index')
                c_pick, c_mov = st.columns([3, 2])

                with c_pick:
                    st.radio(
                        f"{away} @ {home}{' (Neutral)' if neutral else ''}",
                        options=[home, away],
                        key=pick_key,
                        horizontal=True,
                    )

                with c_mov:
                    st.slider(
                        "Margin of Victory",
                        min_value=0, max_value=70, step=1,
                        key=mov_key,
                    )

            submit = st.form_submit_button("Simulate Week")

        # ---------------------------- Apply simulations ---------------------------
        if submit:
            for i, row in wk_games.iterrows():
                home = row["home_team"]; away = row["away_team"]; neutral = bool(row.get("neutral_site", 0))
                picked_home = (st.session_state.get(f"pick_{week}_{i}") == home)
                mov_points = int(st.session_state.get(f"mov_{week}_{i}", 3))  # 0..70; 0 = tie
                # Save result row for standings
                st.session_state.sim_results.append({
                    "season": 2025, "week": int(week),
                    "home": home, "away": away,
                    "winner": "TIE" if mov_points == 0 else ("HOME" if picked_home else "AWAY"),
                    "mov": int(mov_points),
                    "neutral": int(neutral),
                    "is_playoff": 0,
                    "round": "REG"
                })
                # Elo update
                elo_update_one(
                    st.session_state.sim_ratings,
                    home, away, picked_home,
                    neutral_site=neutral,
                    mov_points=mov_points,
                    is_playoff=False
                )
            st.session_state.sim_week += 1
            st.rerun()

        # After/ during regular season: show standings so far
        all_teams_2025 = sorted(pd.unique(pd.concat([future_2025["home_team"], future_2025["away_team"]])))
        rec_df = compute_team_records(all_teams_2025, _results_df())
        render_division_standings_table(rec_df, "Standings (through Week {:.0f})".format(week-1 if week>1 else 0))

    # ---------- POSTSEASON FLOW ----------
    else:
        st.success("Regular season complete — time for the playoffs!")
        results = _results_df()
        all_teams_2025 = sorted(pd.unique(pd.concat([future_2025["home_team"], future_2025["away_team"]])))
        rec_df = compute_team_records(all_teams_2025, results)

        # Build seeds once
        if not st.session_state.playoff_stage:
            st.session_state.playoff_stage = "WC"  # WC -> DIV -> CONF -> SB -> DONE
            st.session_state.playoff_data = {"seeds": build_all_seeds(rec_df, results),
                                             "survivors": {}}

        seeds_all = st.session_state.playoff_data["seeds"]
        render_seeds_pretty(seeds_all)  # clean, non-JSON display

        stage = st.session_state.playoff_stage

        # ----- WILD CARD (one form for AFC+NFC) -----
        if stage == "WC":
            afc_pairs = build_wc_pairs()
            nfc_pairs = build_wc_pairs()

            st.markdown("### Wild Card — Make all your picks, then advance")

            with st.form("WC_all_form"):
                picks_batch = []  # (conf, hi, lo, home_team, away_team, mov_key, pick_key)

                for conf, pairs in [("AFC", afc_pairs), ("NFC", nfc_pairs)]:
                    st.markdown(f"#### {conf}")
                    c1, c2, c3 = st.columns([2, 2, 3])  # simple spacer row header
                    for idx, (hi, lo) in enumerate(pairs):
                        home, away = seeds_all[conf][hi], seeds_all[conf][lo]
                        col_pick, col_mov = st.columns([3, 2])
                        pick_key = f"{conf}_WC_pick_{idx}"
                        mov_key  = f"{conf}_WC_mov_{idx}"
                        with col_pick:
                            st.radio(f"#{lo} {away} @ #{hi} {home}",
                                     options=[home, away], index=0, key=pick_key, horizontal=True)
                        with col_mov:
                            st.slider("Margin of Victory", 1, 70, 3, 1, key=mov_key)
                        picks_batch.append((conf, hi, lo, home, away, mov_key, pick_key))

                go = st.form_submit_button("Advance Wild Card Round")

            if go:
                winners_by_seed = {"AFC": {}, "NFC": {}}
                for conf, hi, lo, home, away, mov_key, pick_key in picks_batch:
                    win_team = st.session_state[pick_key]
                    mov = int(st.session_state[mov_key])
                    picked_home = (win_team == home)

                    # Elo update
                    elo_update_one(
                        st.session_state.sim_ratings,
                        home_team=home, away_team=away,
                        home_win=picked_home,
                        neutral_site=False, mov_points=mov, is_playoff=True
                    )
                    # Log
                    st.session_state.sim_results.append({
                        "season": 2025, "week": 100,
                        "home": home, "away": away,
                        "winner": "HOME" if picked_home else "AWAY",
                        "mov": mov, "neutral": 0, "is_playoff": 1, "round": "WC"
                    })
                    winners_by_seed[conf][hi if picked_home else lo] = win_team

                # survivors include #1 plus WC winners' seeds
                st.session_state.playoff_data["survivors"] = {
                    "AFC": [1] + sorted(list(winners_by_seed["AFC"].keys())),
                    "NFC": [1] + sorted(list(winners_by_seed["NFC"].keys())),
                }
                st.session_state.playoff_stage = "DIV"
                st.rerun()

        # ----- DIVISIONAL (one form for AFC+NFC) -----
        elif stage == "DIV":
            surv = st.session_state.playoff_data.get("survivors", {})
            afc_pairs = reseed_divisional_from_survivors(surv["AFC"])
            nfc_pairs = reseed_divisional_from_survivors(surv["NFC"])

            st.markdown("### Divisional — Make all your picks, then advance")

            with st.form("DIV_all_form"):
                picks_batch = []
                for conf, pairs in [("AFC", afc_pairs), ("NFC", nfc_pairs)]:
                    st.markdown(f"#### {conf}")
                    for idx, (hi, lo) in enumerate(pairs):
                        home, away = seeds_all[conf][hi], seeds_all[conf][lo]
                        pick_key = f"{conf}_DIV_pick_{idx}"
                        mov_key  = f"{conf}_DIV_mov_{idx}"
                        col_pick, col_mov = st.columns([3, 2])
                        with col_pick:
                            st.radio(f"#{lo} {away} @ #{hi} {home}",
                                     options=[home, away], index=0, key=pick_key, horizontal=True)
                        with col_mov:
                            st.slider("Margin of Victory", 1, 70, 3, 1, key=mov_key)
                        picks_batch.append((conf, hi, lo, home, away, mov_key, pick_key))

                go = st.form_submit_button("Advance Divisional Round")

            if go:
                winners_by_seed = {"AFC": {}, "NFC": {}}
                for conf, hi, lo, home, away, mov_key, pick_key in picks_batch:
                    win_team = st.session_state[pick_key]
                    mov = int(st.session_state[mov_key])
                    picked_home = (win_team == home)
                    elo_update_one(
                        st.session_state.sim_ratings,
                        home_team=home, away_team=away,
                        home_win=picked_home,
                        neutral_site=False, mov_points=mov, is_playoff=True
                    )
                    st.session_state.sim_results.append({
                        "season": 2025, "week": 101,
                        "home": home, "away": away,
                        "winner": "HOME" if picked_home else "AWAY",
                        "mov": mov, "neutral": 0, "is_playoff": 1, "round": "DIV"
                    })
                    winners_by_seed[conf][hi if picked_home else lo] = win_team

                # survivors for CONF are just the two seeds that advanced
                st.session_state.playoff_data["survivors"] = {
                    "AFC": sorted(list(winners_by_seed["AFC"].keys())),
                    "NFC": sorted(list(winners_by_seed["NFC"].keys())),
                }
                st.session_state.playoff_stage = "CONF"
                st.rerun()

        # ----- CONFERENCE CHAMPIONSHIPS (one form for AFC+NFC) -----
        elif stage == "CONF":
            surv = st.session_state.playoff_data.get("survivors", {})
            afc = surv["AFC"]; nfc = surv["NFC"]
            afc_pairs = [(min(afc), max(afc))]
            nfc_pairs = [(min(nfc), max(nfc))]

            st.markdown("### Conference Championships — Pick both, then advance")

            with st.form("CONF_all_form"):
                picks_batch = []
                for conf, pairs in [("AFC", afc_pairs), ("NFC", nfc_pairs)]:
                    st.markdown(f"#### {conf}")
                    for idx, (hi, lo) in enumerate(pairs):
                        home, away = seeds_all[conf][hi], seeds_all[conf][lo]
                        pick_key = f"{conf}_CONF_pick_{idx}"
                        mov_key  = f"{conf}_CONF_mov_{idx}"
                        col_pick, col_mov = st.columns([3, 2])
                        with col_pick:
                            st.radio(f"#{lo} {away} @ #{hi} {home}",
                                     options=[home, away], index=0, key=pick_key, horizontal=True)
                        with col_mov:
                            st.slider("Margin of Victory", 1, 70, 3, 1, key=mov_key)
                        picks_batch.append((conf, hi, lo, home, away, mov_key, pick_key))

                go = st.form_submit_button("Advance to Super Bowl")

            if go:
                champs = {}
                for conf, hi, lo, home, away, mov_key, pick_key in picks_batch:
                    win_team = st.session_state[pick_key]
                    mov = int(st.session_state[mov_key])
                    picked_home = (win_team == home)
                    elo_update_one(
                        st.session_state.sim_ratings,
                        home_team=home, away_team=away,
                        home_win=picked_home,
                        neutral_site=False, mov_points=mov, is_playoff=True
                    )
                    st.session_state.sim_results.append({
                        "season": 2025, "week": 102,
                        "home": home, "away": away,
                        "winner": "HOME" if picked_home else "AWAY",
                        "mov": mov, "neutral": 0, "is_playoff": 1, "round": "CONF"
                    })
                    champs[conf] = win_team

                st.session_state.playoff_data["champs"] = champs
                st.session_state.playoff_stage = "SB"
                st.rerun()

        # ----- SUPER BOWL (neutral, one form) -----
        elif stage == "SB":
            champs = st.session_state.playoff_data.get("champs", {})
            afc_champ = champs.get("AFC"); nfc_champ = champs.get("NFC")
            if not afc_champ or not nfc_champ:
                st.info("Waiting for conference champions…")
                st.stop()

            st.markdown("### Super Bowl — Neutral Site")
            with st.form("SB_form"):
                pick_key = "SB_pick"
                mov_key  = "SB_mov"
                col_pick, col_mov = st.columns([3, 2])
                with col_pick:
                    st.radio(f"{nfc_champ} vs {afc_champ}",
                             options=[nfc_champ, afc_champ],
                             index=0, key=pick_key, horizontal=True)
                with col_mov:
                    st.slider("Margin of Victory", 1, 70, 3, 1, key=mov_key)
                go = st.form_submit_button("Crown Champion")

            if go:
                home, away = nfc_champ, afc_champ  # arbitrary; neutral site
                win_team = st.session_state[pick_key]
                mov = int(st.session_state[mov_key])
                picked_home = (win_team == home)
                elo_update_one(
                    st.session_state.sim_ratings,
                    home_team=home, away_team=away,
                    home_win=picked_home,
                    neutral_site=True, mov_points=mov, is_playoff=True
                )
                st.session_state.sim_results.append({
                    "season": 2025, "week": 103,
                    "home": home, "away": away,
                    "winner": "HOME" if picked_home else "AWAY",
                    "mov": mov, "neutral": 1, "is_playoff": 1, "round": "SB"
                })
                st.session_state.playoff_stage = "DONE"
                st.rerun()

        # ----- FINISH -----
        elif stage == "DONE":
            res = _results_df()
            sb = res[res["round"] == "SB"].tail(1)
            if not sb.empty:
                r = sb.iloc[-1]
                champ = r["home"] if r["winner"] == "HOME" else r["away"]
                st.success(f"🏆 {champ} are your champions!")

            # --- NEW: render filled-out bracket image above final standings ---
            render_bracket_tables(st.session_state.playoff_data["seeds"], _results_df())

            # final regular-season standings (rounded via helper)
            render_division_standings_table(rec_df, "Final Regular-Season Standings")


# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "How It Works":
    st.title("How the Smarter Power Ranking works")
    inject_brand_css()
    st.markdown("""
### The 10-second version
We assign every team a **Smart Score** (an Elo rating). After each game we move the winner up and the loser down.
Bigger upsets move more. Home teams get a small bump. Each new season we pull everyone a bit back toward average.

---

### What is Elo, in plain English?
- Imagine every team starts around **1500**.  
- If Team A is higher than Team B, we expect A to win more often.  
- The **bigger** the gap, the **higher** the expected win chance.  
- After a game:
  - If you **do better than expected**, you go up.
  - If you **do worse than expected**, you go down.

---

### Our NFL-specific tweaks
- **Between-season regression:** Before Week 1 each year, we nudge every team **SEASON_REGRESSION = 40%** of the way back to **1500**. This stops last year’s story from overpowering this year.
- **Home-field advantage:** Home team gets **+40 Elo** (≈ ~2 points on the scoreboard). Neutral-site games (Super Bowl, international) get **0**.
- **Margin of victory:** Blowouts move ratings more, but we damp it if the favorite was already much stronger.

- **Playoffs:** Games count a bit extra (× **1.2**).
- **Market blend:** When moneylines are available we convert them to a **fair home win probability** (remove the sportsbook vig) and blend with Elo:

`Final P = (1 − w) * P_elo + w * P_market`, with `w = 0.30`.

If no lines, we use Elo only.

---

### What does the Smart Score mean?
- Scores are centered around **1500**.  
- +50 ≈ a small edge; +100 ≈ clear favorite; +200 ≈ heavy favorite.  
- The **difference** between two teams’ Smart Scores maps to a win chance through a logistic curve (the classic Elo math).

---

### What you see in the app
- **Smarter Power Ranking** for any week = ratings **after** all games in that week.
- **Preseason** = last year’s end-of-season ratings, gently pulled back toward **1500**.
- **2025 Simulator** = same update rules, with a **Margin of Victory** slider:
  - **Regular season:** 0–70 (default **3**). **0 = tie**.
  - **Playoffs:** 1–70 (no ties).
""")

    st.markdown("""
---
### Standings, Tiebreakers & Playoffs — plain English

**Records & standings**
- After each simulated week, the app recomputes each team’s **W-L-T**, **division record**, **conference record**, **net points** (points for minus points against), **Strength of Schedule (SOS)**, and **Strength of Victory (SOV)**.
- **SOS**/**SOV** use opponents’ combined records, **excluding games vs your team** (the standard approach).

**Division tiebreakers** (teams in the **same division**)
1. **Head-to-head** record among the tied teams  
2. **Division record**  
3. **Common games** record (only if each team has **≥ 4** common opponents)  
4. **Conference record**  
5. **Strength of Victory (SOV)**  
6. **Strength of Schedule (SOS)**  
7. **Net points** (overall)  

**Conference / Wild-Card tiebreakers** (ranking teams across the **conference**)
1. **Head-to-head**  
2. **Conference record**  
3. **Common games** record (≥ 4)  
4. **Strength of Victory (SOV)**  
5. **Strength of Schedule (SOS)**  
6. **Net points in conference games**  
7. **Net points overall**

- For tougher **multi-team ties**, we use weighted head-to-head among the tied teams, then apply the criteria in order until the tie breaks (alphabetical is a last-ditch fallback and is extremely rare).

**Seeding**
- Each conference seeds **7 teams**: the **4 division winners** plus **3 wild cards**.  
- The 4 division winners are ordered using the conference tiebreakers.  
- The **#1 seed gets a bye** on Wild Card weekend.

**Reseeding**
- After **Wild Card** weekend, the **#1 seed plays the lowest remaining seed**; the other two winners play each other in the **Divisional** round.  
- **Conference Championships** pair the two remaining teams by seed (higher hosts).  
- The **Super Bowl** is played at a **neutral site**.

**Playoff Elo**
- Postseason games use the playoff multiplier `K_PLAYOFF_MULT` (default **1.2**).  
- Your picks in the playoffs update Elo just like the regular season, but with the playoff multiplier and **no ties**.
""")
