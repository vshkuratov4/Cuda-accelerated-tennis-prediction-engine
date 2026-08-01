"""Feature engineering shared by training and live inference.

This replaces the duplicated Elo/H2H/form/streak/etc. logic that used to live
separately in clean_data.py (batch, one-shot script) and app_tkinter.py
(placeholder neutral values at inference time). ``FeatureState`` is the single
implementation: ``fit()`` replays full match history chronologically to build
both an enriched training table *and* a compact "current form" snapshot of
every player, and ``transform_live()`` reuses that exact snapshot to build
features for a brand-new, not-yet-played matchup - so live predictions use the
model's real learned signal instead of neutral placeholders.

``FeatureState`` must stay picklable, which is why every ``defaultdict``
factory below is a plain module-level function rather than a lambda/closure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
K_FACTOR = 32

ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "Quarter": 5,
    "SF": 6, "Semi": 6,
    "F": 7, "Final": 7,
}

FEATURE_COLUMNS = [
    "P1_rank", "P2_rank", "P1_pts", "P2_pts",
    "P1_b365", "P2_b365", "rank_diff", "odds_diff",
    "P1_Elo_pre", "P2_Elo_pre", "Elo_diff",
    "P1_H2H_pct", "P2_H2H_pct", "P1_H2H_surf", "P2_H2H_surf", "H2H_tot", "H2H_surf_tot",
    "P1_rest", "P2_rest", "P1_rest_missing", "P2_rest_missing",
    "P1_Form5", "P2_Form5",
    "P1_SurfForm20", "P2_SurfForm20",
    "P1_streak", "P2_streak",
    "P1_tourny_ct", "P2_tourny_ct",
    "P1_load30", "P2_load30", "P1_load30_missing", "P2_load30_missing",
    "P1_SetWinPct", "P2_SetWinPct",
    "P1_SurfSetPct", "P2_SurfSetPct",
    "P1_RankMom", "P2_RankMom",
    "Round_num", "Surface_num",
]


def _base_elo_factory() -> float:
    return BASE_ELO


def _pair_factory() -> list:
    return [0, 0]


def _nested_pair_factory():
    return defaultdict(_pair_factory)


def _nested_deque_factory():
    return defaultdict(deque)


def _nat_factory():
    return pd.NaT


def _last_known_factory() -> dict:
    return {"rank": np.nan, "pts": np.nan, "b365": np.nan, "date": pd.NaT}


@dataclass
class FeatureState:
    """Holds every player's "current form" as of the last match seen by fit()."""

    elo: dict = field(default_factory=lambda: defaultdict(_base_elo_factory))
    h2h_all: dict = field(default_factory=lambda: defaultdict(_pair_factory))
    h2h_surf: dict = field(default_factory=lambda: defaultdict(_pair_factory))
    wins_last5: dict = field(default_factory=lambda: defaultdict(list))
    form20: dict = field(default_factory=lambda: defaultdict(_nested_deque_factory))
    streak: dict = field(default_factory=lambda: defaultdict(int))
    tour_fat: dict = field(default_factory=lambda: defaultdict(int))
    workload30: dict = field(default_factory=lambda: defaultdict(deque))
    sets_all: dict = field(default_factory=lambda: defaultdict(_pair_factory))
    sets_surf: dict = field(default_factory=lambda: defaultdict(_nested_pair_factory))
    rank_history: dict = field(default_factory=lambda: defaultdict(deque))
    last_date: dict = field(default_factory=lambda: defaultdict(_nat_factory))
    last_known: dict = field(default_factory=lambda: defaultdict(_last_known_factory))
    surface_map: dict = field(default_factory=dict)
    players: list = field(default_factory=list)
    matches_played: dict = field(default_factory=lambda: defaultdict(int))
    as_of: Optional[str] = None

    # ------------------------------------------------------------------
    # Training-time: replay full history, return enriched df + build state
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date").reset_index(drop=True)

        surface_types = df["Surface"].dropna().unique().tolist()
        self.surface_map = {surf: idx for idx, surf in enumerate(surface_types, start=1)}

        new_cols = {
            "Elo_W_pre": [], "Elo_L_pre": [], "Elo_diff": [],
            "H2H_win_pct": [], "H2H_tot": [],
            "H2H_surf_pct": [], "H2H_surf_tot": [],
            "Rest_days_w": [], "Rest_days_l": [],
            "Form5_w": [], "Form5_l": [],
            "SurfForm20_w": [], "SurfForm20_l": [],
            "Streak_w": [], "Streak_l": [],
            "Tourny_match_w": [], "Tourny_match_l": [],
            "Load30_w": [], "Load30_l": [],
            "SetWinPct_w": [], "SetWinPct_l": [],
            "SurfSetWinPct_w": [], "SurfSetWinPct_l": [],
            "RankMom_w": [], "RankMom_l": [],
        }

        for _, row in df.iterrows():
            w, l = row["Winner"], row["Loser"]
            date = row["Date"]
            surf = row.get("Surface")
            tour = row.get("Tournament")
            raw_wsets, raw_lsets = row.get("Wsets", 0), row.get("Lsets", 0)
            wsets = 0 if pd.isna(raw_wsets) else int(raw_wsets)
            lsets = 0 if pd.isna(raw_lsets) else int(raw_lsets)
            total_sets = wsets + lsets

            ew, el = self.elo[w], self.elo[l]
            new_cols["Elo_W_pre"].append(ew)
            new_cols["Elo_L_pre"].append(el)
            new_cols["Elo_diff"].append(ew - el)

            pair = tuple(sorted([w, l]))
            wins, tot = self.h2h_all[pair]
            new_cols["H2H_win_pct"].append(wins / tot if tot else np.nan)
            new_cols["H2H_tot"].append(tot)

            spair = (*pair, surf)
            sw, st = self.h2h_surf[spair]
            new_cols["H2H_surf_pct"].append(sw / st if st else np.nan)
            new_cols["H2H_surf_tot"].append(st)

            rest_w = (date - self.last_date[w]).days if pd.notna(self.last_date[w]) else np.nan
            rest_l = (date - self.last_date[l]).days if pd.notna(self.last_date[l]) else np.nan
            new_cols["Rest_days_w"].append(rest_w)
            new_cols["Rest_days_l"].append(rest_l)

            new_cols["Form5_w"].append(sum(self.wins_last5[w][-5:]))
            new_cols["Form5_l"].append(sum(self.wins_last5[l][-5:]))

            pfw, pfl = list(self.form20[w][surf]), list(self.form20[l][surf])
            new_cols["SurfForm20_w"].append(np.mean(pfw) if pfw else np.nan)
            new_cols["SurfForm20_l"].append(np.mean(pfl) if pfl else np.nan)

            new_cols["Streak_w"].append(self.streak[w])
            new_cols["Streak_l"].append(self.streak[l])

            new_cols["Tourny_match_w"].append(self.tour_fat[(w, tour)])
            new_cols["Tourny_match_l"].append(self.tour_fat[(l, tour)])

            dq_w, dq_l = self.workload30[w], self.workload30[l]
            new_cols["Load30_w"].append(sum(1 for d in dq_w if (date - d).days <= 30))
            new_cols["Load30_l"].append(sum(1 for d in dq_l if (date - d).days <= 30))

            sw_all, sp_all = self.sets_all[w]
            new_cols["SetWinPct_w"].append(sw_all / sp_all if sp_all else np.nan)
            sw_all_l, sp_all_l = self.sets_all[l]
            new_cols["SetWinPct_l"].append(sw_all_l / sp_all_l if sp_all_l else np.nan)

            ssw, ssp = self.sets_surf[w][surf]
            new_cols["SurfSetWinPct_w"].append(ssw / ssp if ssp else np.nan)
            ssw_l, ssp_l = self.sets_surf[l][surf]
            new_cols["SurfSetWinPct_l"].append(ssw_l / ssp_l if ssp_l else np.nan)

            dq_r_w, dq_r_l = self.rank_history[w], self.rank_history[l]
            while dq_r_w and (date - dq_r_w[0][0]).days > 30:
                dq_r_w.popleft()
            while dq_r_l and (date - dq_r_l[0][0]).days > 30:
                dq_r_l.popleft()
            old_w = dq_r_w[0][1] if dq_r_w else np.nan
            old_l = dq_r_l[0][1] if dq_r_l else np.nan
            new_cols["RankMom_w"].append((old_w - ew) if not np.isnan(old_w) else np.nan)
            new_cols["RankMom_l"].append((old_l - el) if not np.isnan(old_l) else np.nan)

            # --- updates ---
            pw = 1 / (1 + 10 ** ((el - ew) / 400))
            self.elo[w] += K_FACTOR * (1 - pw)
            self.elo[l] += K_FACTOR * (0 - (1 - pw))

            self.h2h_all[pair][1] += 1
            if pair[0] == w:
                self.h2h_all[pair][0] += 1
            self.h2h_surf[spair][1] += 1
            if spair[0] == w:
                self.h2h_surf[spair][0] += 1

            self.wins_last5[w].append(1)
            self.wins_last5[l].append(0)

            self.form20[w][surf].append(1)
            self.form20[l][surf].append(0)
            if len(self.form20[w][surf]) > 20:
                self.form20[w][surf].popleft()
            if len(self.form20[l][surf]) > 20:
                self.form20[l][surf].popleft()

            self.streak[w] = self.streak[w] + 1 if self.streak[w] >= 0 else 1
            self.streak[l] = self.streak[l] - 1 if self.streak[l] <= 0 else -1

            self.tour_fat[(w, tour)] += 1
            self.tour_fat[(l, tour)] += 1

            self.workload30[w].append(date)
            self.workload30[l].append(date)

            self.sets_all[w][0] += wsets
            self.sets_all[w][1] += total_sets
            self.sets_all[l][0] += lsets
            self.sets_all[l][1] += total_sets
            self.sets_surf[w][surf][0] += wsets
            self.sets_surf[w][surf][1] += total_sets
            self.sets_surf[l][surf][0] += lsets
            self.sets_surf[l][surf][1] += total_sets

            self.rank_history[w].append((date, ew))
            self.rank_history[l].append((date, el))

            self.last_known[w] = {"rank": row.get("WRank"), "pts": row.get("WPts"),
                                   "b365": row.get("B365W"), "date": date}
            self.last_known[l] = {"rank": row.get("LRank"), "pts": row.get("LPts"),
                                   "b365": row.get("B365L"), "date": date}

            self.last_date[w] = date
            self.last_date[l] = date

            self.matches_played[w] += 1
            self.matches_played[l] += 1

        df["Surface_num"] = df["Surface"].map(self.surface_map)
        if "Round" in df.columns:
            df["Round_num"] = df["Round"].map(ROUND_MAP)

        for name, vals in new_cols.items():
            df[name] = vals

        df["H2H_win_pct"] = df["H2H_win_pct"].fillna(0.5)
        df["H2H_surf_pct"] = df["H2H_surf_pct"].fillna(0.5)
        df["H2H_tot"] = df["H2H_tot"].fillna(0).astype(int)
        df["H2H_surf_tot"] = df["H2H_surf_tot"].fillna(0).astype(int)

        for c in ["Form5_w", "Form5_l", "SurfForm20_w", "SurfForm20_l",
                  "SetWinPct_w", "SetWinPct_l", "SurfSetWinPct_w", "SurfSetWinPct_l"]:
            df[c] = df[c].fillna(0)

        for c in ["Streak_w", "Streak_l", "Tourny_match_w", "Tourny_match_l",
                  "Load30_w", "Load30_l", "RankMom_w", "RankMom_l"]:
            df[c] = df[c].fillna(0)

        df["Rest_w_missing"] = df["Rest_days_w"].isna().astype(int)
        df["Rest_l_missing"] = df["Rest_days_l"].isna().astype(int)
        df["Load30_w_missing"] = df["Rest_w_missing"]
        df["Load30_l_missing"] = df["Rest_l_missing"]

        df["Rest_days_w"] = df["Rest_days_w"].fillna(-1)
        df["Rest_days_l"] = df["Rest_days_l"].fillna(-1)
        df.loc[df["Load30_w_missing"] == 1, "Load30_w"] = -1
        df.loc[df["Load30_l_missing"] == 1, "Load30_l"] = -1

        df = df.fillna(0)

        self.players = sorted(set(df["Winner"]).union(set(df["Loser"])))
        self.as_of = datetime.now().isoformat()

        return df

    # ------------------------------------------------------------------
    # Inference-time: build one feature row for a not-yet-played matchup
    # ------------------------------------------------------------------
    def transform_live(
        self,
        player1: str,
        player2: str,
        surface: str,
        round_name: Optional[str] = None,
        rank1: Optional[float] = None,
        rank2: Optional[float] = None,
        pts1: Optional[float] = None,
        pts2: Optional[float] = None,
        odds1: Optional[float] = None,
        odds2: Optional[float] = None,
    ) -> pd.DataFrame:
        today = pd.Timestamp(datetime.today())

        lk1 = self.last_known.get(player1, _last_known_factory())
        lk2 = self.last_known.get(player2, _last_known_factory())

        p1_rank = rank1 if rank1 is not None else lk1["rank"]
        p2_rank = rank2 if rank2 is not None else lk2["rank"]
        p1_pts = pts1 if pts1 is not None else lk1["pts"]
        p2_pts = pts2 if pts2 is not None else lk2["pts"]
        p1_b365 = odds1 if odds1 is not None else lk1["b365"]
        p2_b365 = odds2 if odds2 is not None else lk2["b365"]

        p1_rank = float(p1_rank) if p1_rank is not None and not pd.isna(p1_rank) else 0.0
        p2_rank = float(p2_rank) if p2_rank is not None and not pd.isna(p2_rank) else 0.0
        p1_pts = float(p1_pts) if p1_pts is not None and not pd.isna(p1_pts) else 0.0
        p2_pts = float(p2_pts) if p2_pts is not None and not pd.isna(p2_pts) else 0.0
        p1_b365 = float(p1_b365) if p1_b365 is not None and not pd.isna(p1_b365) else 2.0
        p2_b365 = float(p2_b365) if p2_b365 is not None and not pd.isna(p2_b365) else 2.0

        e1, e2 = self.elo[player1], self.elo[player2]

        pair = tuple(sorted([player1, player2]))
        w, t = self.h2h_all[pair]
        p1_h2h = (w / t if pair[0] == player1 else (t - w) / t) if t else 0.5

        spair = (*pair, surface)
        sw, st = self.h2h_surf[spair]
        p1_surf = (sw / st if spair[0] == player1 else (st - sw) / st) if st else 0.5

        d1, d2 = lk1["date"], lk2["date"]
        rest1_missing = 1 if pd.isna(d1) else 0
        rest2_missing = 1 if pd.isna(d2) else 0
        rest1 = -1 if rest1_missing else (today - d1).days
        rest2 = -1 if rest2_missing else (today - d2).days

        load1 = sum(1 for d in self.workload30[player1] if (today - d).days <= 30)
        load2 = sum(1 for d in self.workload30[player2] if (today - d).days <= 30)
        load1_missing = 1 if not self.workload30[player1] else 0
        load2_missing = 1 if not self.workload30[player2] else 0

        sw1, sp1 = self.sets_all[player1]
        sw2, sp2 = self.sets_all[player2]
        ssw1, ssp1 = self.sets_surf[player1][surface]
        ssw2, ssp2 = self.sets_surf[player2][surface]

        dq1, dq2 = self.rank_history[player1], self.rank_history[player2]
        old1 = dq1[0][1] if dq1 else e1
        old2 = dq2[0][1] if dq2 else e2

        feats = {
            "P1_rank": p1_rank, "P2_rank": p2_rank,
            "P1_pts": p1_pts, "P2_pts": p2_pts,
            "P1_b365": p1_b365, "P2_b365": p2_b365,
            "rank_diff": p1_rank - p2_rank,
            "odds_diff": p1_b365 - p2_b365,
            "P1_Elo_pre": e1, "P2_Elo_pre": e2, "Elo_diff": e1 - e2,
            "P1_H2H_pct": p1_h2h, "P2_H2H_pct": 1 - p1_h2h,
            "P1_H2H_surf": p1_surf, "P2_H2H_surf": 1 - p1_surf,
            "H2H_tot": t, "H2H_surf_tot": st,
            "P1_rest": rest1, "P2_rest": rest2,
            "P1_rest_missing": rest1_missing, "P2_rest_missing": rest2_missing,
            "P1_Form5": sum(self.wins_last5[player1][-5:]),
            "P2_Form5": sum(self.wins_last5[player2][-5:]),
            "P1_SurfForm20": (np.mean(self.form20[player1][surface])
                               if self.form20[player1][surface] else 0.0),
            "P2_SurfForm20": (np.mean(self.form20[player2][surface])
                               if self.form20[player2][surface] else 0.0),
            "P1_streak": self.streak[player1], "P2_streak": self.streak[player2],
            "P1_tourny_ct": 0, "P2_tourny_ct": 0,
            "P1_load30": load1, "P2_load30": load2,
            "P1_load30_missing": load1_missing, "P2_load30_missing": load2_missing,
            "P1_SetWinPct": sw1 / sp1 if sp1 else 0.0,
            "P2_SetWinPct": sw2 / sp2 if sp2 else 0.0,
            "P1_SurfSetPct": ssw1 / ssp1 if ssp1 else 0.0,
            "P2_SurfSetPct": ssw2 / ssp2 if ssp2 else 0.0,
            "P1_RankMom": old1 - e1, "P2_RankMom": old2 - e2,
            "Round_num": ROUND_MAP.get(round_name, 0),
            "Surface_num": self.surface_map.get(surface, 0),
        }

        return pd.DataFrame([feats])[FEATURE_COLUMNS]

    def surfaces(self) -> list:
        return list(self.surface_map.keys())

    def match_count(self, player: str) -> int:
        # getattr guards against feature_state.pkl files pickled before this
        # field existed - older artifacts just report 0 (== "unknown history").
        matches_played = getattr(self, "matches_played", None)
        return matches_played.get(player, 0) if matches_played else 0
