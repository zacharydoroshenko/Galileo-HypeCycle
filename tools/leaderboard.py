"""Live standings (same JSON the leaderboard page polls every 10 s).

    python3 tools/leaderboard.py                 # exoplanet-orbits
    python3 tools/leaderboard.py --task all
Page: https://siderius.scitra.org/hackathon/leaderboard?view=projector  (provisional; only reviewed scores)
API:  https://submit.siderius.scitra.org/api/leaderboard
Our team: G2-15 "Hype Cycle". Official baseline (validation): 0.4715 (in-dist 0.5975, high_ecc 0.316, sparse 0.375).
"""
import argparse
import json
import urllib.request

API = "https://submit.siderius.scitra.org/api/leaderboard"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="exoplanet-orbits")
    a = ap.parse_args()
    req = urllib.request.Request(API, headers={"User-Agent": "team-leaderboard-check"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    print("updated", d.get("updated_at"))
    for name, t in d["tasks"].items():
        if a.task != "all" and name != a.task:
            continue
        print(f"\n== {name}  ({t['metric']}, {t['direction']} is better)  baseline={t.get('baseline')}")
        for i, e in enumerate(t.get("entries", []), 1):
            print(f"  {i:2d}. " + "  ".join(f"{k}={v}" for k, v in e.items()))
        if not t.get("entries"):
            print("  (no scored entries yet)")
        for r in t.get("recent", [])[:5]:
            print("  recent:", json.dumps(r)[:200])
