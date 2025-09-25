from __future__ import annotations
from typing import Dict

def badge_for_accuracy(drop_pp: float, rules: Dict[str, Dict[str, float | str]]) -> str:
    # rules: {"good":{"max_abs_drop_pp":0.5,"emoji":"🟢"}, ...}
    drop_pp = abs(drop_pp) if drop_pp is not None else 999.0
    for name in ("good","ok","bad"):
        lim = rules.get(name, {})
        if drop_pp <= float(lim.get("max_abs_drop_pp", 999.0)):
            return str(lim.get("emoji",""))
    return ""

def badge_for_speedup(speedup_x: float, rules: Dict[str, Dict[str, float | str]]) -> str:
    # rules: {"strong":{"min_speedup_x":1.7,"emoji":"🚀"}, ...}
    sx = 0.0 if speedup_x is None else float(speedup_x)
    for name in ("strong","ok","weak"):
        lim = rules.get(name, {})
        if sx >= float(lim.get("min_speedup_x", 0.0)):
            return str(lim.get("emoji",""))
    return ""
