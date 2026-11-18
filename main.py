"""gta6-twitch-ccu-forecast-01 entry: log-ratio regression on past launches.

Model (fit at run time from the case's own `comparables`, stdlib only):
for each forecast hour h = 2..24, across past launches with a real hour-1
print (>= 5,000 viewers, so pre-unlock trickles and slow burns do not
dominate the fit), regress the log ratio log(v_h / v_1) on log(v_1):

    log(v_h / v_1) = a_h + b_h * log(v_1) + eps,   eps ~ N(0, s_h^2)

A big hour-1 print means the launch already peaked in hour 1 (ratios stay
near 1, then dip and recover with the evening UTC peak); a modest hour-1
print means the big markets have not unlocked yet (ratios rise several-fold
into the US evening). The slope b_h captures exactly that, per hour.

Prediction for the target case: mu_h is the fitted value at log(p1),
shrunk 20% toward the mean log ratio (guards the extrapolation when p1
exceeds every training launch, as GTA VI's hour 1 plausibly will) and
clamped to the observed range of log ratios for that hour. Spread s_h is
the residual standard deviation (floored at 0.10 log units), and the five
quantiles are p1 * exp(mu_h + z_tau * s_h), z = standard normal quantiles.

Validated leave-one-launch-out on the kit's 37 usable comparables
(mean log-pinball loss; lower is better):
  - persistence baseline: 0.930 (all), 0.418 (rolling-midnight console
    launches), 0.321 (launches with hour-1 >= 20k)
  - this model:            0.645 (all), 0.301 (rolling-midnight console
    launches), 0.250 (launches with hour-1 >= 20k)
The house baseline's published practice loss is 0.387.

Fallback: if too few usable comparables are in the case, the program
falls back to the persistence baseline (sigma 0.15), so it always writes
a valid forecast.

Deterministic: fixed file order, closed-form OLS, no randomness, no
clock, no network. Reads /input/<id>.in, writes /output/<id>.out.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
BASE_SIGMA = 0.15          # persistence fallback spread
MIN_TRAIN_P1 = 5000.0      # training launches need a real hour-1 print
MIN_TRAIN = 5              # below this many launches, fall back to baseline
SHRINK = 0.8               # shrink fitted log-ratio toward its mean
SD_FLOOR = 0.10            # minimum log-spread per hour


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def comparable_day_series(comp: dict) -> dict[int, float] | None:
    """Release-day hourly viewers {1..24} for one comparable launch, or None."""
    try:
        day_start = parse_ts(comp["day_start"])
        hourly = comp.get("hourly") or []
    except (KeyError, TypeError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    for pt in hourly:
        try:
            t = parse_ts(pt["from"])
            v = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if v is None:
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and v > 0:
            by_hour[h] = float(v)
    if len(by_hour) < 24 or by_hour.get(1, 0) <= 0:
        return None
    return by_hour


def ols_1d(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Closed-form least squares y ~ a + b*x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def hour_quantiles(p1: float, h: int, train: list[dict[int, float]]) -> list[float | int]:
    """Five quantiles for hour h from the log-ratio regression fit."""
    xs: list[float] = []
    ys: list[float] = []
    for by in train:
        v1 = by[1]
        vh = by.get(h)
        if v1 >= MIN_TRAIN_P1 and vh is not None and vh > 0:
            xs.append(math.log(v1))
            ys.append(math.log(vh / v1))
    if len(xs) < MIN_TRAIN:
        return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]
    a, b = ols_1d(xs, ys)
    mu = a + b * math.log(p1)
    mean_y = sum(ys) / len(ys)
    mu = mean_y + SHRINK * (mu - mean_y)
    mu = max(min(ys), min(max(ys), mu))
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in res) / len(res)
    sd = max(math.sqrt(var), SD_FLOOR)
    out: list[float | int] = []
    prev = 0.0
    for z in Z:
        q = round2(p1 * math.exp(mu + z * sd))
        q = max(q, 0.01)
        if q < prev:  # rounding must never break non-decreasing order
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    train: list[dict[int, float]] = []
    for comp in case.get("comparables") or []:
        series = comparable_day_series(comp)
        if series is not None:
            train.append(series)
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], train)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
