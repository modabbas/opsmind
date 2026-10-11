"""Demo data: a small plant with three production lines.

Generates 72 hours of metrics at 10-minute resolution with deliberately
planted faults, plus the maintenance documents the RAG layer answers from.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .timeseries import Point

DEVICES = [
    {"device": "LINE-A", "line": "Bottling A", "location": "Hall 1", "criticality": "high"},
    {"device": "LINE-B", "line": "Bottling B", "location": "Hall 1", "criticality": "high"},
    {"device": "PUMP-07", "line": "Utilities", "location": "Hall 2", "criticality": "medium"},
]

THRESHOLDS = [
    ("LINE-A", "temperature", 78.0, 85.0, "degC"),
    ("LINE-A", "vibration", 4.5, 6.0, "mm/s"),
    ("LINE-A", "throughput", 0.0, 0.0, "units/min"),
    ("LINE-B", "temperature", 78.0, 85.0, "degC"),
    ("LINE-B", "vibration", 4.5, 6.0, "mm/s"),
    ("LINE-B", "throughput", 0.0, 0.0, "units/min"),
    ("PUMP-07", "pressure", 6.2, 7.0, "bar"),
    ("PUMP-07", "vibration", 3.5, 5.0, "mm/s"),
]

PROFILES = {
    ("LINE-A", "temperature"): (68.0, 1.4),
    ("LINE-A", "vibration"): (2.6, 0.35),
    ("LINE-A", "throughput"): (112.0, 6.0),
    ("LINE-B", "temperature"): (70.0, 1.6),
    ("LINE-B", "vibration"): (2.9, 0.4),
    ("LINE-B", "throughput"): (104.0, 7.0),
    ("PUMP-07", "pressure"): (5.4, 0.25),
    ("PUMP-07", "vibration"): (2.2, 0.3),
}

KNOWLEDGE_BASE = [
    {
        "doc_id": "SOP-101",
        "title": "SOP-101 — Bottling Line Overheating Response",
        "text": """
Scope: this procedure applies to bottling lines LINE-A and LINE-B.
A bottling line is considered to be overheating when the sealing head temperature
exceeds 78 degC (warning) or 85 degC (critical). The temperature sensor samples every
ten minutes and the control system averages three consecutive samples before raising
an alarm, to avoid reacting to a single bad reading.
Immediate response at the warning level: reduce line speed to 80 percent, confirm that
the cooling fan on the sealing head is running, and check the coolant return
temperature at the manifold. Do not stop the line at the warning level; a hot stop
leaves product jammed in the sealing head and adds roughly forty minutes to recovery.
Immediate response at the critical level: stop the line using the controlled stop
button, never the emergency stop, unless there is a safety risk to a person. Allow the
sealing head to cool for at least fifteen minutes before opening the guard.
Most common root causes, in the order the maintenance team should check them:
a clogged coolant filter, a failed cooling fan bearing, an incorrectly set sealing
temperature after a product changeover, and ambient hall temperature above 34 degC in
summer. Record the cause in the shift log; repeated overheating on the same line
within seven days must be escalated to the maintenance supervisor.
""",
    },
    {
        "doc_id": "SOP-204",
        "title": "SOP-204 — Vibration Limits and Bearing Health",
        "text": """
Vibration on rotating equipment is measured in mm/s RMS. For bottling lines the
warning limit is 4.5 mm/s and the critical limit is 6.0 mm/s. For PUMP-07 the warning
limit is 3.5 mm/s and the critical limit is 5.0 mm/s, because the pump runs at a higher
speed and its bearings are smaller.
A slow rise in vibration over several days, rather than a sudden spike, is the classic
signature of bearing wear. A sudden spike with no trend usually means mechanical
looseness, a loose guard, or an unbalanced load rather than a failing bearing.
When vibration crosses the warning limit, schedule a bearing inspection within
48 hours and increase the sampling rate to one minute. When it crosses the critical
limit, stop the equipment at the end of the current batch and inspect before restart.
Bearing replacement on a bottling line takes approximately three hours including
realignment. The replacement interval for the main drive bearing is 8000 running hours
or whenever vibration exceeds the warning limit for three consecutive shifts.
Always record the vibration reading before and after maintenance so the baseline for
anomaly detection can be reset.
""",
    },
    {
        "doc_id": "SOP-318",
        "title": "SOP-318 — Pump Pressure Deviations (PUMP-07)",
        "text": """
PUMP-07 supplies cooling water to Hall 1. Normal discharge pressure is 5.2 to 5.6 bar.
The warning limit is 6.2 bar and the critical limit is 7.0 bar. Pressure below 4.5 bar
is equally serious because the bottling lines lose cooling and will begin to overheat
within twenty minutes.
High pressure usually means a closed or partially closed downstream valve, or a blocked
strainer. Low pressure usually means cavitation, a leaking seal, or a dropping tank
level. Check the tank level first; it is the fastest check and the most common cause.
If PUMP-07 pressure falls below 4.5 bar, treat any temperature alarm on LINE-A or
LINE-B as a consequence of the pump, not as a line fault, and fix the pump first.
The pump has a standby unit, PUMP-08, which can be switched in from the control room
in under two minutes. Switching to the standby pump during production is permitted and
does not require supervisor approval.
""",
    },
    {
        "doc_id": "MAINT-LOG",
        "title": "Maintenance Log — Last 90 Days",
        "text": """
Day 12: LINE-A coolant filter replaced after the warning temperature alarm. Cause was a
clogged filter. Downtime 25 minutes.
Day 31: PUMP-07 mechanical seal replaced following a slow pressure decline. Vibration
baseline reset to 2.1 mm/s after the work.
Day 44: LINE-B sealing temperature found set 6 degC too high after a product changeover.
Operator training issue, not a hardware fault. No parts replaced.
Day 58: LINE-A main drive bearing inspected at 7600 running hours. Wear within limits,
no replacement. Next inspection scheduled at 8000 hours.
Day 73: LINE-B cooling fan bearing replaced after vibration exceeded the warning limit
for three consecutive shifts. Downtime 3 hours 10 minutes.
Day 81: Hall 1 ambient temperature reached 36 degC during a heatwave; both bottling
lines were run at 85 percent speed for two shifts as a precaution.
""",
    },
    {
        "doc_id": "KPI-DEF",
        "title": "KPI Definitions and Reporting Rules",
        "text": """
Throughput is measured in units per minute, averaged over the reporting window. The
target for LINE-A is 110 units per minute and for LINE-B is 100 units per minute.
Availability is the percentage of scheduled production time during which the line was
actually running. The plant target is 92 percent.
A shift report must state, for every line: average throughput against target, the number
of alarms raised by severity, every anomaly detected by the monitoring system with its
suspected cause, and any action taken. Reports are written for the production manager,
who is not an engineer, so each finding must be stated in plain language with the
number that supports it.
Any anomaly left without a stated cause must be listed as open and carried into the
next shift report. Never close a finding without a cause.
""",
    },
]


def generate_metrics(hours: int = 72, step_minutes: int = 10, seed: int = 7) -> list[Point]:
    """Realistic series: daily cycle + noise + planted faults the AI should find."""
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=hours)
    steps = int(hours * 60 / step_minutes)
    points: list[Point] = []

    for (device, metric), (base, noise) in PROFILES.items():
        for i in range(steps):
            ts = start + timedelta(minutes=i * step_minutes)
            progress = i / steps
            # Daily production rhythm.
            cycle = math.sin(2 * math.pi * (ts.hour * 60 + ts.minute) / 1440.0)
            value = base + cycle * (base * 0.015) + rng.gauss(0, noise)

            # Planted fault 1: LINE-B sealing head heats up over the last 10 hours.
            if device == "LINE-B" and metric == "temperature" and progress > 0.86:
                value += (progress - 0.86) / 0.14 * 16.0

            # Planted fault 2: LINE-B bearing wear — slow vibration rise over 2 days.
            if device == "LINE-B" and metric == "vibration" and progress > 0.45:
                value += (progress - 0.45) / 0.55 * 2.4

            # Planted fault 3: PUMP-07 short pressure drop mid-window.
            if device == "PUMP-07" and metric == "pressure" and 0.60 < progress < 0.64:
                value -= 1.3

            # Planted fault 4: LINE-B throughput falls as the line heats up.
            if device == "LINE-B" and metric == "throughput" and progress > 0.88:
                value -= (progress - 0.88) / 0.12 * 22.0

            points.append(Point(device=device, metric=metric, value=round(value, 2), ts=ts))

    return points
