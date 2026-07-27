#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import re
import subprocess
from pathlib import Path

root = Path.cwd()
fixtures = root / "Tests/KataGoCoreTests/Fixtures"
goldens = root / "Tests/KataGoCoreTests/Goldens"
goldens.mkdir(parents=True, exist_ok=True)

komi_half_rules = json.dumps({
    "ko": "POSITIONAL",
    "score": "AREA",
    "tax": "NONE",
    "suicide": True,
    "komi": 0.5,
}, separators=(",", ":"))

cases = [
    ("empty-19", 0, "tromp-taylor", 19),
    ("opening-6", 6, "tromp-taylor", 19),
    ("capture-just-made", 8, "tromp-taylor", 19),
    ("simple-ko", 9, "tromp-taylor", 19),
    ("working-ladder", 0, "tromp-taylor", 19),
    ("consecutive-passes", 3, "tromp-taylor", 19),
    ("chinese-9-nn19", 5, "chinese", 19),
    ("japanese-midgame", 8, "japanese", 19),
    ("komi-0.5", 2, komi_half_rules, 19),
]

def channel_sum(data, channel):
    return sum(data["spatialNHWC"][channel::data["numSpatial"]])

for name, move_num, rules, nn_len in cases:
    sgf_path = fixtures / f"{name}.sgf"
    sgf = sgf_path.read_text()
    size = int(re.search(r"SZ\[(\d+)\]", sgf).group(1))
    setup = [
        [player, coord]
        for prop, player in (("AB", "B"), ("AW", "W"))
        for block in re.findall(rf"{prop}((?:\[[a-z]*\])+)", sgf)
        for coord in re.findall(r"\[([^]]*)\]", block)
    ]
    moves = [[player, coord] for player, coord in re.findall(r";([BW])\[([^]]*)\]", sgf)][:move_num]
    raw = subprocess.check_output([
        str(root / ".tools/dump_v7"), str(sgf_path), str(move_num), rules, str(nn_len)
    ])
    data = json.loads(raw)
    data.update({
        "fixture": f"Fixtures/{name}.sgf",
        "boardXSize": size,
        "boardYSize": size,
        "moveNum": move_num,
        "rules": rules,
        "initialStones": setup,
        "moves": moves,
    })

    if name == "empty-19":
        assert channel_sum(data, 0) == 361 and channel_sum(data, 1) + channel_sum(data, 2) == 0
    elif name == "opening-6":
        assert all(channel_sum(data, channel) == 1 for channel in range(9, 14))
    elif name == "capture-just-made":
        assert channel_sum(data, 1) + channel_sum(data, 2) == len(moves) - 1
    elif name == "simple-ko":
        assert channel_sum(data, 6) > 0
    elif name == "working-ladder":
        assert channel_sum(data, 14) > 0 and channel_sum(data, 17) > 0
    elif name == "consecutive-passes":
        assert any(data["global"][:5]) and data["global"][14] == 1
    elif name == "chinese-9-nn19":
        assert channel_sum(data, 0) == 81 and data["global"][6] == 0 and data["global"][8] == 0
    elif name == "japanese-midgame":
        assert data["global"][9] == 1 and data["global"][10] == 1
    elif name == "komi-0.5":
        assert abs(data["global"][5] - (-0.025)) < 1e-6 and abs(data["global"][18] - 0.5) < 1e-6

    (goldens / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"validated and wrote {name}.json")
PY
