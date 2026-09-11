"""Checks that src/config.py's built-in defaults match .env.example.

A default that drifts from the committed template means a checkout with no .env silently
runs a configuration nobody measured, and nothing else would ever report it.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
ROOT = pathlib.Path(__file__).parent.parent

# HISTORY_DB is built as an absolute path from ROOT, so its code default and the template's
# relative filename are the same file written two ways.
SKIP = {"HISTORY_DB"}
# Secrets are blank in the template by design; they have no code default to compare.
BLANK_BY_DESIGN = {"TWILIO_WEBHOOK_URL"}

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


example = {}
for line in (ROOT / ".env.example").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        example[key.strip()] = re.sub(r"\s+#.*$", "", value).strip()

source = (ROOT / "src" / "config.py").read_text()
defaults = dict(re.findall(r'os\.environ\.get\("(\w+)",\s*([^)]+)\)', source))

for name, raw in defaults.items():
    if name in SKIP or name in BLANK_BY_DESIGN:
        continue
    code = raw.strip().strip('"').strip("'")
    check(name in example, f"{name} has a code default but is missing from .env.example")
    if name in example:
        want = example[name]
        check(code.lower() == want.lower(),
              f"{name}: code default {code!r} but .env.example says {want!r}")

# The tuned values are the point of the exercise, so assert them by name too: a sync check
# alone would pass happily if BOTH files drifted together.
from src import config

for name, want in (("TOP_K", 8), ("RRF_TEXT_WEIGHT", 0.1), ("RRF_TEXT_DEPTH", 5),
                   ("SCORE_FLOOR", 0.20), ("CANDIDATES", 20), ("RRF_K", 60)):
    got = getattr(config, name)
    check(got == want, f"{name} is {got}, but the eval was run at {want} (PLAN §12)")

print(f"compared {len(defaults) - len(SKIP | BLANK_BY_DESIGN)} defaults against .env.example")
print("FAILURES:\n" + "\n".join("  ! " + f for f in fail) if fail else "config: ALL CHECKS PASS")
sys.exit(1 if fail else 0)
