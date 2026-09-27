#!/usr/bin/env python3
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
SBML = os.path.join(ROOT, "penkin", "data", "iAL1006.xml")
OUT = os.path.join(RESULTS_DIR, "gene_reaction_map.json")
TARGETS = ["r0747", "r0803", "r0813"]

text = open(SBML, encoding="utf-8", errors="ignore").read()
out = {}
for rid in TARGETS:
    m = re.search(rf'<reaction[^>]*id="R_{rid}"[\s\S]*?</reaction>', text)
    if not m:
        continue
    block = m.group(0)
    name = re.search(r'name="([^"]+)"', block)
    genes = re.findall(r'fbc:geneProduct="([^"]+)"', block)
    notes = re.findall(r'<p>(.*?)</p>', block, flags=re.S)
    notes = [re.sub(r'<[^>]+>', '', n).strip() for n in notes]
    out[rid] = {
        "name": name.group(1) if name else None,
        "gene_products": sorted(set(genes)),
        "notes": notes,
    }

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2)
print(OUT)
