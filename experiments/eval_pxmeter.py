# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate Protenix predictions against RCSB structures with PXMeter.

Runs inside the isolated venv (/root/venvs/pxmeter) because PXMeter pins
gemmi/pdbeccdutils versions that conflict with Protenix.

Usage: /root/venvs/pxmeter/bin/python eval_pxmeter.py <exp_dir> [...] > out.json
"""

import glob
import json
import os
import sys

from joblib import delayed, Parallel
from pxmeter.eval import evaluate

GT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
# label_asym_id of ligands to score (only those present in the input JSON)
LIGANDS = {"7wux": ["F", "G", "J"], "7pzb": ["G", "H"]}
# antigen label_asym_id for antibody-antigen targets
ANTIGEN = {"8eq6": "A", "9jbq": "C"}


def eval_one(name, cif, conf_path):
    res = evaluate(
        ref_cif=os.path.join(GT_DIR, f"{name}.cif"),
        model_cif=cif,
        interested_lig_label_asym_id=LIGANDS.get(name),
    ).to_json_dict()
    conf = json.load(open(conf_path))
    row = {m: round(float(conf[m]), 4) for m in ("ranking_score", "plddt", "ptm", "iptm")}
    row["sample"] = int(cif.rsplit("_", 1)[-1].split(".")[0])
    row["lddt"] = res["complex"]["lddt"]
    row["interfaces"] = {
        k: {"lddt": v.get("lddt"), "dockq": v.get("dockq")}
        for k, v in res.get("interface", {}).items()
    }
    row["ligands"] = {
        k: v["lig_rmsd"] for k, v in res.get("chain", {}).items() if "lig_rmsd" in v
    }
    if name in ANTIGEN:
        ag = ANTIGEN[name]
        dq = [v["dockq"] for k, v in row["interfaces"].items() if ag in k.split(",")]
        row["ab_ag_dockq"] = max(dq) if dq else 0.0
    return row


def main():
    jobs = []
    for exp in sys.argv[1:]:
        for sdir in sorted(glob.glob(os.path.join(exp, "*", "seed_*", "predictions"))):
            name, seed = sdir.split(os.sep)[-3], sdir.split(os.sep)[-2]
            for cif in sorted(glob.glob(os.path.join(sdir, f"{name}_sample_*.cif"))):
                k = cif.rsplit("_", 1)[-1].split(".")[0]
                conf = os.path.join(sdir, f"{name}_summary_confidence_sample_{k}.json")
                jobs.append((exp, f"{name}/{seed}", name, cif, conf))
    rows = Parallel(n_jobs=32)(delayed(eval_one)(n, c, f) for _, _, n, c, f in jobs)
    out = {}
    for (exp, key, *_), row in zip(jobs, rows):
        out.setdefault(exp, {}).setdefault(key, []).append(row)
    for v in out.values():
        for rs in v.values():
            rs.sort(key=lambda r: r["sample"])
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
