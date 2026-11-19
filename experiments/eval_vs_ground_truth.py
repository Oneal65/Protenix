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

"""Compare Protenix predictions against RCSB experimental structures.

Simplified evaluation for the beginner-facing experiment log (not PXMeter):
  * chains matched by residue-name sequence, homomeric copies resolved by
    brute-force permutation (minimum representative-atom RMSD);
  * atoms matched by (chain, label_seq_id, atom_name), no symmetric-atom swap;
  * lDDT: all matched heavy atoms, R0 = 15 A, thresholds 0.5/1/2/4 A;
  * interface lDDT: same, restricted to atom pairs from different chains;
  * ligand RMSD: superpose on pocket CA (<10 A of the ligand in the truth).

Usage: python eval_vs_ground_truth.py <exp_dir> [<exp_dir> ...] > out.json
"""

import glob
import itertools
import json
import os
import sys

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

GT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
REP_ATOMS = ("CA", "C1'")


def load(path, gt=False):
    cif = pdbx.CIFFile.read(path)
    atoms = pdbx.get_structure(cif, model=1, use_author_fields=False)
    atoms = atoms[(atoms.element != "H") & (atoms.element != "D")]
    atoms = atoms[atoms.res_name != "HOH"]
    if gt:
        # ligands carry label_seq_id "." -> -1; give them residue 1 like the prediction
        atoms.res_id[atoms.hetero] = 1
    return atoms


def chain_table(atoms):
    """chain_id -> ({label_seq_id: res_name}, is_polymer)"""
    out = {}
    for c in np.unique(atoms.chain_id):
        m = atoms[atoms.chain_id == c]
        ids, names = struc.get_residues(m)
        out[c] = (dict(zip(ids.tolist(), names)), not m.hetero[0])
    return out


def entity_groups(pred_tab, gt_tab):
    """Pair each GT chain with the pred chains that could be it."""
    groups = {}
    for gc, (gres, gpoly) in gt_tab.items():
        cands = []
        for pc, (pres, ppoly) in pred_tab.items():
            if ppoly != gpoly:
                continue
            # GT observes a subset of SEQRES; every observed position must agree
            same = sum(pres.get(i) == n for i, n in gres.items())
            if same >= 0.9 * len(gres):
                cands.append(pc)
        if cands:
            groups.setdefault(tuple(sorted(cands)), []).append(gc)
    return groups


def match_atoms(pred, gt, mapping):
    """Return index arrays (pi, gi) of matched atoms under chain mapping gt->pred."""
    pkey = {
        (c, r, a): i
        for i, (c, r, a) in enumerate(zip(pred.chain_id, pred.res_id, pred.atom_name))
    }
    pi, gi = [], []
    for j, (c, r, a, rn) in enumerate(
        zip(gt.chain_id, gt.res_id, gt.atom_name, gt.res_name)
    ):
        if c not in mapping:
            continue
        i = pkey.get((mapping[c], r, a))
        if i is not None and pred.res_name[i] == rn:
            pi.append(i)
            gi.append(j)
    return np.array(pi, dtype=int), np.array(gi, dtype=int)


def kabsch_rmsd(x, y):
    """RMSD of x onto y after optimal superposition; also returns transform."""
    xc, yc = x.mean(0), y.mean(0)
    h = (x - xc).T @ (y - yc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    xt = (x - xc) @ rot.T + yc
    return float(np.sqrt(((xt - y) ** 2).sum(1).mean())), (xc, rot, yc)


def apply(t, x):
    xc, rot, yc = t
    return (x - xc) @ rot.T + yc


def lddt(xp, xg, chain, res, inter_only=False, r0=15.0):
    dg = np.linalg.norm(xg[:, None] - xg[None], axis=-1)
    dp = np.linalg.norm(xp[:, None] - xp[None], axis=-1)
    same_res = (chain[:, None] == chain[None]) & (res[:, None] == res[None])
    mask = (dg < r0) & ~same_res
    if inter_only:
        mask &= chain[:, None] != chain[None]
    if mask.sum() == 0:
        return None
    err = np.abs(dg - dp)[mask]
    return float(np.mean([(err < t).mean() for t in (0.5, 1, 2, 4)]))


def best_mapping(pred, gt, groups):
    """Brute-force homomeric permutations; score = representative-atom RMSD."""
    choices = []
    for cands, gchains in groups.items():
        choices.append(
            [dict(zip(gchains, p)) for p in itertools.permutations(cands, len(gchains))]
        )
    best = None
    for combo in itertools.product(*choices):
        mapping = {k: v for d in combo for k, v in d.items()}
        pi, gi = match_atoms(pred, gt, mapping)
        rep = np.isin(gt.atom_name[gi], REP_ATOMS) & ~gt.hetero[gi]
        rmsd, _ = kabsch_rmsd(pred.coord[pi[rep]], gt.coord[gi[rep]])
        # ligands/ions: tie-break by distance after the polymer superposition
        if best is None or rmsd < best[0] - 1e-6:
            best = (rmsd, mapping)
        elif abs(rmsd - best[0]) <= 1e-6:
            best = min(best, (rmsd, mapping), key=lambda b: lig_score(pred, gt, b[1]))
    return best


def lig_score(pred, gt, mapping):
    pi, gi = match_atoms(pred, gt, mapping)
    rep = np.isin(gt.atom_name[gi], REP_ATOMS) & ~gt.hetero[gi]
    _, t = kabsch_rmsd(pred.coord[pi[rep]], gt.coord[gi[rep]])
    het = gt.hetero[gi]
    return float(np.linalg.norm(apply(t, pred.coord[pi[het]]) - gt.coord[gi[het]], axis=1).mean())


def eval_sample(pred_path, gt):
    pred = load(pred_path)
    groups = entity_groups(chain_table(pred), chain_table(gt))
    rmsd, mapping = best_mapping(pred, gt, groups)
    pi, gi = match_atoms(pred, gt, mapping)
    xp, xg = pred.coord[pi], gt.coord[gi]
    chain, res = gt.chain_id[gi], gt.res_id[gi]
    out = {
        "rep_atom_rmsd": round(rmsd, 3),
        "lddt": round(lddt(xp, xg, chain, res), 4),
        "interface_lddt": round(lddt(xp, xg, chain, res, inter_only=True) or float("nan"), 4),
        "n_matched_atoms": int(len(pi)),
        "ligands": {},
    }
    # ligand RMSD after pocket superposition
    poly_ca = (~gt.hetero[gi]) & np.isin(gt.atom_name[gi], REP_ATOMS)
    for gc in np.unique(chain[gt.hetero[gi]]):
        lig = chain == gc
        d = np.linalg.norm(xg[poly_ca][:, None] - xg[lig][None], axis=-1).min(1)
        pocket = poly_ca.copy()
        pocket[poly_ca] = d < 10.0
        if pocket.sum() < 3:
            continue
        _, t = kabsch_rmsd(xp[pocket], xg[pocket])
        lr = np.sqrt(((apply(t, xp[lig]) - xg[lig]) ** 2).sum(1).mean())
        out["ligands"][f"{gt.res_name[gi][lig][0]}:{gc}"] = round(float(lr), 3)
    return out


def main():
    results = {}
    for exp in sys.argv[1:]:
        results[exp] = {}
        for sdir in sorted(glob.glob(os.path.join(exp, "*", "seed_*", "predictions"))):
            name = sdir.split(os.sep)[-3]
            seed = sdir.split(os.sep)[-2]
            gt = load(os.path.join(GT_DIR, f"{name}.cif"), gt=True)
            rows = []
            for cif in sorted(glob.glob(os.path.join(sdir, f"{name}_sample_*.cif"))):
                k = cif.rsplit("_", 1)[-1].split(".")[0]
                conf = json.load(open(os.path.join(sdir, f"{name}_summary_confidence_sample_{k}.json")))
                row = {"sample": int(k)}
                row.update({m: round(float(conf[m]), 4) for m in ("ranking_score", "plddt", "ptm", "iptm")})
                row.update(eval_sample(cif, gt))
                rows.append(row)
            results[exp][f"{name}/{seed}"] = rows
            print(f"[done] {exp} {name} {seed}", file=sys.stderr)
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
