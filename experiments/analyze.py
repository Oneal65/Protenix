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

"""Turn PXMeter results (pxm_all.json) into the markdown tables of the report.

Usage: python analyze.py pxm_all.json > analysis.md
"""

import glob
import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)
N_BOOT = 2000
POLYMER_IFACES = {
    "7r6r": ["A,B", "A,C"],  # protein-DNA
    "7wux": ["A,B", "A,C", "A,D", "B,C", "B,D", "C,D"],
    "7pzb": ["A,B", "A,C", "A,D", "A,E", "A,F", "B,C", "B,D", "B,E", "B,F"],
}
LIG_OK = 2.0  # pocket-aligned ligand RMSD success threshold (A)
DOCKQ_OK = 0.23


def iface_dockq(row, name):
    v = [row["interfaces"][k]["dockq"] for k in POLYMER_IFACES[name] if k in row["interfaces"]]
    return float(np.mean(v))


def lig_ok(row, chains):
    return float(np.mean([row["ligands"][c] < LIG_OK for c in chains]))


def seeds_of(d, exp, name):
    return {k.split("/")[1]: v for k, v in d[exp].items() if k.startswith(name + "/")}


def top1(rows, key="ranking_score"):
    return max(rows, key=lambda r: r[key])


def design_table(d):
    exps = [
        ("E1_base_v1", "base v1.0.0（10 cycle / 200 step）"),
        ("E5a_base_v05_full", "base v0.5.0（10 / 200）"),
        ("E5b_base_v05_miniBudget", "base v0.5.0（4 / 5，mini 同预算）"),
        ("E5c_mini_v05_fullBudget", "mini v0.5.0（10 / 200，base 同预算）"),
        ("E2_mini_v05", "mini v0.5.0（4 / 5）"),
        ("E3_base_v1_noMSA", "base v1.0.0 无 MSA（仅 seed 101）"),
    ]
    out = ["| 模型（cycle / step） | 7r6r lDDT | 7r6r 蛋白-DNA DockQ | 7wux lDDT | 7wux 聚合物界面 DockQ "
           "| 7wux 6OI 成功率 | 7pzb lDDT | 7pzb 聚合物界面 DockQ | 7pzb cGMP 成功率 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for exp, label in exps:
        cells = []
        for name in ("7r6r", "7wux", "7pzb"):
            per_seed = seeds_of(d, exp, name)
            t = [top1(v) for v in per_seed.values()]
            allr = [r for v in per_seed.values() for r in v]

            def fmt(xs):
                xs = np.array(xs)
                return f"{xs.mean():.3f}" + (f" [{xs.min():.3f}–{xs.max():.3f}]" if len(xs) > 1 else "")

            cells += [fmt([r["lddt"] for r in t]), fmt([iface_dockq(r, name) for r in t])]
            if name == "7wux":
                cells.append(f"{np.mean([lig_ok(r, ['G', 'J']) for r in allr]):.0%}（{len(allr)} 个样本）")
            if name == "7pzb":
                cells.append(f"{np.mean([lig_ok(r, ['G', 'H']) for r in allr]):.0%}（{len(allr)} 个样本）")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("数值：每个 seed 取 ranking_score 第一的样本，显示 3 个 seed 的均值 [最小–最大]；"
               "成功率：所有样本中配体口袋对齐 RMSD < 2 Å 的比例。")
    return "\n".join(out)


def scaling(rows_by_seed, metric, ok=None, ks=(1, 2, 5, 10, 20, 40)):
    seeds = list(rows_by_seed)
    pool = [r for v in rows_by_seed.values() for r in v]
    out = []
    for k in ks:
        if k > len(seeds):
            continue
        picked, best = [], []
        for _ in range(N_BOOT if k < len(seeds) else 1):
            sub = RNG.choice(len(seeds), k, replace=False)
            rows = [r for i in sub for r in rows_by_seed[seeds[i]]]
            picked.append(metric(top1(rows)))
            best.append(max(metric(r) for r in rows))
        picked, best = np.array(picked), np.array(best)
        row = {"k": k, "n": 5 * k, "picked": picked.mean(), "oracle": best.mean()}
        if ok is not None:
            row["picked_ok"] = (picked > ok).mean()
            row["oracle_ok"] = (best > ok).mean()
        out.append(row)
    all_m = np.array([metric(r) for r in pool])
    rs = np.array([r["ranking_score"] for r in pool])
    corr = float(np.corrcoef(rs, all_m)[0, 1])
    return out, all_m, corr


def scaling_table(title, res, all_m, corr, ok_name=None):
    head = "| seed 数 k | 样本数 | 模型自选 top-1 的均值 | 池中最好（oracle）均值 |"
    sep = "|---|---|---|---|"
    if ok_name:
        head += f" 自选 top-1 成功率（{ok_name}） | oracle 成功率 |"
        sep += "---|---|"
    lines = [f"**{title}**", "", head, sep]
    for r in res:
        line = f"| {r['k']} | {r['n']} | {r['picked']:.3f} | {r['oracle']:.3f} |"
        if ok_name:
            line += f" {r['picked_ok']:.0%} | {r['oracle_ok']:.0%} |"
        lines.append(line)
    lines.append("")
    extra = f"，其中成功（{ok_name}）比例 {np.mean(all_m > 0.23):.0%}" if ok_name else ""
    lines.append(f"全部 {len(all_m)} 个样本：均值 {all_m.mean():.3f}，范围 {all_m.min():.3f}–{all_m.max():.3f}{extra}；"
                 f"ranking_score 与该指标的 Pearson 相关 {corr:.2f}。")
    return "\n".join(lines)


def attach_ag_pair_iptm(rows_by_seed, exp, name, ag_idx):
    """Max chain-pair ipTM between the antigen chain and either antibody chain."""
    for seed, rows in rows_by_seed.items():
        sdir = os.path.join(exp, name, seed, "predictions")
        for r in rows:
            c = json.load(open(os.path.join(sdir, f"{name}_summary_confidence_sample_{r['sample']}.json")))
            r["ag_pair_iptm"] = max(v for i, v in enumerate(c["chain_pair_iptm"][ag_idx]) if i != ag_idx)


def abag_selector_table(rows_by_seed):
    pool = [r for v in rows_by_seed.values() for r in v]
    dq = np.array([r["ab_ag_dockq"] for r in pool])
    lines = ["| 排序依据 | 与 DockQ 的 Pearson | 每 seed top-1 的平均 DockQ | 每 seed top-1 成功率 | 全池 top-1 的 DockQ |",
             "|---|---|---|---|---|"]
    for key, label in (("ranking_score", "ranking_score（0.8·ipTM+0.2·pTM）"),
                       ("ag_pair_iptm", "抗原-抗体 chain_pair_iptm")):
        s = np.array([r[key] for r in pool])
        t = [top1(v, key)["ab_ag_dockq"] for v in rows_by_seed.values()]
        lines.append(f"| {label} | {np.corrcoef(s, dq)[0, 1]:.2f} | {np.mean(t):.3f} "
                     f"| {np.mean(np.array(t) > DOCKQ_OK):.0%} | {top1(pool, key)['ab_ag_dockq']:.3f} |")
    return "\n".join(lines)


def ligand_ranking(d, exps):
    """For 7wux 6OI: does ranking by the ligand's own chain ipTM pick better?"""
    lines = ["| 模型 | 样本数 | 6OI 成功率（全部样本） | 按 ranking_score 选（每 seed top-1） "
             "| 按 6OI 链 ipTM 选（每 seed top-1） |", "|---|---|---|---|---|"]
    for exp, label in exps:
        per_seed = seeds_of(d, exp, "7wux")
        a, b, n = [], [], 0
        for seed, rows in per_seed.items():
            sdir = glob.glob(os.path.join(exp, "7wux", seed, "predictions"))[0]
            for r in rows:
                c = json.load(open(os.path.join(sdir, f"7wux_summary_confidence_sample_{r['sample']}.json")))
                # prediction chain order: A,B,C,D protein, E P4G, F,G 6OI, H,I,J MG
                r["lig_iptm"] = (c["chain_iptm"][5] + c["chain_iptm"][6]) / 2
            a.append(lig_ok(top1(rows), ["G", "J"]))
            b.append(lig_ok(top1(rows, "lig_iptm"), ["G", "J"]))
            n += len(rows)
        allr = [r for v in per_seed.values() for r in v]
        lines.append(f"| {label} | {n} | {np.mean([lig_ok(r, ['G', 'J']) for r in allr]):.0%} "
                     f"| {np.mean(a):.0%}（{len(a)} 个 seed） | {np.mean(b):.0%} |")
    return "\n".join(lines)


def main():
    d = json.load(open(sys.argv[1]))
    print("## 2×2 设计\n")
    print(design_table(d))
    print("\n## 多采样\n")
    e4 = seeds_of(d, "E4_seed_scaling", "7r6r")
    res, m, c = scaling(e4, lambda r: iface_dockq(r, "7r6r"))
    print(scaling_table("7r6r（简单题，10 seed）· 指标：蛋白-DNA 界面 DockQ", res, m, c))
    # antigen index in prediction chain order (JSON entity order)
    for name, desc, ag_idx in (("8eq6", "PD-1 + Fab", 0), ("9jbq", "PcrV + Fab", 2)):
        rs = seeds_of(d, "E6_abag", name)
        if not rs:
            continue
        res, m, c = scaling(rs, lambda r: r["ab_ag_dockq"], ok=DOCKQ_OK)
        print()
        print(scaling_table(f"{name}（{desc}，{len(rs)} seed）· 指标：抗体-抗原 DockQ", res, m, c, "DockQ>0.23"))
        attach_ag_pair_iptm(rs, "E6_abag", name, ag_idx)
        print()
        print(abag_selector_table(rs))
    print("\n## 配体排序\n")
    print(ligand_ranking(d, [("E1_base_v1", "base v1.0.0"), ("E5a_base_v05_full", "base v0.5.0"),
                             ("E5c_mini_v05_fullBudget", "mini v0.5.0（10 / 200）"),
                             ("E2_mini_v05", "mini v0.5.0（4 / 5）")]))


if __name__ == "__main__":
    main()
