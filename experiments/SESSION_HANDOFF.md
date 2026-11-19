# Session handoff — Protenix beginner report + local experiments

Last updated 2026-11-19. Written by Claude at the end of the session so the next one
can pick up without re-deriving context. The user writes in Chinese; reply in Chinese.

## What the user asked for (in order)

1. Improve `CLAUDE.md` (`/init`) — done: kernel-table defaults (`cuequivariance`) and the
   CI-only-checks-E9/F63/F7/F82 note.
2. `/deep-research` (ultrathink): explain Protenix to a biology beginner — purpose,
   philosophy, inputs/outputs, implementation idea — as a local markdown file.
3. Run experiments **on GPU 6 and GPU 7 only** and record the process in markdown.
4. Self-review the experiments, then fix everything found ("需要, 请直接开始做").
5. Save this session so a future Claude can load it (this file).

## Deliverable

`docs/protenix_beginner_guide_and_experiments_zh.md` (Chinese):
- Part 1: project intro (✅ verified / 📘 general knowledge / ⚠️ self-reported tags).
- Part 2: first-round experiments E1–E4 (2026-11-18), with 【更正】 marks where Part 3
  overturned them. Don't silently rewrite Part 2 — corrections are marked inline on purpose.
- Part 3: supplementary experiments (2026-11-19), §18 is the full error/correction table.

The deep-research raw output lives only in the old session transcript; the report already
contains everything used from it (verified claims, caveats, open questions in appendix C).

## Environment facts (this machine)

- 8× H20; GPUs 0–4 are used by someone else — **only use 6 and 7**.
- No direct internet. Downloads: `curl -x http://star-proxy.oa.com:3128 ...`;
  PyPI: `pip install -i https://mirrors.tencent.com/pypi/simple/ ...`;
  for Python code that downloads (e.g. `protenix msa`), export `https_proxy`/`http_proxy`
  to the same proxy.
- System Python 3.12, torch **2.8.0+cu128** (repo pins 2.7.1 — intentionally not
  downgraded). Protenix installed `pip install --no-deps --no-build-isolation -e .`;
  other requirements installed into **system** Python (minus torch/triton/deepspeed/
  cuequivariance/numpy/wandb). Rolling that back is an **open decision for the user**
  (it might affect the jobs on GPUs 0–4) — do not do it unasked.
- PXMeter 1.1.1 is in an isolated venv `/root/venvs/pxmeter` (`--system-site-packages`)
  because it pins gemmi 0.7.0 / pdbeccdutils 0.8.5 vs Protenix's 0.6.7 / 1.0.0.
- `PROTENIX_ROOT_DIR` unset → `~`. Checkpoints in `~/checkpoint/`:
  base_default_v1.0.0, base_default_v0.5.0, mini_default_v0.5.0. CCD cache in `~/common/`.
- cuequivariance not installed → always pass `--trimul_kernel torch --triatt_kernel triattention`.
- No kalign/hmmer/seqres DB → templates and RNA MSA always off. Protein MSA either from
  `examples/` or the Protenix MSA server (`protenix msa`, needs proxy env).
- No `/usr/bin/time`; use `experiments/run.sh <gpu> <tag> <protenix pred args>` or
  `experiments/run_raw.sh <gpu> <tag> <runner/inference.py dotted args>` (logs → `experiments/logs/`).

## Experiments (all under `experiments/`, all finished rc=0)

| dir | model / setting | targets | seeds |
|---|---|---|---|
| E1_base_v1 | base v1.0.0, 10 cycle / 200 step, MSA, no template | 7r6r 7wux 7pzb | 101–103 |
| E2_mini_v05 | mini v0.5.0, 4 / 5 (default) | same | 101–103 |
| E3_base_v1_noMSA | base v1.0.0, `--use_msa false` | same | 101 |
| E4_seed_scaling | base v1.0.0 | 7r6r | 101–110 |
| E5a_base_v05_full | base v0.5.0, 10 / 200 | same 3 | 101–103 |
| E5b_base_v05_miniBudget | base v0.5.0 weights, 4 / 5, gamma0=0 (runner/inference.py) | same 3 | 101–103 |
| E5c_mini_v05_fullBudget | mini v0.5.0, 10 / 200 | same 3 | 101–103 |
| E6_abag | base v1.0.0, server MSA | 8eq6 (PD-1+Fab), 9jbq (PcrV+Fab) | 201–240 |
| K1_kernel_torch | base v1.0.0, `--triatt_kernel torch` | 7r6r | 101 |

Ground truth: `experiments/ground_truth/{7r6r,7wux,7pzb,8eq6,9jbq}.cif` (RCSB).

Evaluation pipeline (re-run after adding experiments):
```bash
cd /root/Protenix/experiments
/root/venvs/pxmeter/bin/python eval_pxmeter.py E1_base_v1 E2_mini_v05 E3_base_v1_noMSA \
  E4_seed_scaling E5a_base_v05_full E5b_base_v05_miniBudget E5c_mini_v05_fullBudget E6_abag > pxm_all.json
python3 analyze.py pxm_all.json > analysis.md     # all Part-3 tables
```
`eval_pxmeter.py` has per-target ligand IDs (`LIGANDS`) and antigen chain (`ANTIGEN`) tables;
`analyze.py` has interface lists and the antigen index in prediction chain order — extend
them for new targets. `eval_vs_ground_truth.py` / `all_eval.json` are the superseded
first-round evaluator, kept for history.

## Key findings (details + numbers in report Part 3)

- MSA is essential: without it every interface DockQ < 0.07.
- Base weights cannot do 5-step sampling (chain breaks, Cα–Cα 9.6 Å); mini's few-step
  ability is in its weights. Mini gains nothing from 10/200. Mini default ≈ 8× faster than base v1.
- Size-only (v0.5, 10/200): base > mini on all 3 targets, biggest on protein–DNA.
- v1 vs v0.5 (no templates): mixed — v0.5 alone gets the 7wux homodimer interface right
  (DockQ 0.92 vs ~0.32).
- Ab–Ag (200 samples each): correct poses exist (2–4%; oracle hits 100% at 40 seeds) but
  ranking_score / chain_pair_iptm are negatively correlated with DockQ → model-picked
  success 0%. ipTM 0.66–0.76 (grey zone), not "confidently wrong".
- 7r6r: random-resampled scaling gives a small monotonic gain (overturned Part 2's claim).
- Ligand 6OI: selecting by the ligand chain's own `chain_iptm` beats ranking_score.
- Kernel swap differences are within same-kernel rerun noise; runs are not bitwise reproducible.

## Pitfalls hit this session (don't repeat)

- `pgrep -f`/`pkill -f` with a pattern that appears in your own command line matches
  itself (stuck wait loop; killed own shell). Wait on a log line instead:
  `until grep -q "^\[end\]" experiments/logs/X.log; do sleep 15; done`.
- `protenix pred` auto-writes `<input>-update-msa.json` next to the input and
  `protenix/model/tri_attention/TriAttentionCache/` (both untracked artefacts).
- MSA server output dirs are not in chain order (`msa/1` may be entity 1) — verify each
  a3m query line matches its chain sequence.
- PXMeter drops reference chains absent from the input (e.g. 7wux PGE chain E); only pass
  `-l` ligand IDs that were actually predicted.

## Open items / possible next steps

- User decision pending: roll back the system-Python packages installed for Protenix?
- Nothing is committed to git (CLAUDE.md, the report, `experiments/`, artefacts are untracked).
- Not done: templates (needs hmmer/kalign + PDB seqres), cuequivariance kernel,
  constraint model (could test epitope `pocket` constraints on 8eq6/9jbq — the natural
  follow-up to "ranking, not sampling, is the bottleneck"), PoseBusters checks, more Ab–Ag targets.
