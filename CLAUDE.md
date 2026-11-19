# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Protenix is ByteDance's trainable PyTorch reproduction of AlphaFold 3 (biomolecular
structure prediction). Module and function docstrings cite AF3 algorithm numbers;
when in doubt about intent, the AF3 supplement is the spec this code follows.

## Commands

```bash
pip3 install -e .            # dev install (Python >= 3.11); setup.py --cpu drops the nvidia/cuda pins
pytest tests/                # what CI runs
pytest tests/test_frame.py -k test_name    # single test (tests are unittest.TestCase run under pytest)
pre-commit run --all-files   # flake8 (+bugbear/pep8-naming/torchfix), ufmt (black 22.12 + usort), pydoclint, insert-license
```

CI (`.github/workflows/ci.yml`, Python 3.11) only fails flake8 on syntax errors and
undefined names (`E9,F63,F7,F82`); the stricter `.flake8` rules (max line 120) are
enforced locally by pre-commit alone.

`pre-commit`'s `insert-license` hook prepends `assets/license_header.txt` to every
new `.py` and `.sh` — add the header yourself when creating files.

Running from a source checkout (not the installed wheel) needs
`export PYTHONPATH="${PYTHONPATH}:$(pwd)"`, as the demo scripts do.

### Running the model

Two front doors, both ending in `runner/inference.py`:

```bash
# packaged CLI (setup.py console_script -> runner.batch_inference:protenix_cli)
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0 --use_default_params true
protenix prep  # MSA + template + RNA-MSA search, writes features into the input JSON
protenix json  # FASTA / PDB / CIF -> input JSON;  protenix msa | mt  -> search only

# raw runner: GPU compute only, input JSON must already carry MSA/template features
python3 runner/inference.py --input_json_path examples/input.json --model_name protenix_base_default_v1.0.0 \
    --model.N_cycle 10 --sample_diffusion.N_step 200 --use_template true

bash train_demo.sh / finetune_demo.sh / inference_demo.sh   # worked examples of every flag combination
```

`runner/*.py` take **dotted flags that mirror the config tree** (`--model.N_cycle`,
`--data.posebusters_0925.base_info.max_n_token`); `protenix pred` instead takes
short click options and maps them onto the same config. `inference_demo.sh` is the
most complete catalogue of model names × kernel × feature combinations.

Environment: `PROTENIX_ROOT_DIR` (data/checkpoint root, defaults to `~`),
`CUTLASS_PATH` (only for the DeepSpeed attention kernel). Template/RNA-MSA search
shells out to `kalign` and `hmmer`.

## Architecture

### Config is the spine

Every entry point builds one `ml_collections.ConfigDict` and `Protenix.__init__`
instantiates the entire network from it. Adding a model variant is usually a new
entry in `configs/configs_model_type.py`, not new code.

* `configs/configs_base.py` — `basic/data/optim/model/perm/loss` defaults merged into
  `configs` at the bottom; `configs/configs_data.py` — datasets, cropping, sampler,
  MSA/template/CCD paths, all rooted at `PROTENIX_ROOT_DIR`;
  `configs/configs_inference.py` — inference-only knobs.
* `configs/configs_model_type.py` — the model zoo: per-checkpoint **deltas** from base
  (block counts, `N_cycle`, `esm.enable`, `use_msa`, `load_strict`). Its module
  docstring is the authoritative size/feature table.
* `protenix/config/config.py` (`ConfigManager`, `parse_configs`) flattens the nested
  defaults to dotted keys and registers one argparse flag per leaf, then coerces types
  back. `protenix/config/extend_types.py` supplies the sentinels used in the default
  dicts: `RequiredValue`, `ValueMaybeNone`, `DefaultNoneWithType`, `ListValue`, and
  `GlobalConfigValue` (nested key inherits a top-level one, e.g. `crop_size` ←
  `train_crop_size`).
* **Two-pass CLI parse**, identical in `runner/train.py:main` and `runner/inference.py:run`:
  parse once only to learn `model_name`, deep-merge that model's deltas into the base
  defaults, then re-parse so explicit flags win. Preserve this when touching config plumbing.

### Model (`protenix/model/`)

`Protenix(nn.Module)` in `protenix/model/protenix.py` owns everything; there is no
separate trunk class. `get_pairformer_output` runs embedders → N_cycle recycling of
(template embedder → MSA module → pairformer stack), with gradients enabled only on
the last cycle. `forward` dispatches on `mode in {train, eval, inference}` to
`main_train_loop` / `main_inference_loop`.

* Blocks live in `protenix/model/modules/` (`embedders`, `pairformer`, `transformer`,
  `diffusion`, `confidence`, `head`), triangle ops in `protenix/model/triangular/`.
* Diffusion sampling and noise schedules: `protenix/model/generator.py`
  (`sample_diffusion` is also where TFG guidance hooks in).
* Losses are all in `protenix/model/loss.py`, aggregated by `ProtenixLoss`, and are
  invoked **only from the trainer**, never inside `Protenix`.
* Confidence logits → pLDDT/PAE/PTM/ipTM/clash summaries and ranking scores:
  `protenix/model/sample_confidence.py`.

### Kernels are orthogonal switches, each with a pure-torch fallback

Changing one must not change numerics. Relevant when debugging "works on my GPU":

| switch | where | options |
|---|---|---|
| `LAYERNORM_TYPE` (env) | `protenix/model/triangular/layers.py` picks `FusedLayerNorm` vs `OpenFoldLayerNorm` | `fast_layernorm` (default, JIT-compiles `protenix/model/layer_norm/kernel/*`) / `torch` |
| `triangle_attention` (config / `--triatt_kernel`; env `TRIANGLE_ATTENTION` in `runner/train.py` only) | `triangular/layers.py` | `cuequivariance` (default) / `triattention` (Triton, `protenix/model/tri_attention/`) / `deepspeed` (needs `CUTLASS_PATH`) / `torch` |
| `triangle_multiplicative` (config / `--trimul_kernel`; env `TRIANGLE_MULTIPLICATIVE` in `runner/train.py` only) | `triangular/triangular.py` | `cuequivariance` (default) / `torch` |
| `FUSED_DROPOUT_RESIDUAL` (env) | `protenix/model/modules/fused_ops.py` | Triton fused dropout+residual |

`update_gpu_compatible_configs` in `runner/inference.py` silently downgrades kernels on
pre-Ampere GPUs, and `protenix/model/tri_attention/__init__.py` falls back to plain
PyTorch SDPA when Triton is unavailable. Also config-gated:
`enable_diffusion_shared_vars_cache`, `enable_efficient_fusion`, and `skip_amp.*`
(which blocks run outside autocast). See `docs/kernels.md`.

### Data (`protenix/data/`)

Two front-ends converging on one featurizer core:

* **Inference (JSON)** — `inference/infer_dataloader.py:process_one` is the canonical
  ordering: `inference/json_parser.py` builds biotite `AtomArray`s per entity (RDKit for
  ligands/SMILES) → `inference/json_to_feature.py:SampleDictToFeatures` adds
  inter-entity bonds and annotations → `get_feature_dict` runs **tokenizer → constraint
  generator → featurizer → frames** → MSA/template/ESM features are appended → missing
  modalities zero-filled by `make_dummy_feature`.
* **Training (mmCIF)** — `pipeline/dataset.py` (`BaseSingleDataset` over precomputed
  bioassembly pickles, cropping via `protenix/utils/cropping.py`, weighted/cluster
  samplers in `pipeline/dataloader.py`), with `core/parser.py` + `core/filter.py` doing
  the mmCIF → bioassembly conversion.
* **Tokenization** follows the AF3 rule (`data/tokenizer.py`): one token per standard
  polymer residue, one token per atom for ligands and non-standard residues; a
  `TokenArray` parallels biotite's `AtomArray`. `data/core/featurizer.py` turns
  `(token_array, atom_array)` into the model feature dict.
* **CCD** (`data/core/ccd.py`) reads `$PROTENIX_ROOT_DIR/common/components.cif(+.rdkit_mol.pkl)`
  through `lru_cache`d accessors; `runner/inference.py:download_inference_cache` fetches them.
* MSA (`data/msa/`), templates (`data/template/`) and constraints (`data/constraint/`)
  each follow the same shape: a source manager, a feature assembly line, a training
  featurizer, and an `Inference*Featurizer` that reads features straight from the JSON.
* `data/tools/` wraps external binaries (hmmer family, kalign) and monkeypatches biotite.

### Runners

`runner/train.py` (`AF3Trainer`: model + DDP + `ProtenixLoss` + `SymmetricPermutation` +
`LDDTMetrics`, EMA via `runner/ema.py`), `runner/inference.py` (`InferenceRunner`;
`update_inference_configs` adapts chunk sizes per sample size),
`runner/batch_inference.py` (the click CLI; wraps `InferenceRunner` rather than
duplicating it), `runner/dumper.py` (`DataDumper` → `<dump_dir>/<dataset>/<sample>/seed_<seed>/`
CIF + confidence JSON, samples sorted by ranking score), and
`runner/{msa,rna_msa,template}_search.py` (also used by `protenix/web_service/`).

### Smaller packages

`protenix/utils/permutation/` — AF3 chain/atom symmetry resolution (needed before loss);
`protenix/utils/cropping.py` — contiguous/spatial/interface crops;
`protenix/metrics/` — lDDT, clash, RMSD on predicted structures;
`protenix/tfg/` — Training-Free Guidance potentials steering the diffusion sampler
(`sample_diffusion.guidance.enable`); `protenix/web_service/` — Colab/notebook front-end
and visualization.

## Docs worth reading before changing behaviour

`docs/training_inference_instructions.md` (install, dataset download via
`scripts/database/download_protenix_data.sh`, cost tables), `docs/infer_json_format.md`
(input JSON schema incl. constraints), `docs/kernels.md`, `docs/supported_models.md`,
`docs/msa_template_pipeline.md`, `docs/prepare_training_data.md`.
