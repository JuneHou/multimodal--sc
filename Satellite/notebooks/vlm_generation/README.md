# vlm_generation — VLM direct generation of the untreated month

Direction opened 2026-09-08 (advisor): show a vision-language model the pictures a synthetic control
would use and ask it to draw the target site's next month. Plan `quirky-growing-wombat`.
**Generation only; evaluation is a separate plan.**

## What the model gets and returns
* Pictures only. The prompt says which parcel is the TARGET and which are SIMILAR. No weights, no block labels, no hurricane.
* One picture per sensor per month = the collaborator's own preview of the composite
  (`data/scripts/07` `create_rgb_preview`: Sentinel-2 natural colour, bands 3,2,1, one 2–98 percentile stretch;
  `data/scripts/08` `create_sentinel1_preview`: Sentinel-1 VV, VH, VV−VH as red, green, blue, each 2–98 stretched).
  `vg_render.py` copies both verbatim; `vg_tests.py` proves equality (max |diff| = 0 on 20 chips per sensor).
* Two donor structures, two outputs generated independently, two text arms:

| arm | sites in the prompt | months shown for output P10 | for output P11 |
|---|---|---|---|
| shc | TARGET only | P01–P09 (9) | P01–P10 (10) |
| scm | TARGET + SIMILAR 1–5 (control_rank 1–5, the SCM donor set used throughout the repo) | P08–P09 | P09–P10 |

  Every site stops at the month before the output month (the similar sites never show the output month).
  Band text arm: `band1` appends one line per picture with the chip's mean per band (`panel_lib.feats_from_raw`, native units); `band0` has pictures only.
* Output: one picture of the TARGET parcel at the output month, saved as returned.

## Pipeline
| step | where | file | writes |
|---|---|---|---|
| 1 | here (`satellite` env) | `vg_prepare.py` | `previews/` (702 PNG, 101×101), `sheets/` (216 contact sheets) + `sheets/persite/` (one sheet per site), `vg_prompts.jsonl` (full prompt text, both layouts), `vg_prompts_index.csv`, `vg_prompts_skipped.csv` |
| 2 | here | `vg_tests.py` | checks: previews == collaborator's arrays; months, donor ranks, image counts, sheet geometry; prints one prompt per arm |
| 3 | tinkercliffs login node, once | `setup_tinkercliffs.sh` | env `/projects/slmreasoning/junh/envs/vlmgen`, BAGEL repo clone, weights (HF cache / `hf_models/BAGEL-7B-MoT`) |
| 4 | tinkercliffs (SLURM, 1× A100) | `vg_generate.sbatch` → `vg_generate.py` | `outputs/<model>_<layout>/<prompt_id>_s<seed>.png`, `vg_outputs_<model>_<layout>.jsonl` (manifest: settings, size, seconds, errors) |
| 5 | here | `vg_collect.py` | `vg_generated_index.csv` (manifest joined to prompt fields, PNG readability) |
| 6 | here | `build_nb31.py` → `31_vlm_generation_gallery.ipynb` | per site: SCM sheet, observed P10/P11, every generated picture. Looking only. |

Server commands:
```
rsync -av --exclude outputs --exclude __pycache__ /data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/vlm_generation/ junh@tinkercliffs1.arc.vt.edu:/projects/slmreasoning/junh/vlm_generation/
ssh junh@tinkercliffs1.arc.vt.edu 'cd /projects/slmreasoning/junh/vlm_generation && bash setup_tinkercliffs.sh'   # once
ssh junh@tinkercliffs1.arc.vt.edu 'cd /projects/slmreasoning/junh/vlm_generation && MODEL=qwen  LAYOUT=persite sbatch vg_generate.sbatch --limit 4'   # smoke test
ssh junh@tinkercliffs1.arc.vt.edu 'cd /projects/slmreasoning/junh/vlm_generation && MODEL=qwen  LAYOUT=persite sbatch vg_generate.sbatch'
ssh junh@tinkercliffs1.arc.vt.edu 'cd /projects/slmreasoning/junh/vlm_generation && MODEL=bagel LAYOUT=sheet ENV=bagel sbatch vg_generate.sbatch'
rsync -av junh@tinkercliffs1.arc.vt.edu:/projects/slmreasoning/junh/vlm_generation/outputs/ /data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/vlm_generation/outputs/
rsync -av 'junh@tinkercliffs1.arc.vt.edu:/projects/slmreasoning/junh/vlm_generation/vg_outputs_*.jsonl' /data/wang/junh/githubs/latent-synthetic-control/Satellite/notebooks/vlm_generation/
```

## Fixed knobs
* Grid: model {Qwen-Image-Edit-2511, BAGEL-7B-MoT} × sensor {S2 treated 1–15, S1 treated 15–27} × arm {shc, scm} × band {0, 1} × output {P10, P11} × seed 0 = 216 prompts per model. Built 2026-09-08: 216 prompts, 0 skipped.
* Layouts: `sheet` = one contact sheet per prompt (rows = sites, TARGET first; columns = months; tiles 2× nearest, labelled "TARGET P07 2024-05"); `persite` = one sheet per site, list order TARGET, SIMILAR 1…5. Qwen resizes each condition image to ≈384×384 px for its vision encoder (diffusers `CONDITION_IMAGE_SIZE`), so `persite` is the default for Qwen; BAGEL's multi-image input is undocumented, so `sheet` for BAGEL (its `interleave_inference` path is implemented for `persite` too).
* Qwen settings: `true_cfg_scale 4.0, 40 steps, guidance 1.0, 1024×1024, negative_prompt " "`, `torch.Generator` seeded. BAGEL settings: `cfg_text 4.0, cfg_img 2.0, 50 timesteps, timestep_shift 3.0, cfg_renorm text_channel, 1024×1024`, `torch.manual_seed`.
* Calendar: P01 = 2023-11 … P10 = 2024-08, P11 = 2024-09 (`nb25_monthly_long_period_definitions.csv`).

## Provenance / gates
* Data read-only from `data/monthly_long_datasets`, paths and usability from `panel_monthly.build_index` (`panel_monthly_index.csv`).
* Preview equality gate and prompt-consistency gate: `vg_tests.py` (all passed 2026-09-08).
* Nothing in `notebooks/ts_SCM_ASCM/` or `Docs/` is modified by this folder.
* Not here: any evaluation; control sites as targets; other models; multiple seeds.
