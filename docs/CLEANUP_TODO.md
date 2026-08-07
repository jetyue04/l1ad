# Cleanup TODO

Items identified for removal. Review each one before deleting.

---

## Files to Delete

### Dead prototype scripts (break immediately if run)

| File | Reason |
|------|--------|
| `train_axo.py` | Imports `example.trainer`, `example.loader`, `example.architectures` — none of which exist in this repo. This was the original prototype from before the codebase was restructured. Superseded by `train_vicreg_wnae.py`. |
| `src/model/wnae/train_axo_standardize.py` | Same dead imports (`example.*`). Also has a bare `import_module("example.config")` without importing `import_module`. A training script buried inside the model directory — doesn't belong there either. |
| `train_wnae_from_vicreg.py` | Early manual prototype of `train_vicreg_wnae.py`. Writes its own training loop without using the trainer module, has no embedding standardization, and hardcodes the signal key. Not called by anything. Superseded by `train_vicreg_wnae.py` + `sweep_train_wnae.py`. |

### Stale job file

| File | Reason |
|------|--------|
| `nrp/wnae_job.yaml` | Runs `train_axo.py` (dead script above) and uses an old working directory `/axovol/wnae_10_14`. Superseded by `nrp/vicreg_wnae_job.yaml`. |

---

## Code to Clean Up (edit, not delete)

### `src/model/vae.py`
- Lines 49–100 are a commented-out alternative implementation of the same two classes.
- The live implementation (lines 1–48) is correct and used.
- Safe to delete everything from line 49 onwards.

### `src/stages/vicreg_stage.py`
- Completely empty (1-line file with no content).
- Either implement it as a proper pipeline stage or delete it.
- Currently it's just noise.

### `src/model/base_model.py`
- Defines a `BaseModel` class that is not imported by anything active.
- WNAE inherits from `nn.Module` directly, not `BaseModel`.
- Could be deleted, or kept if you plan to use it as scaffolding later.

---

## Notes

- `train_vae.ipynb` and `data.ipynb` — likely exploratory notebooks from early development. Confirm they are no longer needed before deleting.
- `config/config.yaml` — still used by `train_wnae_from_vicreg.py` (dead script). Once that script is deleted, this config is only used for the standalone WNAE path (not the VICReg→WNAE pipeline). Consider whether to keep or fold into `vicreg_wnae_config.yaml`.
