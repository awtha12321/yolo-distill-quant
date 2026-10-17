# CLAUDE.md

## What this project is
`yolo-distill-quant` is a research project on teacher-student knowledge distillation and
quantization to get ultra-compact YOLO models onto edge devices. It is also a **public toolkit**:
every module is a reusable CLI for other people's YOLO models, not a one-off script for the author's
setup. Read `README.md` for the pipeline and module status.

Target models: Ultralytics YOLO v12 and newer (anything with an Ultralytics model YAML).

## Core rule: build for other users
- Never hardcode a model, config, weights path, dataset, device, or user-specific path
  (no `/home/<user>/...`, no `PROJECT_ROOT = parents[N]`). Everything comes from CLI args.
- Don't branch on the YOLO version. Rely on Ultralytics' YAML and checkpoint formats so newer
  versions work without code changes.
- Defaults are fine for convenience (e.g. `--epochs`), but required inputs like `--config` and
  `--weights` stay required.

## Layout and conventions
- `src/yolo_distill_quant/<stage>/`: one subpackage per pipeline stage. Currently `scale_search`.
  Add new stages (distillation, quantization) as sibling subpackages and register commands in
  `pyproject.toml` under `[project.scripts]`.
- Shared helpers go in `src/yolo_distill_quant/common.py` (candidate loading, temp YAMLs,
  table printing, result export). Reuse them; do not duplicate logic across modules.
- Name modules and commands for what they do: `evaluate_candidates_map.py` ->
  `evaluate-candidates-map`. No vague names like `utils2` or `run`.
- Optional extra metrics are opt-in true/false flags (`--gflops true`, parsed with `common.str2bool`):
  when off, the value is not computed or shown.
- Each command module exposes `parse_args(argv=None)` and `main(argv=None) -> int`.
- Exit codes: `0` success, `1` some run failed, `2` bad user input (raise `InputError`, print
  `error: ...` to stderr).
- Temp files go in the system temp dir via `temp_model_yaml` and are always cleaned up.
- Optional result export through `--json-out` / `--csv-out`.
- Import `torch` and `ultralytics` inside `main()` (not at module top) so `--help` and the unit tests
  stay fast and work without a GPU.

## Commands
```bash
pip install -e ".[test]"
pytest                       # unit tests; need only pyyaml and pytest
evaluate-candidates-map --help
```

## Testing
- Put pure-logic tests (parsing, validation, file handling) in `tests/` so they run without
  Ultralytics. End-to-end checks need Ultralytics installed and, for mAP scoring, a pretrained
  `.pt` file.
- Reference check: the YOLOv12n scale `[0.50, 0.25, 1024]` gives about 2.6M parameters. GFLOPs at
  640px measured 7.63 with Ultralytics 8.4.175 (`get_flops`), while the comment in `configs/yolov12.yaml`
  says 6.7; the difference is not explained, so compare GFLOPs only across runs on the same Ultralytics version.

## Things to keep in mind
- The project is licensed AGPL-3.0 because it depends on Ultralytics.
- `weights/`, `runs/` and `*.pt` are git-ignored; never commit them.
- Weights transfer through `model.train(pretrained=<.pt>)`, which copies only layers with matching name and shape, so it is for similar architectures. Do not load weights into a YAML-built `YOLO` model before `train()`: the trainer rebuilds the model and discards them.
- Training defaults are memory-conservative (`--batch 8`, `--workers 2`); the author runs WSL2 with 7.5 GB RAM and a 4 GB GPU.
- Keep the README module table in sync when you add or change a module.
