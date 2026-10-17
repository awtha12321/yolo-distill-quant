# yolo-distill-quant

A teacher-student knowledge distillation and quantization framework for building ultra-compact
Ultralytics YOLO models that run on edge devices. This is a research project, and every module is
built as a reusable command-line tool for other people's models, datasets and configs.

## Pipeline

```
 1. Pick a student size   ->  2. Distill from a teacher  ->  3. Quantize  ->  4. Deploy on edge
    scale_search                (planned)                    (planned)
```

| Stage | Module | Status | What it does |
|-------|--------|--------|--------------|
| Student sizing | `scale_search` | Available | Compare `[depth, width, max_channels]` candidates by GFLOPs, parameters and post-fine-tune mAP |
| Distillation | _TBD_ | Planned | Train the chosen student against a larger teacher |
| Quantization | _TBD_ | Planned | Quantize the student for the target edge device |

## Install

```bash
pip install -e .            # add ".[test]" to run the tests
```

## Design principles

- **Model-agnostic.** Tools target Ultralytics YOLO v12 and newer. They take any model YAML or
  checkpoint from the command line and never hardcode a model, dataset or path.
- **One tool, one job.** Each command is named for what it does and takes all its settings as arguments.
- **Shared code lives in `common.py`.** No copy-pasted helpers between modules.
- **Fails clearly.** Bad input prints a message and exits with code 2; a failed run exits with code 1.

## Module: `scale_search`

Command:

- `evaluate-candidates-map`: builds each candidate, copies shape-matching weights from a pretrained
  checkpoint, fine-tunes briefly, and reports parameters, mAP50 and mAP50-95. Pass `--gflops true` to
  also compute GFLOPs (at `--imgsz`) and add it to the table and exports; the default is `false`, which
  skips the calculation.

Candidates are a YAML or JSON file mapping a name to `[depth, width, max_channels]`
(see `examples/candidates.yaml`):

```yaml
"Pico-A (Depth cut)": [0.25, 0.25, 1024]
"Pico-B (Depth + cap 512)": [0.25, 0.25, 512]
```

```bash
evaluate-candidates-map \
  --config configs/yolov12.yaml \
  --candidates examples/candidates.yaml \
  --weights weights/yolo12n.pt \
  --data coco128.yaml --epochs 15 --imgsz 640 --batch 8 --workers 2 --gflops true --param-budget 1.6
```

Run the command with `--help` for every option.

Notes:
- The base config must be an Ultralytics model YAML with a `scales` block.
- Weights are transferred by Ultralytics (`pretrained=<.pt>`): only layers whose name and tensor shape match are copied. It is meant for fine-tuning
  from a similar architecture; the other layers start random.

## Repository layout

```
src/yolo_distill_quant/            the package; shared code in common.py
  scale_search/                    stage 1: student-size search (one subpackage per stage)
configs/                 example base model YAMLs
examples/                example inputs (candidates file)
tests/                   unit tests (no Ultralytics or GPU needed)
CLAUDE.md                context for AI coding assistants working on this repo
```

## Contributing a new module

See `CLAUDE.md` for the conventions (naming, argument handling, error codes, tests).

## License

Copyright (C) 2026 awtha12321. Licensed under AGPL-3.0 (see `LICENSE`), because it depends on Ultralytics.
