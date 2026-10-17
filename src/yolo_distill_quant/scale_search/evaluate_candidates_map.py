"""CLI: warm-start each candidate from pretrained weights, fine-tune briefly, and report mAP."""
import argparse
import gc
import sys
from pathlib import Path

from ..common import (
    InputError,
    load_base_config,
    load_candidates,
    print_table,
    slugify,
    str2bool,
    temp_model_yaml,
    write_results,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="evaluate-candidates-map",
        description="For each scale candidate: build the model, copy shape-matching weights from a "
        "pretrained checkpoint, fine-tune, then validate and report mAP. Intended for similar "
        "architectures (only layers with identical tensor shapes are transferred).",
    )
    p.add_argument("--config", required=True, help="base Ultralytics model YAML containing a 'scales' block")
    p.add_argument("--candidates", required=True, help="YAML/JSON file mapping name -> [depth, width, max_channels]")
    p.add_argument("--weights", required=True, help="pretrained .pt checkpoint to transfer weights from")
    p.add_argument("--data", default="coco128.yaml", help="dataset YAML (default: coco128.yaml, auto-downloaded)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="batch size; lower it if you run out of memory")
    p.add_argument("--workers", type=int, default=2, help="dataloader workers; each one costs RAM")
    p.add_argument("--cache", default="False", help="'ram', 'disk' or 'False' (default: False)")
    p.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu (default: GPU 0 if available, else cpu)")
    p.add_argument("--project", default="runs/candidates", help="directory for training runs")
    p.add_argument("--gflops", type=str2bool, default=False, metavar="{true,false}",
                   help="true: also compute GFLOPs at --imgsz for each candidate and add it to the table and "
                   "exports; false (default): skip the calculation entirely")
    p.add_argument("--param-budget", type=float, default=None, help="parameter budget in millions; flags candidates over it")
    p.add_argument("--json-out", help="write results to this JSON file")
    p.add_argument("--csv-out", help="write results to this CSV file")
    return p.parse_args(argv)


def main(argv=None):
    
    """_summary_

    Raises:
        InputError: _description_
        
    Returns:
        _type_: _description_
    """
    
    
    args = parse_args(argv)
    try:
        base_cfg = load_base_config(args.config)
        candidates = load_candidates(args.candidates)
        if not Path(args.weights).is_file():
            raise InputError(f"file not found: {args.weights}")
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    import torch
    from ultralytics import YOLO

    if args.gflops:
        from ultralytics.utils.torch_utils import get_flops

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\nWeights: {args.weights}\n")

    results = []
    for name, scales in candidates.items():
        
        print("=" * 70)
        print(f"Processing: {name} with scales {scales}")
        print("=" * 70)
        model = None
        row = {"name": name, "scale": str(scales), "params_m": "ERR", "map50": "ERR", "map50_95": "ERR", "error": ""}
        if args.gflops:
            row["gflops"] = "ERR"
        try:
            with temp_model_yaml(base_cfg, scales, name) as path:
                model = YOLO(path)
                if args.gflops:
                    # architecture-only number, so compute it before training: it survives a later failure
                    flops = get_flops(model.model, args.imgsz)
                    row["gflops"] = round(flops, 2) if flops else "n/a"  # get_flops returns 0.0 without `thop`
                # pretrained=<.pt> makes the trainer copy every layer whose name and tensor shape match;
                # loading weights into `model` first would be discarded because it is built from a YAML
                model.train(
                    pretrained=args.weights,
                    data=args.data,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    cache=False if args.cache.lower() == "false" else args.cache,
                    device=device,
                    project=str(Path(args.project).resolve()),
                    name=slugify(name),
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                )
                metrics = model.val(data=args.data, imgsz=args.imgsz, device=device, verbose=False)
                row["map50"] = round(float(metrics.box.map50), 4)
                row["map50_95"] = round(float(metrics.box.map), 4)
                row["params_m"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
        except Exception as e:
            print(f"[ERROR] Failed processing {name}: {e}", file=sys.stderr)
            row["error"] = str(e)
        finally:
            # free RAM/VRAM before the next candidate so memory does not pile up
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(row)

    print()
    columns = [("Candidate", "name", 26, None), ("Params (M)", "params_m", 10, None)]
    if args.gflops:
        columns.append(("GFLOPs", "gflops", 8, None))
    columns += [("mAP50", "map50", 8, None), ("mAP50-95", "map50_95", 8, None)]
    print_table(results, columns)
    if args.param_budget is not None:
        over = [r["name"] for r in results if isinstance(r["params_m"], float) and r["params_m"] > args.param_budget]
        print(f"Parameter budget: <= {args.param_budget}M. Over budget: {', '.join(over) if over else 'none'}")
    print("Higher mAP is better; compare against the parameter budget.\n")

    write_results(results, args.json_out, args.csv_out)
    return 1 if any(r["error"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
