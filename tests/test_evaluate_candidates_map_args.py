from yolo_distill_quant.scale_search.evaluate_candidates_map import main, parse_args


def test_memory_safe_defaults():
    a = parse_args(["--config", "m.yaml", "--candidates", "c.yaml", "--weights", "w.pt"])
    assert a.batch == 8 and a.workers == 2 and a.cache == "False"


def test_missing_weights_file_exits_2(tmp_path, capsys):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("scales: {n: [0.5, 0.25, 1024]}\n")
    cand = tmp_path / "c.yaml"
    cand.write_text("A: [0.25, 0.25, 512]\n")
    code = main(["--config", str(cfg), "--candidates", str(cand), "--weights", str(tmp_path / "nope.pt")])
    assert code == 2
    assert "file not found" in capsys.readouterr().err


def test_gflops_flag():
    base = ["--config", "m.yaml", "--candidates", "c.yaml", "--weights", "w.pt"]
    assert parse_args(base).gflops is False
    assert parse_args(base + ["--gflops", "true"]).gflops is True
    assert parse_args(base + ["--gflops", "false"]).gflops is False
