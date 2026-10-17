import json
import os

import pytest
import yaml

from yolo_distill_quant.common import (
    InputError,
    load_base_config,
    load_candidates,
    slugify,
    temp_model_yaml,
)


def test_load_candidates_yaml(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("A: [0.25, 0.25, 512]\n")
    assert load_candidates(f) == {"A": [0.25, 0.25, 512]}


def test_load_candidates_json(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"A": [0.5, 0.25, 1024]}))
    assert load_candidates(f) == {"A": [0.5, 0.25, 1024]}


@pytest.mark.parametrize("body", ["A: [1, 2]\n", "A: [1, 2, x]\n", "A: [0, 1, 1]\n", "[]\n", "A: 5\n"])
def test_load_candidates_invalid(tmp_path, body):
    f = tmp_path / "c.yaml"
    f.write_text(body)
    with pytest.raises(InputError):
        load_candidates(f)


def test_missing_file():
    with pytest.raises(InputError):
        load_candidates("/nonexistent.yaml")


def test_base_config_requires_scales(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("nc: 80\n")
    with pytest.raises(InputError):
        load_base_config(f)


def test_temp_model_yaml_overrides_scales_and_cleans_up():
    base = {"nc": 80, "scales": {"n": [0.5, 0.25, 1024]}}
    with temp_model_yaml(base, [0.25, 0.25, 256], "Pico C") as path:
        cfg = yaml.safe_load(open(path))
        assert cfg["scales"] == {"custom": [0.25, 0.25, 256]}
        assert base["scales"] == {"n": [0.5, 0.25, 1024]}  # base untouched
    assert not os.path.exists(path)


def test_slugify():
    assert slugify("Pico-B (Depth + cap 512)") == "pico_b_depth_cap_512"


def test_str2bool():
    from yolo_distill_quant.common import str2bool

    assert str2bool("True") is True and str2bool("yes") is True and str2bool("1") is True
    assert str2bool("false") is False and str2bool("0") is False
    with pytest.raises(Exception):
        str2bool("maybe")
