"""Exporter integrity for the discrete step-state schema (v2). TP only — the
currently shipped figure — to stay fast."""

import pytest

from tpviz import configs
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.model import count_collectives
from tpviz.web import serializers
from tpviz.web.recorder import record
from tpviz.web.schema import DocumentBuilder, validate


@pytest.fixture(scope="module")
def doc():
    from tpviz.scenes.tp_train import TPTrainScene

    builder = DocumentBuilder()
    serializers.UNSERIALIZED.clear()
    rec, _ = record(TPTrainScene)
    assert not serializers.UNSERIALIZED, serializers.UNSERIALIZED
    builder.add_timeline("tp_train", rec, train_steps(configs.TP))
    return builder.build()


def test_document_validates(doc):
    assert validate(doc) == []


def test_base_state_and_steps(doc):
    tl = doc["timelines"]["tp_train"]
    assert len(tl["base"]) > 40  # lanes, headers, weights, decks all placed
    assert len(tl["steps"]) == len(train_steps(configs.TP))
    # every step has an exact teleport diff or lifecycle change or is a no-op pulse
    assert all(("d" in s and "beats" in s) for s in tl["steps"])


def test_forward_backward_split(doc):
    tl = doc["timelines"]["tp_train"]
    fwd = [s for s in tl["steps"] if not s["bwd"]]
    bwd = [s for s in tl["steps"] if s["bwd"]]
    assert fwd and bwd
    # forward steps all precede backward steps
    first_bwd = tl["steps"].index(bwd[0])
    assert all(s["bwd"] for s in tl["steps"][first_bwd:])


def test_summary_and_duality_notes(doc):
    tl = doc["timelines"]["tp_train"]
    steps = train_steps(configs.TP)
    assert tl["summary"]["coll"] == count_collectives(steps)
    assert tl["summary"]["mm"] == count_matmuls(steps)
    bwd_colls = [
        s for s in tl["steps"]
        if s["bwd"] and s["kind"] in ("AllGather", "ReduceScatter")
    ]
    assert len(bwd_colls) == 8
    assert all("noteh" in s for s in bwd_colls)
    assert tl["steps"][-1]["coll"] == sum(tl["summary"]["coll"].values())
    assert tl["steps"][-1]["mm"] == sum(tl["summary"]["mm"].values())


def test_every_tensor_has_tooltip(doc):
    tl = doc["timelines"]["tp_train"]
    for o in tl["objects"]:
        if o["c"] in ("deck", "wrect"):
            assert "tip" in o and "tex" in o


def test_no_strip_objects(doc):
    tl = doc["timelines"]["tp_train"]
    assert all(o["z"] < 10 for o in tl["objects"])


def test_merge_documents_rekeys_glyphs_and_tooltips():
    """Re-recording one timeline merges into a previous full export: glyph ids
    are positional and tooltips index-based, so both must be remapped."""
    from tpviz.web.export import merge_documents

    base = {
        "glyphs": {"G0": "M0", "G1": "M1"}, "eq": {"e1": {"w": 1, "h": 1, "u": [["G1", 0, 0]], "r": []}},
        "tooltips": [{"kind": "act", "dims": ["B"]}],
        "timelines": {"tp_train": {"objects": [{"c": "deck", "tip": 0}]}},
        "meta": {"strategies": {"tp": {}}, "modes": {"tp_train": "train"}}, "dims": {},
    }
    new = {
        "glyphs": {"G0": "M1", "G1": "M9"},  # G0 here is base's G1; G1 is new
        "eq": {"e2": {"w": 1, "h": 1, "u": [["G0", 0, 0], ["G1", 1, 1]], "r": []}},
        "tooltips": [{"kind": "wt", "dims": ["D"]}, {"kind": "act", "dims": ["B"]}],
        "timelines": {"5d_train": {"objects": [{"c": "wrect", "tip": 0}, {"c": "deck", "tip": 1}]}},
        "meta": {"strategies": {"5d": {}}, "modes": {"5d_train": "train"}}, "dims": {"B": 8},
    }
    out = merge_documents(base, new)
    assert set(out["timelines"]) == {"tp_train", "5d_train"}
    assert out["glyphs"] == {"G0": "M0", "G1": "M1", "G2": "M9"}
    assert out["eq"]["e2"]["u"] == [["G1", 0, 0], ["G2", 1, 1]]
    assert out["tooltips"] == [{"kind": "act", "dims": ["B"]}, {"kind": "wt", "dims": ["D"]}]
    objs = out["timelines"]["5d_train"]["objects"]
    assert [o["tip"] for o in objs] == [1, 0]  # new tooltip appended, shared one reused
    assert out["meta"]["modes"] == {"tp_train": "train", "5d_train": "train"}
