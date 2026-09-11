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
