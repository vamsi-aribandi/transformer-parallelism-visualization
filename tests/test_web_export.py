"""Exporter integrity: recording, schema, and consistency with the pinned
semantic layer. Uses TP only (the currently shipped timelines) to stay fast."""

import pytest

from tpviz import configs
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.model import count_collectives
from tpviz.web import serializers
from tpviz.web.recorder import record
from tpviz.web.schema import DocumentBuilder, validate


@pytest.fixture(scope="module")
def doc():
    from tpviz.scenes.tp import TPScene
    from tpviz.scenes.tp_train import TPTrainScene

    builder = DocumentBuilder()
    for name, cls, steps in (
        ("tp_fwd", TPScene, None),
        ("tp_train", TPTrainScene, train_steps(configs.TP)),
    ):
        serializers.UNSERIALIZED.clear()
        rec, _ = record(cls)
        assert not serializers.UNSERIALIZED, serializers.UNSERIALIZED
        from tpviz.core.model import forward_steps

        builder.add_timeline(name, rec, steps or forward_steps(configs.TP))
    return builder.build()


def test_document_validates(doc):
    assert validate(doc) == []


def test_steps_monotone_and_within_duration(doc):
    for name, tl in doc["timelines"].items():
        last = 0
        for st in tl["steps"]:
            assert st["t0"] >= last, f"{name}: step order broken"
            assert st["t1"] <= tl["dur"]
            last = st["t0"]


def test_summary_matches_semantic_layer(doc):
    tl = doc["timelines"]["tp_train"]
    steps = train_steps(configs.TP)
    assert tl["summary"]["coll"] == count_collectives(steps)
    assert tl["summary"]["mm"] == count_matmuls(steps)
    # program shows the mirrored AG/RS rhythm with duality notes
    bwd_colls = [s for s in tl["steps"] if s["bwd"] and s["kind"] in ("AllGather", "ReduceScatter")]
    assert len(bwd_colls) == 8
    assert all("note" in s for s in bwd_colls)


def test_tracks_lie_within_object_lifetimes(doc):
    for name, tl in doc["timelines"].items():
        for row in tl["tracks"]:
            o = tl["objects"][row[0]]
            assert row[2] >= o["t0"] - 1 and row[3] <= o["t1"] + 1, f"{name}: stray track"


def test_every_deck_has_tooltip(doc):
    for tl in doc["timelines"].values():
        for o in tl["objects"]:
            if o["c"] in ("deck", "wrect"):
                assert "tip" in o and "tex" in o
