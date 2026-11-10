import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tests/ (fixtures)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # semantic_uv の親

import numpy as np

from semantic_uv.core import pipeline as P
from semantic_uv.core.settings import Settings, UnwrapBackend


def settings(**kw) -> Settings:
    kw.setdefault("backend", UnwrapBackend.BUILTIN_LSCM)
    return Settings(**kw)


def run(snap, s, expect_valid=True):
    rep = P.validate_input(snap, s)
    rep.raise_if_failed("input")
    scope = P.build_scope(snap, s)
    dom = P.build_domain(snap, scope, s)
    plan = P.compute_plan(snap, scope, dom, s)
    vr = P.validate_plan(snap, plan, s)
    if expect_valid:
        assert vr.ok, [i.describe() for i in vr.issues]
    return plan, vr


def frozen_copy(obj):
    return {name: arr.copy() for name, arr in obj.uv.items()}, obj.seams.copy()


def assert_unchanged(obj, frozen):
    uvs, seams = frozen
    for name, arr in uvs.items():
        assert np.array_equal(obj.uv[name], arr), f"入力UV {name} が変化"
    assert np.array_equal(obj.seams, seams)
