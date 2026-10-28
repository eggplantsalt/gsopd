import pytest

from research.gs_opd.smoke.object_pair_clearance import measure_forbidden_object_pairs


def test_object_pair_detects_violations_before_contact_and_keeps_pair_identity():
    pairs = [{"object_id_a": "carried", "object_id_b": "obstacle",
              "geom_ids_a": {1, 2}, "geom_ids_b": {3, 4}}]
    result = measure_forbidden_object_pairs(
        pairs, lambda a, b: {"status": "complete", "minimum_distance_m": 0.009,
                             "pair": [min(a), min(b)]},
    )
    assert result["status"] == "unsafe"
    assert result["pairs"][0]["measurement"]["pair"] == [1, 3]
    assert result["pairs"][0]["object_id_b"] == "obstacle"
    assert result["pairs"][0]["verdict"] == "violation"
    at_margin = measure_forbidden_object_pairs(
        pairs, lambda a, b: {"status": "complete", "minimum_distance_m": 0.010},
    )
    assert at_margin["status"] == "complete"
    assert at_margin["pairs"][0]["verdict"] == "pass"


def test_missing_geometry_evidence_cannot_pass_and_margin_cannot_be_reduced():
    pair = {"object_id_a": "a", "object_id_b": "b", "geom_ids_a": {1}, "geom_ids_b": {2}}
    result = measure_forbidden_object_pairs([pair], lambda a, b: {"status": "incomplete"})
    assert result["status"] == "incomplete"
    assert not result["measurement_complete"]
    with pytest.raises(ValueError, match="at_least_10mm"):
        measure_forbidden_object_pairs([pair], lambda a, b: {}, minimum_clearance_m=0.009)


def test_real_mujoco_pair_distance_detects_9mm_gap():
    import ast
    from pathlib import Path
    import numpy as np

    mujoco = pytest.importorskip("mujoco")
    source = Path(__file__).with_name("safety_bridge.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_minimum_geom_clearance")
    namespace = {"np": np, "mujoco": mujoco}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "safety_bridge.py", "exec"), namespace)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="a" type="sphere" size="0.01" pos="0 0 0"/>
      <geom name="b" type="sphere" size="0.01" pos="0.029 0 0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pair = {"object_id_a": "a", "object_id_b": "b", "geom_ids_a": {0}, "geom_ids_b": {1}}
    result = measure_forbidden_object_pairs(
        [pair], lambda a, b: namespace["_minimum_geom_clearance"](model, data, a, b),
    )
    assert result["status"] == "unsafe"
    assert result["pairs"][0]["measurement"]["minimum_distance_m"] == pytest.approx(0.009)
