from __future__ import annotations

import pytest

from snow_grass.workflow.data_migration import convert_graph


def _legacy_graph() -> dict:
    return {
        "nodes": [
            {"id": "input", "type": "core.input", "title": "输入", "x": 1, "y": 2},
            {"id": "classify", "type": "feedback.classify", "title": "分类", "x": 3, "y": 4},
        ],
        "edges": [{"id": "edge", "from": "input", "to": "classify", "condition": None}],
    }


def test_known_graph_is_converted_with_ids_and_ports_preserved() -> None:
    converted, changed = convert_graph(_legacy_graph())
    assert changed is True
    assert [node["id"] for node in converted["nodes"]] == ["input", "classify"]
    assert converted["edges"] == [
        {
            "id": "edge",
            "source_node_id": "input",
            "source_port": "payload",
            "target_node_id": "classify",
            "target_port": "payload",
        }
    ]
    assert convert_graph(converted) == (converted, False)


def test_unknown_node_and_condition_are_explicit_blockers() -> None:
    unknown = _legacy_graph()
    unknown["nodes"][1]["type"] = "custom.unknown"
    with pytest.raises(ValueError, match="Unknown legacy node"):
        convert_graph(unknown)
    conditional = _legacy_graph()
    conditional["edges"][0]["condition"] = {"field": "x"}
    with pytest.raises(ValueError, match="conditional edge"):
        convert_graph(conditional)
