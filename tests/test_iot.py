"""Tests for the IoT MapReduce (ROADMAP [3.4])."""

from plantguard.services.iot import map_sensor_data, reduce_sensor_data


def test_map_parses_json_string_value():
    records = [{"value": '{"temperature": 20, "humidity": 50, "soil": 30}'}]
    mapped = map_sensor_data(records)
    assert ("temperature", 20.0) in mapped
    assert ("humidity", 50.0) in mapped
    assert ("soil", 30.0) in mapped
    assert len(mapped) == 3


def test_map_handles_dict_value():
    assert map_sensor_data([{"value": {"temperature": 25}}]) == [("temperature", 25.0)]


def test_map_skips_malformed_records():
    assert map_sensor_data([{"value": "not json"}, {}, {"value": None}]) == []


def test_reduce_aggregates():
    reduced = reduce_sensor_data([("soil", 10), ("soil", 20), ("soil", 30)])
    assert reduced["soil"] == {"count": 3, "min": 10.0, "max": 30.0, "avg": 20.0}


def test_reduce_empty():
    assert reduce_sensor_data([]) == {}
