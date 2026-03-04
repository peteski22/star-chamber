"""Tests for spec file access via importlib.resources."""

from __future__ import annotations

import json

import pytest

from star_chamber.schema import get_schema, list_schemas


class TestListSchemas:
    def test_returns_known_schemas(self):
        names = list_schemas()
        assert "code-review-result" in names
        assert "council-config" in names
        assert "council-output" in names
        assert "design-advice-result" in names
        assert "provider-config" in names
        assert "provider-response" in names
        assert "review-request" in names

    def test_returns_sorted(self):
        names = list_schemas()
        assert names == sorted(names)

    def test_returns_strings(self):
        names = list_schemas()
        assert all(isinstance(n, str) for n in names)

    def test_strips_schema_suffix(self):
        """Schema names should not include .schema.json suffix."""
        names = list_schemas()
        assert all(not n.endswith(".json") for n in names)


class TestGetSchema:
    def test_returns_valid_json_string(self):
        content = get_schema("code-review-result")
        data = json.loads(content)
        assert "$schema" in data or "type" in data

    def test_all_schemas_loadable(self):
        for name in list_schemas():
            content = get_schema(name)
            data = json.loads(content)
            assert isinstance(data, dict)

    def test_unknown_schema_raises(self):
        with pytest.raises(FileNotFoundError):
            get_schema("nonexistent-schema")

    def test_returns_string_not_bytes(self):
        content = get_schema("code-review-result")
        assert isinstance(content, str)
