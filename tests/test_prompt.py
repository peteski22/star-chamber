from __future__ import annotations

from star_chamber.prompt import (
    render_code_review_prompt,
    render_design_prompt,
)


class TestRenderCodeReviewPrompt:
    def test_basic_render(self):
        result = render_code_review_prompt({"main.py": "print('hello')"})
        assert "senior software craftsman" in result
        assert "main.py" in result
        assert "print('hello')" in result
        assert "Output Format" in result

    def test_multiple_files(self):
        files = {
            "app.py": "import os",
            "utils.py": "def helper(): pass",
        }
        result = render_code_review_prompt(files)
        assert "app.py" in result
        assert "import os" in result
        assert "utils.py" in result
        assert "def helper(): pass" in result

    def test_context_injection(self):
        result = render_code_review_prompt(
            {"a.py": "x = 1"},
            context="This is a Flask web application.",
        )
        assert "This is a Flask web application." in result

    def test_no_context_default(self):
        result = render_code_review_prompt({"a.py": "x = 1"})
        assert "(No project-specific context provided.)" in result

    def test_review_focus_areas_present(self):
        result = render_code_review_prompt({"a.py": "x = 1"})
        assert "Craftsmanship" in result
        assert "Architecture" in result
        assert "Correctness" in result
        assert "Invariants" in result
        assert "Maintainability" in result

    def test_json_output_format_fields_present(self):
        result = render_code_review_prompt({"a.py": "x = 1"})
        assert "quality_rating" in result
        assert "issues" in result
        assert "praise" in result
        assert "summary" in result


class TestRenderDesignPrompt:
    def test_basic_render(self):
        result = render_design_prompt("Should we use microservices?")
        assert "senior software architect" in result
        assert "Should we use microservices?" in result

    def test_context_injection(self):
        result = render_design_prompt(
            "Should we use microservices?",
            context="We are building an e-commerce platform.",
        )
        assert "We are building an e-commerce platform." in result

    def test_advisory_focus_areas_present(self):
        result = render_design_prompt("Should we use microservices?")
        assert "Trade-offs" in result
        assert "Fit" in result
        assert "Risk" in result
        assert "Recommendation" in result

    def test_no_context_default(self):
        result = render_design_prompt("Should we use microservices?")
        assert "(No project-specific context provided.)" in result

    def test_json_output_format_fields_present(self):
        result = render_design_prompt("Should we use microservices?")
        assert "recommendation" in result
        assert "approaches" in result
        assert "summary" in result
