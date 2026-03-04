"""Prompt template rendering for star-chamber council modes."""

from __future__ import annotations

_CODE_REVIEW_TEMPLATE = """\
You are a senior software craftsman reviewing code for quality, idioms, and architectural soundness.

## Project Context
{context}

## Code to Review
{files_section}

## Review Focus
1. Craftsmanship: Is this idiomatic, clean, well-structured?
2. Architecture: Does this fit the project's patterns? Any design concerns?
3. Correctness: Any logical issues, edge cases, or bugs?
4. Invariants: Do classifications (terminal, final, immutable) match runtime reality? \
Are there states where cleanup or cancellation is assumed but not enforced? \
Does the code's model of the system match what actually happens?
5. Maintainability: Will this be easy to understand and modify later?

## Output Format
Provide your review as structured JSON:
```json
{{
  "quality_rating": "excellent | good | fair | needs-work",
  "issues": [
    {{
      "severity": "high | medium | low",
      "location": "file:line",
      "category": "craftsmanship | architecture | correctness | invariants | maintainability",
      "description": "What is wrong",
      "suggestion": "How to fix it"
    }}
  ],
  "praise": [
    "Things done well"
  ],
  "summary": "Overall assessment in 2-3 sentences"
}}
```"""

_DESIGN_TEMPLATE = """\
You are a senior software architect advising on design decisions.

## Project Context
{context}

## Design Question
{question}

## Advisory Focus
1. Trade-offs: What are the key trade-offs between the possible approaches?
2. Fit: How well does each approach fit the project's existing architecture and constraints?
3. Risk: What are the risks associated with each approach?
4. Recommendation: Which approach do you recommend and why?

## Output Format
Provide your advice as structured JSON:
```json
{{
  "recommendation": "Your recommended approach",
  "approaches": [
    {{
      "name": "Approach name",
      "pros": ["advantage"],
      "cons": ["disadvantage"],
      "risk_level": "low | medium | high",
      "fit_rating": "excellent | good | fair | poor"
    }}
  ],
  "summary": "Overall recommendation in 2-3 sentences"
}}
```"""

_SYNTHESIS_TEMPLATE = """
## Other council members' feedback (round {round_number}):
{synthesis}
Please provide your perspective on these points. Note where you agree, disagree, or have additional insights."""

_NO_CONTEXT = "(No project-specific context provided.)"


def _build_files_section(files: dict[str, str]) -> str:
    """Render the code-to-review section from a path-to-content mapping.

    Args:
        files: Mapping of file paths to their source content.

    Returns:
        Formatted markdown section with each file in a fenced code block.
    """
    parts: list[str] = []
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def render_code_review_prompt(files: dict[str, str], context: str = "") -> str:
    """Render the code-review prompt template.

    Args:
        files: Mapping of file paths to source content to review.
        context: Optional project-specific context string.

    Returns:
        The fully rendered prompt string ready for LLM submission.
    """
    return _CODE_REVIEW_TEMPLATE.format(
        context=context if context else _NO_CONTEXT,
        files_section=_build_files_section(files),
    )


def render_design_prompt(question: str, context: str = "") -> str:
    """Render the design-question prompt template.

    Args:
        question: The design question to present to the council.
        context: Optional project-specific context string.

    Returns:
        The fully rendered prompt string ready for LLM submission.
    """
    return _DESIGN_TEMPLATE.format(
        context=context if context else _NO_CONTEXT,
        question=question,
    )


def augment_with_synthesis(original_prompt: str, synthesis: str, round_number: int) -> str:
    """Append an anonymous Chatham House synthesis to the prompt for debate rounds.

    Args:
        original_prompt: The original prompt to augment.
        synthesis: The anonymised synthesis from prior round responses.
        round_number: The current debate round number.

    Returns:
        The original prompt with the synthesis section appended.
    """
    return original_prompt + _SYNTHESIS_TEMPLATE.format(
        round_number=round_number,
        synthesis=synthesis,
    )
