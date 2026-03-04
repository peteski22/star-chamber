"""Click-based CLI for the star-chamber multi-LLM council SDK."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import click

from star_chamber.config import ConfigError, load_config
from star_chamber.types import CodeReviewResult, CouncilConfig, DesignQuestionResult


def _load_config(config_path: str | None) -> CouncilConfig:
    """Load council configuration with error handling.

    Args:
        config_path: Optional path to a providers.json file.  When None the
            default config resolution is used.

    Returns:
        A validated CouncilConfig.

    Raises:
        ConfigError: If the config cannot be loaded.
    """
    path = Path(config_path) if config_path else None
    return load_config(path)


def _print_code_review_result(result: CodeReviewResult) -> None:
    """Print a human-readable summary of a code review council result.

    Args:
        result: The aggregated code review result.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo("Code Review Council Results")
    click.echo(f"{'=' * 60}")
    click.echo(f"Providers used: {', '.join(result.providers_used)}")

    if result.failed_providers:
        click.echo(f"\nFailed providers ({len(result.failed_providers)}):")
        for fp in result.failed_providers:
            click.echo(f"  - {fp.provider}: {fp.error}")

    if result.quality_ratings:
        click.echo("\nQuality ratings:")
        for provider, rating in result.quality_ratings.items():
            click.echo(f"  - {provider}: {rating}")

    if result.consensus_issues:
        click.echo(f"\nConsensus issues ({len(result.consensus_issues)}):")
        for issue in result.consensus_issues:
            click.echo(f"  [{issue.severity}] {issue.location} ({issue.category})")
            click.echo(f"    {issue.description}")
            click.echo(f"    Suggestion: {issue.suggestion}")

    if result.majority_issues:
        click.echo(f"\nMajority issues ({len(result.majority_issues)}):")
        for issue in result.majority_issues:
            flagged = ", ".join(issue.flagged_by)
            click.echo(f"  [{issue.severity}] {issue.location} ({issue.category}) — flagged by {flagged}")
            click.echo(f"    {issue.description}")
            click.echo(f"    Suggestion: {issue.suggestion}")

    if result.summary:
        click.echo(f"\nSummary: {result.summary}")

    click.echo("")


def _print_design_result(result: DesignQuestionResult) -> None:
    """Print a human-readable summary of a design question council result.

    Args:
        result: The aggregated design result.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo("Design Question Council Results")
    click.echo(f"{'=' * 60}")
    click.echo(f"Question: {result.prompt}")
    click.echo(f"Providers used: {', '.join(result.providers_used)}")

    if result.failed_providers:
        click.echo(f"\nFailed providers ({len(result.failed_providers)}):")
        for fp in result.failed_providers:
            click.echo(f"  - {fp.provider}: {fp.error}")

    if result.consensus_recommendation:
        click.echo(f"\nConsensus recommendation: {result.consensus_recommendation}")

    if result.approaches:
        click.echo(f"\nApproaches ({len(result.approaches)}):")
        for approach in result.approaches:
            click.echo(f"\n  {approach.name} (risk: {approach.risk_level})")
            if approach.fit_rating:
                click.echo(f"    Fit rating: {approach.fit_rating}")
            click.echo(f"    Recommended by: {approach.recommended_by} provider(s)")
            if approach.pros:
                click.echo("    Pros:")
                for pro in approach.pros:
                    click.echo(f"      + {pro}")
            if approach.cons:
                click.echo("    Cons:")
                for con in approach.cons:
                    click.echo(f"      - {con}")

    if result.summary:
        click.echo(f"\nSummary: {result.summary}")

    click.echo("")


@click.group()
def main() -> None:
    """Star-chamber: multi-LLM council review tool."""


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-p", "--provider", "providers", multiple=True, help="Provider name to include (repeatable).")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Path to providers.json.")
@click.option("--timeout", type=int, default=None, help="Per-provider timeout in seconds.")
@click.option(
    "--context-file",
    type=click.Path(exists=True),
    default=None,
    help="File containing project context to include in the prompt.",
)
@click.option("--output", type=click.Path(), default=None, help="Write JSON result to file.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="Output format.")
def review(
    files: tuple[str, ...],
    providers: tuple[str, ...],
    config_path: str | None,
    timeout: int | None,
    context_file: str | None,
    output: str | None,
    fmt: str,
) -> None:
    """Review source files with a multi-LLM council."""
    from star_chamber.council import run_council_sync

    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)

    # Filter providers if requested.
    if providers:
        filtered = tuple(p for p in config.providers if p.provider in providers)
        if not filtered:
            click.echo(f"Error: no configured providers match: {', '.join(providers)}", err=True)
            sys.exit(2)
        config = CouncilConfig(
            providers=filtered,
            timeout_seconds=config.timeout_seconds,
            consensus_threshold=config.consensus_threshold,
            platform=config.platform,
        )

    # Override timeout if specified.
    if timeout is not None:
        config = CouncilConfig(
            providers=config.providers,
            timeout_seconds=timeout,
            consensus_threshold=config.consensus_threshold,
            platform=config.platform,
        )

    # Read context file if provided.
    context = ""
    if context_file:
        context = Path(context_file).read_text(encoding="utf-8")

    # Read file contents.
    file_contents: dict[str, str] = {}
    for file_path in files:
        path = Path(file_path)
        file_contents[str(path)] = path.read_text(encoding="utf-8")

    result = run_council_sync(
        files=file_contents,
        config=config,
        mode="code-review",
        context=context,
    )

    # Output handling.
    if output:
        Path(output).write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
        click.echo(f"Results written to {output}")
    elif fmt == "json":
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        _print_code_review_result(result)  # type: ignore[arg-type]


@main.command()
@click.argument("question")
@click.option("-p", "--provider", "providers", multiple=True, help="Provider name to include (repeatable).")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Path to providers.json.")
@click.option("--timeout", type=int, default=None, help="Per-provider timeout in seconds.")
@click.option(
    "--context-file",
    type=click.Path(exists=True),
    default=None,
    help="File containing project context to include in the prompt.",
)
@click.option("--output", type=click.Path(), default=None, help="Write JSON result to file.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="Output format.")
def ask(
    question: str,
    providers: tuple[str, ...],
    config_path: str | None,
    timeout: int | None,
    context_file: str | None,
    output: str | None,
    fmt: str,
) -> None:
    """Ask a design question to a multi-LLM council."""
    from star_chamber.council import run_council_sync

    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)

    # Filter providers if requested.
    if providers:
        filtered = tuple(p for p in config.providers if p.provider in providers)
        if not filtered:
            click.echo(f"Error: no configured providers match: {', '.join(providers)}", err=True)
            sys.exit(2)
        config = CouncilConfig(
            providers=filtered,
            timeout_seconds=config.timeout_seconds,
            consensus_threshold=config.consensus_threshold,
            platform=config.platform,
        )

    # Override timeout if specified.
    if timeout is not None:
        config = CouncilConfig(
            providers=config.providers,
            timeout_seconds=timeout,
            consensus_threshold=config.consensus_threshold,
            platform=config.platform,
        )

    # Read context file if provided.
    context = ""
    if context_file:
        context = Path(context_file).read_text(encoding="utf-8")

    result = run_council_sync(
        prompt=question,
        config=config,
        mode="design-question",
        context=context,
    )

    # Output handling.
    if output:
        Path(output).write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
        click.echo(f"Results written to {output}")
    elif fmt == "json":
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        _print_design_result(result)  # type: ignore[arg-type]


@main.command("list-providers")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Path to providers.json.")
def list_providers(config_path: str | None) -> None:
    """List configured LLM providers."""
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)

    click.echo(f"\nConfigured providers ({len(config.providers)}):\n")
    for provider in config.providers:
        if provider.local:
            status = "local"
        elif config.platform:
            status = f"platform ({config.platform})"
        else:
            status = "direct"
        click.echo(f"  {provider.provider} — {provider.model} [{status}]")
    click.echo("")


@main.command()
@click.argument("name")
def schema(name: str) -> None:
    """Show a protocol schema. Use 'list' to see available schemas."""
    from star_chamber.schema import get_schema, list_schemas

    if name == "list":
        for schema_name in list_schemas():
            click.echo(schema_name)
        return

    try:
        content = get_schema(name)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)
    click.echo(content)
