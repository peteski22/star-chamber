# star-chamber

Multi-LLM council protocol SDK. Fan out code reviews and design questions to multiple LLM providers, then classify findings by consensus.

## Installation

```bash
pip install star-chamber
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add star-chamber
```

For platform-managed API key resolution, install the optional `platform` extra:

```bash
pip install star-chamber[platform]
```

## Configuration

Create `~/.config/star-chamber/providers.json`:

```json
{
  "providers": [
    {"provider": "openai", "model": "gpt-4o", "api_key": "${OPENAI_API_KEY}"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "${ANTHROPIC_API_KEY}"}
  ],
  "timeout_seconds": 90,
  "consensus_threshold": 2
}
```

API keys can be literal values or `${ENV_VAR}` references that are resolved at runtime.

### Platform mode (any-llm)

Instead of managing API keys per provider, you can use [Mozilla AI's any-llm platform](https://github.com/mozilla-ai/any-llm) for centralised key management. Set `ANY_LLM_KEY` in your environment and add `"platform": "any-llm"` to your config:

```json
{
  "providers": [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
  ],
  "platform": "any-llm",
  "timeout_seconds": 90,
  "consensus_threshold": 2
}
```

When `platform` is set, API keys are fetched from the platform service — no `api_key` fields needed. Install the platform extra: `pip install star-chamber[platform]`.

Override the config path with the `STAR_CHAMBER_CONFIG` environment variable.

## CLI

### Code review

```bash
star-chamber review src/auth.py src/db.py
```

### Design question

```bash
star-chamber ask "Should we use Redis or Memcached for session storage?"
```

### Options

```
--provider, -p    Provider to include (repeatable)
--config          Path to providers.json
--timeout         Per-provider timeout in seconds
--context-file    File containing project context to include in the prompt
--council-context File containing prior council round feedback (debate mode)
--format          Output format: text or json
--output          Write JSON result to file
```

### List providers

```bash
star-chamber list-providers
```

### Protocol schemas

The SDK ships the council protocol specification as package data.

```bash
# List available schemas.
star-chamber schema list

# Print a specific schema.
star-chamber schema code-review-result
```

## Python API

```python
from star_chamber import run_council_sync, CouncilConfig, ProviderConfig

config = CouncilConfig(
    providers=(
        ProviderConfig(provider="openai", model="gpt-4o"),
        ProviderConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    ),
    timeout_seconds=90,
    consensus_threshold=2,
)

# Code review.
result = run_council_sync(
    files={"auth.py": open("auth.py").read()},
    config=config,
    mode="code-review",
)

print(result.summary)
for issue in result.consensus_issues:
    print(f"  [{issue.severity}] {issue.location}: {issue.description}")

# Design question.
result = run_council_sync(
    prompt="Should we use a monorepo or polyrepo?",
    config=config,
    mode="design-question",
)

print(result.consensus_recommendation)
```

### Async

```python
import asyncio
from star_chamber import run_council

result = asyncio.run(run_council(
    files={"auth.py": open("auth.py").read()},
    mode="code-review",
))
```

### Schema access

```python
from star_chamber import get_schema, list_schemas

# List available schema names.
names = list_schemas()

# Get a specific schema as a JSON string.
schema_json = get_schema("code-review-result")
```

## Consensus classification

Issues from multiple providers are grouped by file, line proximity (within 5 lines), and category, then classified as:

- **Consensus** -- all providers agree.
- **Majority** -- two or more providers agree, but not all.
- **Individual** -- flagged by a single provider.

Results are sorted by severity within each bucket.

## License

Apache-2.0
