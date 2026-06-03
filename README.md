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

### Otari gateway

Instead of managing API keys per provider, you can route all non-local providers through [Otari](https://github.com/mozilla-ai/otari), Mozilla AI's OpenAI-compatible LLM gateway, by adding a top-level `otari` object:

```json
{
  "providers": [
    {"provider": "openai", "model": "openai:gpt-4o"},
    {"provider": "anthropic", "model": "anthropic:claude-sonnet-4-20250514"}
  ],
  "otari": {
    "api_base": "https://your-gateway.example/v1",
    "api_key": "${OTARI_API_KEY}"
  },
  "timeout_seconds": 90,
  "consensus_threshold": 2
}
```

In Otari mode, Otari — not the SDK — picks the upstream provider, so the per-provider `provider` field becomes a label. Otari expects the `model` field to use a `provider:model` prefix such as `"openai:gpt-4o"`; consult Otari's documentation for its model-naming convention.

`api_base` and `api_key` may also be omitted from the config, in which case they are resolved from the `OTARI_API_BASE` and `OTARI_API_KEY` environment variables. The `api_key` field supports `${ENV_VAR}` references.

For Bearer-token auth against a hosted Otari platform, omit `api_key` and set the `OTARI_PLATFORM_TOKEN` environment variable instead — the Otari client detects it and switches to platform mode automatically.

Providers marked `"local": true` always bypass Otari and continue to use their own `api_base`.

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
