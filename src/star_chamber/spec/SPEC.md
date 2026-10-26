# Council Protocol Specification v0.9

This document specifies the portable council deliberation protocol — the invocation modes, prompt templates, response parsing, consensus classification, and output format that any orchestrator can implement independently.

The orchestrator-specific implementation for Claude Code lives in `plugins/pragma/skills/star-chamber/PROTOCOL.md`. That document handles Bash subprocess isolation, `uv run` invocation, temp file management, and other runtime concerns. This spec covers only the portable protocol logic.

**Status:** Draft (v0.9). Breaking changes may occur before v1.0.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Invocation Modes](#2-invocation-modes)
- [3. Provider Configuration](#3-provider-configuration)
- [4. Review Request](#4-review-request)
- [5. Prompt Templates](#5-prompt-templates)
- [6. Execution](#6-execution)
- [7. Anonymous Synthesis](#7-anonymous-synthesis)
- [8. Response Parsing](#8-response-parsing)
- [9. Consensus Classification](#9-consensus-classification)
- [10. Output Format](#10-output-format)
- [11. Security](#11-security)
- [12. Error Handling](#12-error-handling)

## 1. Introduction

The council protocol defines how multiple LLM providers are queried in parallel, how their responses are parsed and compared, and how the results are classified by agreement level. It is designed so that any orchestrator — CI pipelines, GitHub Actions, editor plugins, SDKs — can implement the protocol without coupling to a specific runtime.

### Versioning

This spec follows [Semantic Versioning](https://semver.org/). The current version is **0.9** (pre-stable). The version will advance to 1.0 when the schema and algorithm definitions are considered stable.

A version is published when it ships in a package release. An unpublished version is amended in place, so a change that lands before the next release does not move it. A number this spec has already stated is spent and is not reused, whether or not it was published.

A published version MUST move when the older copy of a schema and the new one disagree about any document. A tightening rejects a document the older copy accepted. A widening, such as a value added to an emitted enum, produces a document the older copy rejects. Either way one number names two checks, and a consumer cannot tell which of them its own copy carries. A rejection discards the whole document rather than the one field, so the version is the only warning a consumer gets.

### Spec Version in Output

An orchestrator implementing version 0.7 or later MUST emit `spec_version`, and MUST emit the version of this spec rather than its own package version, because a package can release a patch without changing the protocol. The schema leaves the property optional so that a document produced before 0.7 still validates, and that is the only reason it is optional. A producer at 0.7 or later that omits it makes its own output indistinguishable from a document written years earlier.

A consumer that finds the field absent is therefore reading a document produced before 0.7. That is a closed range rather than missing data. The consumer MAY apply the semantics this spec documented for those versions, and MUST treat as ambiguous any field whose meaning changed within that range, because the document does not say which of those versions produced it. `summary.synthesis` and `recommended_by` are both in that set.

The field exists to guard against staleness, and guarding is its only sanctioned use. A consumer reading a document produced by a version above the one its own instructions were written against MUST state, in whatever it produces, that the document comes from a newer version of this protocol and that its field-level reading is therefore unverified. It MUST NOT abort on that ground alone, and it MUST NOT silently drop a field it does not recognize. Rendering the document with that statement attached is the required behavior, because the alternative is withholding a document the reader can still mostly use.

A consumer MAY select behavior per field for a version at or below the one its instructions were written against. Those versions form a closed set that this spec documents, so a rule written for them stays correct, and a reader can check it.

A consumer MUST NOT select behavior per field for a version above that. Such a rule guesses at semantics this spec had not defined when the rule was written, and it goes silent at the next version without announcing it, which converts an open ambiguity into a closed enumeration that someone has to maintain, in a document that cannot be recompiled and that an agent executes literally. Reporting an unverified reading keeps the ambiguity open and visible, which is the weaker claim and the honest one.

A consumer that forwards part of a document, such as passing only `summary` into a prompt, MUST carry the version with it or state it alongside. A fragment carries no provenance of its own, and whoever reads the fragment has no way to ask.

The schema MUST constrain the value no further than a string. A pattern or an enum on this field would let a consumer reject the whole document on the value of the one field that exists to tell it the document is newer than its instructions, which is the failure the field was added to prevent. The same reasoning applies to the version this spec states for itself: it is a version, not a format that consumers parse.

Opening a published schema and carrying the version answer different questions. The open schema stops a consumer rejecting a document it could otherwise read. The version tells a consumer that its reading may be out of date. Neither substitutes for the other, and the version is only reachable by a consumer whose schema already accepts the document, which is why the output schema is published open before this field is relied upon.

### Schema Strictness

A schema describing what the council emits MAY be published open to keys it does not declare. A schema describing what the council receives MUST stay closed, because strictness there guards untrusted input. Only [`schemas/council-output.schema.json`](schemas/council-output.schema.json) is published open.

A consumer that validates a document against an older copy of a closed schema rejects it on the fields that copy has never heard of. Every field the protocol has added has had that effect on every orchestrator pinned to the previous schema, and the schemas ship inside the package, so the pinned copy is the common case.

Producers are still held to the full strictness of the format. The strict variant is derived from the published schema rather than maintained beside it, because a strict schema and a tolerant one written by hand can disagree with each other and no test would catch it. The derivation states `additionalProperties: false` on every subschema that declares `properties` for an instance of its own and states no `additionalProperties` already.

A subschema describes an instance of its own when it is reached through `$defs`, `additionalProperties`, `contains`, `items`, `patternProperties`, `prefixItems`, `properties`, `propertyNames`, or `unevaluatedProperties`. A subschema reached through `allOf`, `anyOf`, `dependentSchemas`, `else`, `if`, `not`, `oneOf`, or `then` constrains the same instance as the schema that carries it. The derivation MUST NOT enter one of those, nor anything beneath it. Such a branch describes part of an instance that another branch also describes, so closing it rejects the keys the other branch declares, and closing an `if` stops the branch matching at all, which withdraws the rule it carries rather than enforcing it. The derivation MUST also ignore `const`, `default`, `enum`, and `examples`, whose contents are instances rather than subschemas.

A schema therefore gains no strictness for a shape declared only inside such a branch. A branch MUST state which of the properties already declared for an instance are required, and MUST NOT introduce a property of its own. `additionalProperties` is blind to composition, so a property declared only in a branch is rejected by the strictness derived where the instance is described.

`additionalProperties` therefore carries three meanings in a published schema, and none of the three is spare:

| Form | Meaning |
|------|---------|
| Absent | Open to a consumer, strict to the producer under the derived variant. |
| `true` | Open to everyone, the producer included. |
| A schema | The object is a map, and the schema describes its values. |

An object the producer may extend freely MUST state `additionalProperties: true`. An implementation MUST NOT state `true` merely to document that a consumer may read unknown keys, because absence already says that, and `true` is the only remaining way to express an object no derived strictness applies to.

This package exports the derivation as `star_chamber.strict_variant`, so a consumer applying the producer's own check runs the same transform instead of reimplementing this rule.

Open publication covers undeclared keys and nothing else. A consumer validating against an older copy still rejects a document on a value added to an enum after that copy, because an enum closes a set of values rather than a set of keys. Widening an emitted enum therefore remains a change that breaks a pinned consumer.

### Terminology

| Term | Definition |
|------|-----------|
| **Provider** | An LLM service (e.g., OpenAI, Anthropic, Gemini) or a local model endpoint. |
| **Council** | The set of providers consulted for a single review or question. |
| **Round** | One fan-out/fan-in cycle across all providers. |
| **Consensus** | An issue or recommendation flagged by all successful providers. |
| **Majority** | An issue or recommendation flagged by more than one but not all providers. |

## 2. Invocation Modes

The protocol supports two invocation modes. The mode determines the prompt template and aggregation strategy.

| Mode | Trigger | Prompt Template | Aggregation |
|------|---------|----------------|-------------|
| **Code Review** | Default. Files provided, no design question. | [5.1 Code Review Prompt](#51-code-review-prompt) | Group by file location, classify by agreement. |
| **Design Question** | User asks an architecture/design question. | [5.2 Design Question Prompt](#52-design-question-prompt) | Group by approach recommendation. |

**Code review** is the default mode. It reviews files for craftsmanship, architecture, correctness, invariants, and maintainability.

**Design question** mode is triggered when the user asks about architecture, design trade-offs, or approach (e.g., "should we use event sourcing or CRUD?"). No files are reviewed — providers give advisory recommendations.

## 3. Provider Configuration

Provider configuration defines which LLM providers the council consults and how to reach them.

**Schemas:**
- Single provider entry: [`schemas/provider-config.schema.json`](schemas/provider-config.schema.json)
- Full config file: [`schemas/council-config.schema.json`](schemas/council-config.schema.json)

### Provider Fields

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | Provider name (e.g., `openai`, `anthropic`, `llamafile`, `ollama`). |
| `model` | Yes | Model identifier (e.g., `gpt-4o`, `claude-sonnet-4-20250514`). |
| `api_key` | No | API key or `${ENV_VAR}` reference. Omit when routing through Otari or for keyless local providers. |
| `max_tokens` | No | Budget in tokens for this member's reply, sent to the provider as given. Not a capability read from the model. Default: 16384. |
| `api_base` | No | Custom base URL. Use for local/self-hosted LLMs. Omit for cloud providers — SDKs use built-in defaults. |
| `local` | No | Set to `true` for local/self-hosted providers. Default: `false`. |
| `display_name` | No | Unique identity for this member in council output, used in place of the provider name. Lets several entries share one provider (e.g. an OpenAI-compatible gateway) while staying distinct voters. Defaults to the provider name. |

`max_tokens` is a budget, so an orchestrator neither reads a limit from the model nor checks the value against one. A budget the reply does not fit is reported by the orchestrator, and a budget the model does not permit is reported by the provider.

### API Key Resolution

Keys are resolved in this order:

1. `api_key` field in the provider config (literal value or `${ENV_VAR}` reference).
2. Environment variable matching the provider convention (e.g., `OPENAI_API_KEY`).

### Otari Gateway

When the top-level `otari` object is set, non-local providers are dispatched through [Otari](https://github.com/mozilla-ai/otari), an OpenAI-compatible gateway. Every non-local provider entry is routed to the same `api_base` with the same `api_key`; the per-provider `provider` field is treated as an identifier label rather than as the routing target. `api_base` and `api_key` may be specified inline in the config or omitted and resolved from the environment: `api_base` from `OTARI_API_BASE`, then `GATEWAY_API_BASE`; `api_key` from `OTARI_API_KEY` for the hosted otari.ai platform, or `GATEWAY_API_KEY` for a self-hosted gateway. Both fields support `${ENV_VAR}` references. `api_base` is the gateway origin with no path because the client adds the API path. An effective base that ends in `/v1` or `/api/v1` is a configuration error, reported before any provider call with the setting that supplied it and the corrected origin.

### Local Provider Semantics

Local providers (`local: true`) use `api_base` to reach a local or self-hosted endpoint and are dispatched directly even when Otari is configured. The `local` flag preserves direct routing to the local endpoint.

## 4. Review Request

A review request captures everything an orchestrator needs to execute a council round.

**Schema:** [`schemas/review-request.schema.json`](schemas/review-request.schema.json)

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `mode` | Yes | `"code-review"` or `"design-question"`. |
| `files` | Code review only | Array of file objects with `path` and `content`. |
| `prompt` | Design question only | The user's design question. |
| `context` | No | Project rules and architecture context. |
| `providers` | No | Override which providers to consult (defaults to all configured). |

## 5. Prompt Templates

### 5.1 Code Review Prompt

The code review prompt instructs providers to review code for five focus areas and return structured JSON.

```
You are a senior software craftsman reviewing code for quality, idioms, and architectural soundness.

## Project Context
{Injected project rules, if any}
{Architecture context, if available}

## Code to Review
{For each file: ### {path}\n{content}}

## Review Focus
1. Craftsmanship: Is this idiomatic, clean, well-structured?
2. Architecture: Does this fit the project's patterns? Any design concerns?
3. Correctness: Any logical issues, edge cases, or bugs?
4. Invariants: Do classifications (terminal, final, immutable) match runtime reality?
   Are there states where cleanup or cancellation is assumed but not enforced?
   Does the code's model of the system match what actually happens?
5. Maintainability: Will this be easy to understand and modify later?

## Output Format
Provide your review as structured JSON:
{
  "provider": "your-name",
  "quality_rating": "excellent|good|fair|needs-work",
  "issues": [
    {
      "severity": "high|medium|low",
      "location": "file:line or file:firstLine-lastLine",
      "category": "craftsmanship|architecture|correctness|invariants|maintainability",
      "description": "What is wrong",
      "suggestion": "How to fix it"
    }
  ],
  "praise": ["What is done well"],
  "summary": "One paragraph overall assessment"
}
```

**Response schema:** [`schemas/code-review-result.schema.json`](schemas/code-review-result.schema.json)

### 5.2 Design Question Prompt

The design question prompt instructs providers to advise on architecture/design decisions.

```
You are a senior software architect advising on design decisions.

## Project Context
{Injected project rules, if any}
{Architecture context, if available}

## Design Question
{The user's question}

## Advisory Focus
1. Trade-offs: What are the pros and cons of each approach?
2. Fit: Which approach best fits this project's existing patterns and constraints?
3. Risk: What are the risks of each option? What could go wrong?
4. Recommendation: What would you recommend and why?

## Output Format
Provide your advice as structured JSON:
{
  "provider": "your-name",
  "recommendation": "Your recommended approach",
  "approaches": [
    {
      "name": "Approach name",
      "pros": ["..."],
      "cons": ["..."],
      "risk_level": "low|medium|high",
      "fit_rating": "excellent|good|fair|poor"
    }
  ],
  "summary": "One paragraph overall recommendation with reasoning"
}
```

**Response schema:** [`schemas/design-advice-result.schema.json`](schemas/design-advice-result.schema.json)

## 6. Execution

### 6.1 Parallel Mode (default)

All providers are queried simultaneously in a single round. Responses are collected and passed to the parsing and classification stages. One member that answered then writes the session's closing assessment (see [section 7.2](#72-closing-assessment)).

```
Prompt → [Provider A] → Response A
      → [Provider B] → Response B    (all at once, independent)
      → [Provider C] → Response C
```

### 6.2 Debate Mode

Debate mode runs multiple rounds where providers see anonymised summaries of prior responses.

**State machine:**

```
INIT → ROUND_1 → SYNTHESIZE → ROUND_2 → SYNTHESIZE → ... → ROUND_N → FINAL
```

**Round flow:**

1. Fan out the prompt to all providers in parallel.
2. Collect responses.
3. Synthesize responses anonymously (see [section 7](#7-anonymous-synthesis)).
4. Augment the original prompt with the synthesis.
5. Repeat from step 1 for the next round.
6. After the final round, pass last-round responses to parsing and classification.

**Convergence detection:** If responses in round N are substantively identical to round N-1 (providers agree with no new points), the orchestrator MAY stop early. Completing all requested rounds is also acceptable.

**Error handling during rounds:** If a provider fails during a round, continue with remaining providers. Note failed providers in the output. Failed providers are excluded from convergence detection.

## 7. Anonymous Synthesis

A synthesis reads across the members and reports on the session, never on a member. The protocol synthesizes at two points: between debate rounds, and once at the end of every session.

### 7.1 Between Rounds

Between debate rounds, the orchestrator creates a single anonymous summary of all responses from the previous round.

### Rules

1. Synthesize by content theme — never attribute points to individual providers.
2. Group similar concerns together.
3. Highlight areas of agreement and disagreement.
4. Ask providers to engage with the ideas, not the sources.

### Synthesis Template

```
## Other council members' feedback (round {N}):

**Issues raised:**
- {Grouped concern 1}
- {Grouped concern 2}

**Points of agreement:**
- {Shared observation 1}

**Points of disagreement:**
- {Divergent view 1}

Please provide your perspective on these points. Note where you agree, disagree, or have additional insights.
```

This approach follows [Chatham House rules](https://www.chathamhouse.org/about-us/chatham-house-rule): information can be used freely, but the identity of the source is not revealed.

### 7.2 Closing Assessment

After the final round, one member writes the `summary.synthesis` the output carries. The rules above apply unchanged: the material it reads carries no member identity, so the assessment cannot attribute a point to a member.

**Author.** The orchestrator MUST ask the first member of the configured roster that returned usable output, so one roster always asks the same member. A council whose members all failed is asked for nothing.

**Material.** The orchestrator MUST pass each answering member's own summary, and MAY pass the ratings, recommendations, and the concerns or approaches already grouped by agreement level. It MUST NOT pass a member display name. Where the material is empty, no member is asked and `summary.synthesis` is empty, because an author given nothing can only invent.

**Material.** Every entry MUST be reduced to one line, then empty entries dropped, then the rest sorted. Sorting last is what makes the order independent of leading whitespace. Sorting at all is what stops the material preserving reply order or roster order, which would let a reader count each unnamed member's points or line one section up against another. A per-member verdict list MUST NOT appear; a tally of verdicts MAY.

**Result.** The prompt MUST ask its author for one or two sentences of prose, and the assessment MUST NOT be assembled from the members' own summaries. The orchestrator MUST NOT reject an assessment on a sentence count. A sentence count cannot be decided reliably on prose, where an abbreviation ends a sentence that continues, and a reader is better served by a third sentence than by the empty field a false rejection would leave. Length is bounded instead, as the next rule describes. `summary.synthesized_by` MUST name the member asked, whether or not it wrote one, so a reader can check the author against the rule above. Where the author returns nothing usable, including a reply too long to be the prose asked for, `summary.synthesis` MUST be empty and `summary.synthesis_error` MUST name the member asked and the reason. A concatenation of the members' summaries is not a synthesis and MUST NOT stand in for one. An empty assessment with no recorded failure means none was requested; a reader tells "no member answered" from "nothing to assess" by whether `providers_responded` is empty.

**Reporting it.** `summary.synthesis_error` and `summary.synthesized_by` MUST both be present. Where a member was asked, `summary.synthesized_by` names it, and `summary.synthesis_error` is null unless that member wrote nothing usable. Where no member was asked, both MUST be null. An absent field states nothing, so the schema requires both rather than leaving the question to the reader. The schema states the pairing as a conditional on `summary`, because a failure recorded against no member reports a contradiction rather than an absence.

**Older documents.** A consumer validating against this version's schema rejects a document whose summary omits either field, and one whose summary carries a combination these rules forbid. A document produced before 0.5 carried neither field. A document produced by a published version from 0.5 on was described by prose that stated both, while its schema declared both and required neither, so a producer could omit one and pass. That rejection does not contradict [Schema Strictness](#schema-strictness). Open publication stops a consumer rejecting a document newer than its own schema, where the undeclared key is still readable and `spec_version` warns that the reading may be stale. A missing required field is the opposite case, because the document is older rather than newer, and a document from before 0.7 cannot say how much older it is. A reader that cannot tell which member was asked cannot check the author against the rule above, and that check is the reason both fields exist.

**Cost.** The assessment costs one model call beyond the fan-out, and it runs after them, so a session's worst case is the fan-out timeout plus one more. Debate mode pays that second timeout once per round.

**Declining it.** An orchestrator MUST honor a council configured with `synthesis` false by asking no member and spending no call. The output then carries the same empty assessment and null failure as a session where none was requested for any other reason, because none was.

## 8. Response Parsing

### Algorithm

For each provider response:

1. **Try extracting JSON from a Markdown code block.** Look for `` ```json `` followed by a JSON object and closed by `` ``` ``. Extract and parse the JSON object.
2. **Fallback: try parsing the entire response as JSON.** If step 1 finds no code block, attempt to parse the full response text as JSON.
3. **Malformed response:** If both steps fail, mark the provider as failed with a "malformed response" error. Continue processing other providers.

### Validation

After extraction, validate the parsed JSON against the appropriate schema:
- Code review mode: [`schemas/code-review-result.schema.json`](schemas/code-review-result.schema.json)
- Design question mode: [`schemas/design-advice-result.schema.json`](schemas/design-advice-result.schema.json)

If validation fails, the orchestrator SHOULD still attempt to use the response (best-effort parsing) but MAY mark it as degraded.

### Closed Sets

Five fields name a closed set: `quality_rating`, `severity` and `category` on an issue, and `risk_level` and `fit_rating` on an approach. A value outside the set is not a case for best-effort parsing. The orchestrator MUST fail that member's reply, and the error MUST name the field and the values a member may send, so the reply can be corrected rather than guessed at. An orchestrator that copies the value through instead emits a document its own output schema rejects while reporting that member as a success, so a reader is given neither the failure nor a usable document.

An orchestrator MUST NOT map an unexpected value onto a nearer permitted one. Mapping invents a verdict the member never gave, and a category outside the set has nothing to map onto.

`risk_level` is the one field whose permitted values are narrower than the values the output carries. `unknown` reports that no member rated an approach, so a member MUST NOT send it, and an orchestrator that receives it MUST treat it as outside the set. An approach a member leaves unrated carries `unknown` because the orchestrator put it there, which is the same value reached by a different route.

### Provider Identity

The response wrapper ([`provider-response.schema.json`](schemas/provider-response.schema.json)) carries two identity fields: `provider` is the provider configured for the member, preserved from the request (not necessarily the effective route — under Otari routing the call is dispatched through Otari while this field keeps the configured provider), and `display_name` is the member identity — set by the orchestrator, equal to the configured `display_name` or the provider name when none is set. The parsed result body ([`code-review-result.schema.json`](schemas/code-review-result.schema.json), [`design-advice-result.schema.json`](schemas/design-advice-result.schema.json)) also carries a `provider` field, but that inner value is set by the LLM and may not match. Orchestrators SHOULD key all classification and reporting on the wrapper `display_name`, and MAY ignore the inner `provider` field. Keying on `display_name` is what lets several members share one provider (e.g. behind an OpenAI-compatible gateway) while remaining distinct voters.

## 9. Consensus Classification

### 9.1 Issue Grouping (Code Review)

Issues from different providers are grouped when they refer to the same underlying concern:

- **Same file** AND **same line range** (overlapping, or within ±5 lines tolerance) AND **same category** = same issue.
- A location MAY cite a single line (`file.py:205`) or a line range (`file.py:203-207`). A single line is treated as a range of one line, so the two forms group with each other.
- A location whose trailing segment is neither is kept whole as the file path, so a path containing a colon is not split.
- Orchestrators MAY supply a custom semantic similarity function for more sophisticated grouping.

### 9.2 Approach Grouping (Design Question)

The approaches array includes all approaches mentioned by any provider — both recommended and discussed-but-rejected alternatives. An approach with `recommended_by: 0` was considered by one or more providers but not recommended by any.

Approaches from different providers are grouped when they refer to the same approach (by name match or semantic similarity). The `recommended_by` count reflects how many providers selected that approach as their primary recommendation.

Names MUST be matched with case and repeated spacing ignored, so `Repository` and `repository  layer` are the same name only if the remaining words match. Punctuation is ignored except for characters that tell two names apart, which MUST be preserved, so `C++`, `C#` and `C` remain three approaches rather than one. Letters and digits MUST be preserved whatever script they are written in, so an approach named outside the Latin alphabet keeps a name to group and match on. Orchestrators MAY add semantic similarity on top, as in section 9.1. A grouped approach carries the union of the pros and the union of the cons, in the order providers first stated them, with duplicates removed.

An aggregated `risk_level` is the level the most providers stated, and an aggregated `fit_rating` is the rating the most providers stated. A tie MUST resolve toward the more cautious value: the higher risk, or the lower fit. A provider that stated no level or no rating does not vote, so `risk_level` is `unknown` and `fit_rating` is null only when no provider stated one. An approach MUST carry `fit_rating`, because a null states that nobody rated the approach while an absent field states nothing. Each provider votes once per approach, so a provider that names one approach more than once MUST NOT outvote a provider that names it once.

Providers cannot report `recommended_by`, because the member output schema does not carry it. An orchestrator therefore derives it: a provider counts toward an approach when its `recommendation` names that approach as whole words. A name MUST be counted only where it stands on its own, so a recommendation of `CRUD with domain events` does not also count toward an approach named `CRUD`. The recommendation MUST be read against every grouped name before the eligible names are selected, because the longer name may be one another provider raised. Only the approaches that provider named itself are eligible, so one provider cannot be counted for an approach that another provider raised, and a provider that recommends a name it never listed counts toward nothing. Two limits follow from naming alone. A recommendation that names an approach only to reject it is counted, and a recommendation that describes an approach without naming it is not.

### 9.3 Classification Buckets

Issues/recommendations fall into mutually exclusive buckets:

| Bucket | Definition | Confidence |
|--------|-----------|------------|
| **Consensus** | Every member that answered flagged it. | Highest |
| **Majority** | More than one but not all members that answered. | High |
| **Individual** | Exactly one member that answered. | Lower |

Classification counts a member only when it returned a usable response. The denominator is therefore `providers_responded`, not `providers_used`. A consensus across two members when three were asked is a weaker signal than the same word across three, so consumers that report agreement MUST state which roster it was computed over.

**Design question consensus:** For design questions, `consensus_recommendation` captures high-level directional agreement (e.g., "use CRUD, not event sourcing"), which a synthesis step can establish even when providers recommend different specific variants. It is not populated when providers disagree on direction. Per-approach `recommended_by` counts track the specific variant each provider chose.

Without a synthesis step an orchestrator MUST populate it only when every provider that answered named one and the same approach, and the value MUST be that approach's name. It MUST be null otherwise. A consumer can therefore always read the value as a name that appears in `approaches`, rather than having to tell a name from a provider's prose. Directional agreement across differently named variants reads the shared direction out of differing prose, which needs synthesis, so an orchestrator that performs none reports null in that case rather than guessing.

### 9.4 Confidence Ordering

Results are ordered: consensus first, then majority, then individual. Within each bucket, order by severity (high → medium → low) for code review, or by provider count for design questions.

Orchestrators MAY apply per-provider calibration weights to adjust confidence, but the default is equal weighting.

## 10. Output Format

**Schema:** [`schemas/council-output.schema.json`](schemas/council-output.schema.json)

### 10.1 Code Review Output

The structured output includes:

- `mode`: `"code-review"`
- `spec_version` (optional): The version of this spec that produced the document, as [section 1](#spec-version-in-output) requires.
- `files_reviewed[]`: List of reviewed file paths.
- `providers_used[]`: The roster asked. All council members consulted, including those that failed, by display name.
- `providers_responded[]`: The roster that answered. Members that returned a usable review, by display name. This is the denominator the classifications below were computed over.
- `failed_providers[]`: Members that failed or returned malformed responses, by display name, each with the model it was asked and the reply it sent.
- `consensus_issues[]`: Issues flagged by every member in `providers_responded`.
- `majority_issues[]`: Issues flagged by more than one member in `providers_responded`. Each item includes `provider_count` and `flagged_by` (array of member display names) to identify which members raised it.
- `individual_issues{}`: Issues keyed by member display name.
- `quality_ratings{}`: Per-member quality rating, keyed by display name.
- `reviews[]`: Each answering member's parsed review in full, including its raw response text.
- `summary{}`: Counts (`total_issues`, `consensus_count`, `majority_count`), `synthesis` (a short overall assessment, written as [section 7.2](#72-closing-assessment) requires, empty when no member wrote one), `synthesis_error` (the member asked and the reason it wrote none, null otherwise), and `synthesized_by` (the member asked, null when none was). `total_issues` counts distinct issues across all three buckets.
- `debate{}` (optional): Debate metadata (rounds_completed, converged).

**Markdown presentation** (for human consumption):

```markdown
## Council Review

**Files:** {list of files reviewed}
**Providers asked:** {providers_used}
**Providers answered:** {providers_responded}

### Consensus Issues (All {M} Providers That Answered Agree)

These issues were flagged by every council member that answered. Address these first.

1. `{file}:{line}` **[{SEVERITY}]** - {description}
   - **Suggestion:** {how to fix}

### Majority Issues ({N}/{M} Providers)

These issues were flagged by most council members.

1. `{file}:{line}` **[{SEVERITY}]** ({which providers}) - {description}
   - **Suggestion:** {how to fix}

### Individual Observations

Issues raised by a single provider. May be valid specialised insights.

- **{Provider}:** `{location}` - {observation}

### Summary

| Provider | Quality Rating | Issues Found |
|----------|---------------|--------------|
| {name}   | {rating}      | {count}      |

**Overall:** {1-2 sentence synthesis of the review}
```

### 10.2 Design Question Output

The structured output includes:

- `mode`: `"design-question"`
- `spec_version` (optional): The version of this spec that produced the document, as [section 1](#spec-version-in-output) requires.
- `prompt`: The original design question (enables rendering the Markdown template without external context).
- `providers_used[]`: The roster asked. All council members consulted, including those that failed, by display name.
- `providers_responded[]`: The roster that answered. Members that returned usable advice, by display name. This is the denominator `consensus_recommendation` was computed over.
- `failed_providers[]`: Members that failed or returned malformed responses, by display name, each with the model it was asked and the reply it sent.
- `consensus_recommendation`: High-level directional agreement when every member that answered agrees. Differing specifics count only where a synthesis step establishes the shared direction. Without one the field carries the name of the one approach every member named, and is null otherwise, as section 9.3 requires.
- `approaches[]`: All approaches mentioned by any provider, with `recommended_by` count, merged pros/cons, `risk_level`, and `fit_rating`. `risk_level` is `unknown` when no provider stated one. `fit_rating` is null when no provider assessed fit for the approach; when providers disagree, use the most common rating.
- `summary{}`: Contains `synthesis` (a short overall assessment, written as [section 7.2](#72-closing-assessment) requires, empty when no member wrote one), `synthesis_error` (the member asked and the reason it wrote none, null otherwise), and `synthesized_by` (the member asked, null when none was).
- `debate{}` (optional): Debate metadata (rounds_completed, converged).

**Markdown presentation:**

```markdown
## Council Advisory

**Question:** {the design question}
**Providers asked:** {providers_used}
**Providers answered:** {providers_responded}

### Consensus Recommendation

{If all providers agree on an approach, state it here}

### Approaches Considered

**{Approach name}** - Recommended by {N}/{M} providers
- **Pros:** {merged pros}
- **Cons:** {merged cons}
- **Risk:** {risk level}

### Dissenting Views

{Any provider that recommended a different approach, with their reasoning}

### Summary

| Provider | Recommendation | Fit Rating |
|----------|---------------|------------|
| {name}   | {approach}    | {rating}   |

**Overall:** {1-2 sentence synthesis}
```

## 11. Security

### Key Handling

- **Never log, echo, or print API key values.** Only check presence (e.g., "key is set" vs "key is missing").
- Avoid shell expansions (`${VAR:-...}`, `${VAR:+...}`) on key variables in log/print statements — these can leak values when the variable is set.
- Use `${ENV_VAR}` references in config files rather than hardcoding keys.
- Never commit config files containing actual API keys to version control.

### Redaction

Error messages MUST redact API key values. Recommended redaction patterns:
- `sk-*` → `sk-***`
- `ANY.v1.*` → `ANY.v1.***`
- Any string matching a known key prefix → first 6 characters + `***`

### Config File Permissions

Orchestrators SHOULD set restrictive permissions on config files containing key references (e.g., `chmod 600`).

## 12. Error Handling

### Provider Failure

If a provider fails (network error, auth error, timeout), continue with remaining providers. Record the failure in `failed_providers[]` with the error message.

Every recorded failure MUST name the `model` that was asked. A member display name does not give it when several members share one provider, which is the arrangement a gateway makes ordinary.

Every recorded failure MUST carry `raw_content`, whatever the member sent. A description of a failure is not evidence of it, and the reply is the only copy: an orchestrator holds it for the length of one session and nothing else persists it. A failure that produced no reply at all states `raw_content` as null. That means the call returned no message from the member: a timeout, an authentication failure, a response carrying no choices, or a failure nothing anticipated. A message that arrived carrying no content is a reply rather than an absence, so it states `raw_content` as empty, which is a different fact and MUST NOT be reported as null. The two say different things to whoever reads the failure: null points at the network or the gateway, and empty points at the model or the prompt.

The reply MUST be retained whole rather than truncated. A reply is diagnosed from where it stops, so keeping its beginning and discarding its end withholds the part that explains the failure. The reply is already bounded by the output token budget the request set.

A reply the provider cut short at that budget, or withheld, is a failure rather than partial content, and its reply MUST be retained on the failure like any other. These are the failures most worth diagnosing, because the reply shows where it stopped.

A provider that declines to answer may state its reason outside the reply content. Where it does, that statement MUST be retained as the reply, because it is the member explaining itself and nothing else records it.

A failure that no part of this protocol anticipated MUST be recorded against the member that caused it, and MUST NOT end the round. The output document is the only durable record of a session, so a round that ends early discards the replies of every member that answered.

### All Providers Fail

If all providers fail, report the error with all failure details. Do not produce partial output — there is no consensus to classify.

### Timeout

Each provider MAY have a configurable timeout (`timeout_seconds` in the council config). The default is 60 seconds. On timeout, mark the provider as failed and continue.

### Malformed Response

If a provider returns a response that cannot be parsed (see [section 8](#8-response-parsing)), mark it as failed and continue with other providers. Record the unparsed reply as `raw_content` on the failure. A reply that could not be parsed is the one reply a reader needs, and the parse error names a symptom rather than showing it.

### Synthesis Failure

If the closing assessment cannot be obtained, report it as [section 7.2](#72-closing-assessment) requires. The rest of the output is unaffected.

---

## Examples

See the [`examples/`](examples/) directory for complete worked examples:

- [`examples/code-review/`](examples/code-review/) — A 3-provider code review with consensus, majority, and individual findings.
- [`examples/design-question/`](examples/design-question/) — A 2-provider design advisory with different recommendations.
