# Decision Hub Demo

Current flagship path:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop --no-bars
```

This is the read-only General Decision Loop preview. It shows:

```text
signal -> state -> candidates -> decision -> approval
-> skill resolution -> context pack -> SDEP handoff
-> fixture outcome -> state feedback snapshot
```

It does not call an executor, send SDEP, persist state, or modify the legacy
runtime. For the focused quickstart, see
[`docs/general_loop_quickstart.md`](../../docs/general_loop_quickstart.md).

To validate the same path locally:

```sh
python examples/decision_hub_demo/smoke_general_loop.py --quiet
```

This example demonstrates a minimal simulation-driven Spice decision loop while
keeping the implementation out of Spice core.

Flow:

```text
Observation
-> WorldState reducer
-> ActiveDecisionContext builder
-> deterministic conflict detection
-> fixed candidate registry
-> structured consequence estimation
-> GuidedDecisionPolicy / decision.md selection
-> recommendation + trace
-> optional confirmation
-> execution request
-> execution_result_observed
-> WorldState reducer update
```

The demo also exposes a General Core read-only path:

```text
provider/demo signal
-> GenericObservation
-> GeneralDecisionState
-> GenericCandidate
-> GenericPolicyAdapter
-> human-readable Decision Card
```

This path is for showing the decision object before confirmation or execution.
It does not call SDEP, Hermes, Codex, or the execution adapter.

Boundary:

- The reducer updates facts only.
- `ActiveDecisionContext` is a derived slice for one decision, not source of truth.
- Candidate actions are fixed by the demo registry.
- The optional LLM simulation model may estimate structured consequences only.
- The LLM may not create candidates or choose the recommendation.
- Final selection is performed by `GuidedDecisionPolicy` using `decision.md`.
- Execution results never mutate state directly; they return as
  `execution_result_observed` observations and pass through the reducer.

Implemented candidate actions:

- `handle_now`
- `quick_triage_then_defer`
- `ignore_temporarily`
- `delegate_to_executor`
- `ask_user`

`delegate_to_executor` is enabled only when Spice has ingested an
`executor_capability_observed` observation and that capability is available for
the required scope. The demo does not use a boolean flag to pretend an executor
exists. Capability must enter `WorldState` first, then `ActiveDecisionContext`
derives whether delegation is runtime-real.

Minimal capability observation shape:

```json
{
  "observation_type": "executor_capability_observed",
  "source": "hermes",
  "observed_at": "2026-04-17T08:00:00+00:00",
  "confidence": 1.0,
  "attributes": {
    "capability_id": "cap.external_executor.codex",
    "action_type": "delegate_to_executor",
    "executor": "codex",
    "supported_scopes": ["triage", "review_summary"],
    "requires_confirmation": true,
    "reversible": true,
    "default_time_budget_minutes": 10,
    "availability": "available"
  }
}
```

In the demo code, `observed_at` is represented by the shared
`Observation.timestamp` field.

The current demo only models `codex` via Hermes as a `delegate_to_executor`
capability. Unsupported scopes or unavailable executors disable the delegate
candidate and record the reason in the trace. `ask_user` is enabled only when
the active context has missing critical information or low-confidence facts.

## Confirmation loop

Recommendations include `requires_confirmation`. The value comes from action
metadata and capability facts, not from LLM output.

When `delegate_to_executor` is selected with `requires_confirmation: true`, the
demo returns a stable confirmation request:

```json
{
  "confirmation_id": "confirm.2026-04-17T08:00:00Z.delegate_to_executor.ab12cd34",
  "decision_id": "decision.2026-04-17T08:00:00Z.workitem.github_pr_123.ab12cd34",
  "selected_action": "delegate_to_executor",
  "acted_on": "workitem.github.dyalwayshappy_spice.123",
  "options": [
    {"key": "1", "value": "confirm"},
    {"key": "2", "value": "reject"},
    {"key": "3", "value": "details"}
  ]
}
```

This is intentionally shaped for WhatsApp mapping:

```text
1 同意执行
2 拒绝
3 查看详情
```

`confirm` creates an execution request and applies the structured outcome back
through `execution_result_observed`. `reject` does not execute and does not
pretend work was handled. `details` returns the decision trace explanation and
keeps the confirmation pending.

`ask_user` does not enter the execution path. It returns a structured prompt for
missing information. `ignore_temporarily` is a no-op in this demo and does not
create an execution outcome.

Minimal delegate execution request params:

```json
{
  "scope": "triage",
  "time_budget_minutes": 10,
  "target_title": "Fix decision guidance validation",
  "target_url": "https://github.com/Dyalwayshappy/Spice/pull/123",
  "success_criteria": "Return status, blocker, risk_change, followup_needed, and a concise summary."
}
```

This demo intentionally does not implement persistent storage, real GitHub
polling, or WhatsApp ingress. Its default execution path is SDEP-backed:
confirmations produce SDEP `execute.request` messages, Hermes/Codex execution
stays behind the SDEP wrapper, and outcomes return as `execution_result_observed`.
Mock or direct Hermes executors are explicit test/debug overrides, not the
public default.

## General Decision Loop preview

The General path is the first display layer for Spice's decision runtime. It
shows the decision as a visible object, then previews how that decision would
cross the approval and execution boundary.

Start with the Decision Card:

```sh
python examples/decision_hub_demo/run_demo.py --general-decision-card --no-bars
```

This renders only the decision comparison: decision-relevant state, candidate
decisions, selected recommendation, and why-not-others. It does not authorize,
execute, or call SDEP.

To inspect the approval checkpoint:

```sh
python examples/decision_hub_demo/run_demo.py --general-approval
```

This approval checkpoint is read-only. It does not store a confirmation, resolve
a confirmation, create an `ExecutionIntent`, or call an executor.

To inspect the planned execution handoff:

```sh
python examples/decision_hub_demo/run_demo.py --general-execution-plan
```

This command uses an in-memory confirmed approval fixture to render the planned
SDEP handoff. It creates an `ExecutionIntent` and an SDEP `execute.request`
payload for inspection only. It does not write confirmation state, send the
request, call Hermes/Codex, receive an outcome, or update state.

The next read-only step shows how a returned SDEP `execute.response` is
attributed back to the same decision, candidate, approval, and execution id:

```sh
python examples/decision_hub_demo/run_demo.py --general-outcome-return
```

This command uses a local fixture response. It converts the response into a
General `OutcomeRecord` and `GenericObservation(kind=outcome)` for inspection.
It does not call an executor, process a live response, or update state.

The final read-only step applies that outcome observation to a new General
state snapshot:

```sh
python examples/decision_hub_demo/run_demo.py --general-state-feedback
```

This command uses the local fixture response and the General reducer to render
the state feedback view. It does not persist state, call an executor, send SDEP,
or modify the legacy demo runtime state.

For the screenshot-friendly full preview, use the flagship read-only General
loop:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop --no-bars
```

This prints the complete read-only chain:

```text
signal -> state -> candidates -> decision -> approval
-> skill resolution -> context pack -> SDEP handoff
-> fixture response -> outcome -> state feedback snapshot
```

In this path, a skill is an execution template / capability hint, the context
pack is compressed execution context, and SDEP remains the protocol boundary.
The executor is still the component that would perform real work.
The text output explicitly marks `read_only`, `sdep_request_sent=false`,
`executor_called=false`, and `persisted=false`.
It is intended to show: Spice selected, resolved a skill, compressed context,
and planned the SDEP handoff.

For machine-readable inspection:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop-json --no-bars
```

Both commands are read-only. They do not persist state, call an executor, send
SDEP, or modify the legacy demo runtime state.
The confirmed approval and SDEP response in this full-loop view are local
fixtures.

The demo can also export a stable compare artifact from the same General Core
read-only path:

```sh
python examples/decision_hub_demo/run_demo.py --write-compare-artifact
```

This writes:

```text
examples/decision_hub_demo/compare_artifacts/meeting_vs_pr_conflict.json
```

The artifact is not a raw trace dump. It is a decision comparison object built
from normalized demo signals and shaped for human-readable inspection.

This path renders a Decision Card only. It does not authorize actions, execute
actions, or call SDEP, Hermes, Codex, or an executor. The legacy scenario path
remains the actual confirmation / execution demo path.

Inspect it with:

```sh
python -m spice.entry decision compare \
  --input examples/decision_hub_demo/compare_artifacts/meeting_vs_pr_conflict.json
```

To include the execution boundary placeholder:

```sh
python -m spice.entry decision compare \
  --input examples/decision_hub_demo/compare_artifacts/meeting_vs_pr_conflict.json \
  --show-execution
```

This section does not execute actions.

Use `--json` to inspect the normalized comparison payload directly.
