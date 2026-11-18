# langchain-jev

Decision middleware for [`deepagents`](../../deepagents) built on System One models: TypeSafe's hosted [Jev](https://docs.typesafe.ai/), or an open-source model such as H2O-Lightning served behind a System One shim.

Jev reads text and returns typed answers with calibrated probabilities instead of generated text. This package uses those answers for the decisions an agent makes around its LLM. Every agent blueprint from Typesafe's [use-case map](https://docs.typesafe.ai/concepts/use-case-map) is implemented: guardrails, model routing, a fast path that skips the LLM, skill suggestion, RAG passage classification, memory, tool gating, answer verification with escalation, and run review.

Three properties hold throughout:

- **Swap the Jev model by name.** Every blueprint talks to Jev through one `BaseDecisionModel`, so changing models is a configuration change.
- **Turn any blueprint off at any time.** Set it at build time, change it without a restart, or override it for a single run.
- **Questions come from the docs.** Every question is in `questions.py` next to the Typesafe page it comes from, and every threshold is in `thresholds.py`.

## Quick start

```python
from deepagents import create_deep_agent
from langchain_jev import JevConfig, register_jev_profile

# Install the Jev stack on every deep agent whose model comes from these providers:
# the main agent, the general-purpose subagent, and declarative subagents.
register_jev_profile(["openai"], JevConfig(model="h2o-lightning-4b"))

agent = create_deep_agent(model=chat_model, tools=tools)
```

`register_jev_profile` uses `HarnessProfile.extra_middleware`, so subagents pass the same checks as the main agent. Passing `middleware=jev_middleware(config)` to `create_deep_agent` also works, but only covers the main agent. A tool call that the main agent delegates through `task` then runs without the tool gate.

## Blueprints

| Id | Where it acts | Hook | Typesafe source |
| --- | --- | --- | --- |
| `input_guard` | Before the LLM sees the user's message | entry request, `before_agent` | [Guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails) |
| `tool_result_guard` | Tool results, before the LLM reads them | `before_model` | [RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) (injection) |
| `model_router` | Picks the model for each call | entry request, `wrap_model_call` | [LangChain integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe) |
| `chat_mode` (opt-in) | Drops tools when one reply answers the request | entry request, `wrap_model_call` | open-source-jev mode question |
| `fast_path` | Answers simple requests without the LLM | entry request, `before_agent` | [Function calling](https://docs.typesafe.ai/cookbooks/function_calling), [companion Noul](https://docs.typesafe.ai/patterns/companion-noul) |
| `skill_suggestion` | Hints at one skill per turn | `before_model`, `wrap_model_call` | [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion) |
| `skill_pinning` | Loads the winning `SKILL.md` directly | `pinned_skills` | deepagents `SkillsMiddleware` |
| `skill_load_check` | Checks `read_file` on a `SKILL.md` | `wrap_tool_call` | [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion) (fits check) |
| `rag_passages` | Filters retrieval tool results | `wrap_tool_call` | [Classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) |
| `memory_read` / `memory_write` | Loads saved facts / saves lasting ones | `before_agent` / `after_agent` | [Simple memory](https://docs.typesafe.ai/cookbooks/basic_memory) |
| `tool_gate` | Intent, leak, irreversibility, and injection checks before a tool runs | `wrap_tool_call` | `AutoModeMiddleware`, [agent trace safety checks](https://evals.typesafe.ai/agent_trace_observability) |
| `output_guard` | Before the user sees the reply | `after_model` | [Guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails) |
| `verify_answer` | Does the answer follow from the evidence? | `after_agent` | [Verifying answers](https://docs.typesafe.ai/cookbooks/verify_llm_outputs) |
| `citation_check` | Are quotes in their sources, and do they support the claim? | `after_agent` | [Citation check](https://docs.typesafe.ai/cookbooks/citation_check) |
| `answer_scoring` | Weighted quality scores | `after_agent` | [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring) |
| `escalation` | Retries a failed answer on the stronger model | `after_agent` → `wrap_model_call` | [SDE cascade](https://docs.typesafe.ai/cookbooks/sde_cascade) |
| `run_review` | Hands finished runs to the trace review | `after_agent`, `review_run()` | [Agent trace observability](https://evals.typesafe.ai/agent_trace_observability) |

Questions about the user's message share one Jev request. This covers the guards, routing, chat mode, and fast-path routes, following [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out). Blueprints that need settings are only installed when those settings are given: `router` for routing and escalation, `fast_path` routes, `rag` retrieval tools, and `review` callbacks.

## Swapping the Jev model

```python
from langchain_jev import JevConfig, JevModel, SystemOneClassifier, register_jev_model

register_jev_model(
    JevModel(
        name="better-jev",
        factory=lambda: SystemOneClassifier(base_url="http://gpu-box:8741", model="better-jev", send_model=False),
        thresholds={"input_guard.action": 0.8},  # re-tuned for this model's probabilities
    )
)
config = JevConfig(model="better-jev")  # or JEV_MODEL=better-jev with JevConfig()
```

- **Built-in names:**
  - `h2o-lightning-4b` (the default; reads `H2O_BASE_URL`)
  - `h2o-lightning-31b`
  - `typesafe` (reads `TYPESAFE_API_KEY` and `TYPESAFE_MODEL`)
- **One API for every server.** `SystemOneClassifier` speaks `POST /v1/systemone`. It fills in fields some servers leave out, using the formulas from the [confidence guide](https://docs.typesafe.ai/confidence), so answers from different servers compare on the same scale.
- **Different models per blueprint:** `JevConfig(models={"input_guard": "h2o-lightning-31b"})`.
- **Try a candidate safely:** `JevConfig(shadow_model="better-jev")` asks a candidate alongside the main model and reports the questions they disagree on.
- **Clamped probabilities.** A server that clamps Noul probabilities (the h2o shim's `noul_floor`, 0.801 by default) turns every threshold between the clamp edges into 0.5. The package logs which thresholds are affected. Set the server's `noul_floor` to 0 to keep the raw probabilities.
- **Threshold order.** Thresholds are layered, with later layers winning: package defaults, then the guard policy, then the model's own `thresholds`, then `JevConfig.thresholds`.

## Switching blueprints

Each blueprint is `off`, `shadow` (ask Jev and record, but change nothing), or `on`.

| Level | How | Takes effect |
| --- | --- | --- |
| Build | `JevConfig(blueprints={"fast_path": "off"})`; `off` leaves it out of the stack | when the agent is built |
| Operator | `mode_source=FileModeSource("jev_modes.json")`, `EnvModeSource()` (`JEV_MODES=tool_gate=off`), or `StaticModeSource` | next run, no restart |
| Per run | `agent.invoke(..., context={"jev_modes": {"tool_gate": "shadow"}})` | that run |

- **Precedence.** The more specific level wins. Overrides cannot enable a blueprint that was not built.
- **Parts.** A part narrows its blueprint but never widens it. For example: `input_guard.medical_advice`, `output_guard.self_harm`, `tool_gate.execute`, `tool_result_guard.<tool>`, `rag_passages.<tool>`, or `skill_load_check.<skill>`.
- **One snapshot per run.** Modes are read once when a run starts and kept for the whole run, including after a human-in-the-loop interrupt resumes.
- **Off means plain deepagents.** A blueprint that is off falls back to standard deepagents behavior. For example, the default model is used instead of the routed one, retrieval results pass through unchanged, and `MemoryMiddleware` keeps injecting all of `AGENTS.md`.
- **Dependencies.** `skill_pinning` needs `skill_suggestion`, and `escalation` needs `model_router`. A dependent you did not configure is dropped along with its dependency; one you configured explicitly raises an error.

## When Jev cannot answer

Blueprints listed in `JevConfig.fail_closed` refuse when Jev cannot be reached; every other blueprint lets the agent continue. The default list is `input_guard`, `output_guard`, and `tool_gate`. Every decision, error, and shadow disagreement goes to the `langchain_jev` logger and to `JevConfig.on_event`.

## Notes and limits

- **Not yet run against a live Jev server.** The unit tests use scripted decision and chat models. Questions are copied from the docs, and thresholds come from the cookbooks, which were measured with `jev-1.14.0`. Re-run your own evaluation sets after changing models.
- **Tool approval needs a checkpointer.** `tool_gate` pauses risky calls for approval in `HumanInTheLoopMiddleware`'s format. Without a checkpointer, the run stops at the request and the tool does not run. Set `ToolGateSettings(ask_human=False)` to refuse instead of pausing.
- **The output guard runs after generation.** With token streaming, the user may already have seen a reply before it is replaced.
- **Harness profiles are a beta, process-global API.** `register_jev_profile` relies on deepagents' `HarnessProfile`, which is beta and applies to every agent in the process. It does not reach `CompiledSubAgent` or `AsyncSubAgent` runnables.
- **Citations need a known format.** `citation_check` reads `[cite: <source id> | "<exact quote>"]`. Set `VerifySettings(add_citation_instructions=True)` to ask the model for that format.

## Development

```bash
make test
make lint
```
