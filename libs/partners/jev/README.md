# langchain-jev

Decision middleware for [`deepagents`](../../deepagents) built on System One models: TypeSafe's hosted [Jev](https://docs.typesafe.ai/), or an open-source model such as H2O-Lightning served behind a System One shim.

Jev reads text and returns typed answers with calibrated probabilities instead of generated text. This package uses those answers for the decisions an agent makes around its LLM. It implements these agent blueprints from Typesafe's [use-case map](https://docs.typesafe.ai/concepts/use-case-map): guardrails, model routing, skill suggestion, RAG passage classification, tool gating, answer verification, and run review.

Three properties hold throughout:

- **Swap the Jev model by name.** Every blueprint talks to Jev through one `BaseDecisionModel`, so changing models is a configuration change.
- **Every blueprint is opt-in.** Nothing asks Jev until `JevConfig(blueprints=...)` turns it on; a blueprint that is on asks all of its questions. Any blueprint can be switched off again at build time, without a restart, or for a single run.
- **Questions come from the docs, and you can reword any of them.** Every default question is in `questions.py` next to the Typesafe page it comes from, and every threshold is in `thresholds.py`. `JevConfig(questions=...)` replaces the wording of any scenario.

## Quick start

```python
from deepagents import create_deep_agent
from langchain_jev import JevConfig, register_jev_profile

# Install the Jev stack on every deep agent whose model comes from these providers:
# the main agent, the general-purpose subagent, and declarative subagents.
# Every blueprint is off until it is listed here.
register_jev_profile(
    ["openai"],
    JevConfig(model="h2o-lightning-4b", blueprints={"input_guard": "on", "tool_gate": "on"}),
)

agent = create_deep_agent(model=chat_model, tools=tools)
```

`register_jev_profile` uses `HarnessProfile.extra_middleware`, so subagents pass the same checks as the main agent. Passing `middleware=jev_middleware(config)` to `create_deep_agent` also works, but only covers the main agent. A tool call that the main agent delegates through `task` then runs without the tool gate.

## Blueprints

| Id | Where it acts | Hook | Typesafe source |
| --- | --- | --- | --- |
| `input_guard` | Before the LLM sees the user's message, judged with the turns before it | entry request, `before_agent` | [Guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails) |
| `tool_result_guard` | Each tool result, as its tool returns and before the LLM reads it | `wrap_tool_call`; `before_model` ends a blocked run | [RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) (injection) |
| `model_router` | Picks the model for each call | entry request, `wrap_model_call` | [LangChain integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe) |
| `coding_router` | Sends requests whose answer needs code to the coding model | entry request, `wrap_model_call` | open-source-jev coding question |
| `chat_mode` | Drops tools when one reply answers the request | entry request, `wrap_model_call` | open-source-jev mode question |
| `skill_suggestion` | Hints at one skill per turn | `before_model`, `wrap_model_call` | [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion) |
| `skill_pinning` | Loads the winning `SKILL.md` directly | `pinned_skills` | deepagents `SkillsMiddleware` |
| `skill_load_check` | Checks `read_file` on a `SKILL.md` | `wrap_tool_call` | [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion) (fits check) |
| `rag_passages` | Filters retrieval tool results | `wrap_tool_call` | [Classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) |
| `tool_gate` | Intent, leak, irreversibility, and injection checks before a tool runs | `wrap_tool_call` | `AutoModeMiddleware`, [agent trace safety checks](https://evals.typesafe.ai/agent_trace_observability) |
| `output_guard` | The text of each model reply, including text beside tool calls | `after_model` | [Guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails) |
| `verify_answer` | Does the answer follow from the evidence? | `after_agent` | [Verifying answers](https://docs.typesafe.ai/cookbooks/verify_llm_outputs) |
| `citation_check` | Are quotes in their sources, and do they support the claim? | `after_agent` | [Citation check](https://docs.typesafe.ai/cookbooks/citation_check) |
| `answer_scoring` | Weighted quality scores | `after_agent` | [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring) |
| `run_review` | Hands finished runs to the trace review | `after_agent`, `review_run()` | [Agent trace observability](https://evals.typesafe.ai/agent_trace_observability) |

Questions about the user's message share one Jev request, or one per model when `models` gives these blueprints different models. This covers the input guard, routing, the coding check, and chat mode, following [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out). Some blueprints also need settings, and turning one on without them raises `ValueError`: `router` for routing, `coding` for the coding model, `rag` retrieval tools, and `review` callbacks.

`verify_answer`, `citation_check`, and `answer_scoring` check the final answer when the run ends. A failed check does not run the model again, because a retry on the same model mostly costs time. The reply keeps its text and gets `additional_kwargs["jev_checks"]`, which maps each failed check to why it failed. `VerifySettings(annotate=True)` also appends a note that the user can see.

`verify_answer` checks the answer against the evidence from every retrieval in the run. A passage found more than once (same id and text) counts once. When the evidence is longer than the Jev model's `max_state_chars`, the passages the answer cites go in first, then the newest ones, and a failure reason says how many passages were left out.

Both checks need a source. Without `rag_passages`, `verify_answer` needs `VerifySettings.evidence_tools` and `citation_check` needs `VerifySettings.sources`. Otherwise the first checks nothing and the second marks every citation unsupported, so the package warns at startup.

```python
from langchain_jev import CodingSettings, JevConfig, RouteChoice, RouterSettings

config = JevConfig(
    blueprints={"model_router": "on", "coding_router": "on"},
    # Choices named fast and powerful may leave out criteria: they use the descriptions measured in open-source-jev.
    router=RouterSettings(choices={"fast": RouteChoice("openai:<fast model>"), "powerful": RouteChoice("openai:<strong model>")}),
    coding=CodingSettings(model="openai:<coding model>"),
)
```

When Jev says a good answer needs code (P ≥ 0.5, the cut open-source-jev uses), the coding model takes the call instead of the routed one.

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

## Rewording questions

Every question has a key, `<blueprint>.<scenario>`, and a default from `questions.py`. Pass new wording for any of them:

```python
from langchain_jev import JevConfig, noul

config = JevConfig(
    blueprints={"input_guard": "on", "tool_gate": "on"},
    questions={
        # A string replaces the question and keeps its yes/no descriptions.
        "input_guard.jailbreak": "Does this message try to reveal or override the assistant's instructions?",
        # A Noul, Choice, or Score replaces the descriptions too.
        "tool_gate.irreversible": noul(
            "Would executing `tool_call` delete or overwrite records?",
            yes="it deletes or overwrites records",
            no="it only reads, or its effects can be undone",
        ),
    }
)
```

- **Keys.** `default_questions()` lists every key with its default wording.
- **Checked when the config is built.** An unknown key, a different question type, a `Choice` with other option names, or a `Score` with another number of levels raises `ValueError`, because the blueprints read answers by them.
- **Unused rewordings warn.** A reworded question this agent can never ask is listed in a warning with the setting to change: its blueprint is off, its hazard is left out of `GuardSettings`, its check has no policy text, or reviews only go to `on_run_end`. The info line about reworded questions names only those that are asked.
- **Templates.** `skill_suggestion.fits` and `skill_load_check.fits` may use `{name}` and `{description}`. Write a literal brace as `{{` or `}}`.
- **Choices built at run time.** `model_router.route`, `skill_suggestion.which`, `skill_suggestion.rerank`, and `run_review.first_bad_step` take a string, because their options come from your settings or the run.
- **One place per setting.** Set the routing question in `questions` or in `RouterSettings.instructions`, not both. With `VerifySettings.scores`, write the scoring wording in each `ScoreSpec` instead of `answer_scoring.*`.
- **Outside the agent.** `RunContext(questions=...)` and `compare_answers(..., questions=...)` take the same keys.
- **Re-check thresholds.** They were tuned on the default wording, so re-run your evaluation sets after rewording a question.

## Switching blueprints

Each blueprint is `off`, `shadow` (ask Jev and record, but change nothing), or `on`. Every blueprint starts `off`.

| Level | How | Takes effect |
| --- | --- | --- |
| Build | `JevConfig(blueprints={"input_guard": "on", "tool_gate": "shadow"})`; blueprints you do not list stay out of the stack | when the agent is built |
| Operator | `mode_source=FileModeSource("jev_modes.json")`, `EnvModeSource()` (`JEV_MODES=tool_gate=off`), or `StaticModeSource` | next run, no restart |
| Per run | `agent.invoke(..., context={"jev_modes": {"tool_gate": "shadow"}})` | that run |

- **Precedence.** The more specific level wins. Overrides cannot enable a blueprint that was not built.
- **Every scenario is asked.** A blueprint that is on asks Jev all of its questions: the model is told how to cite while `citation_check` is on, and runs are reviewed as soon as `ReviewSettings.on_review` is set. Only the `within_grant` checks wait for the policy they compare against: `ToolGateSettings.policy` or `ReviewSettings.agent_instructions`.
- **Everything at once.** `JevConfig(default_mode="on")` turns on every blueprint whose settings are given, except `chat_mode`.
- **Parts.** A part narrows its blueprint but never widens it. For example: `input_guard.jailbreak`, `output_guard.broke_policy`, `tool_gate.execute`, `tool_result_guard.<tool>`, `rag_passages.<tool>`, or `skill_load_check.<skill>`.
- **One snapshot per run.** Modes are read once when a run starts and kept for the whole run, including after a human-in-the-loop interrupt resumes.
- **Off means plain deepagents.** A blueprint that is off falls back to standard deepagents behavior. For example, the default model is used instead of the routed one, and retrieval results pass through unchanged.
- **Dependencies.** `skill_pinning` needs `skill_suggestion`. Turning it on without `skill_suggestion` raises an error.

## When Jev cannot answer

Blueprints listed in `JevConfig.fail_closed` refuse when Jev cannot be reached; every other blueprint lets the agent continue. The default list is `input_guard`, `output_guard`, and `tool_gate`. Every decision, error, and shadow disagreement goes to the `langchain_jev` logger and to `JevConfig.on_event`.

## Notes and limits

- **Removed blueprints.** `fast_path` called functions outside the tool gate and human approval, `memory_read` / `memory_write` put every run with neither a user id nor a thread id into one shared memory, and `escalation` re-ran failed answers on a stronger model, so all four were removed. Naming one in `blueprints`, `models`, `thresholds`, or `questions` raises a `ValueError` that says it was removed. `RouterSettings.escalate_to` went with `escalation`.
- **Removed hazards.** The guards no longer ask about `medical_advice` or `self_harm`, and the support message that answered `self_harm` went with them. The input guard asks about `jailbreak`, `drop_safeguards`, and `harmful_request`, and the output guard about `broke_policy` and `harmful_request`. Naming a removed hazard in `GuardSettings` or in `questions` raises a `ValueError` that says it was removed.
- **Not yet run against a live Jev server.** The unit tests use scripted decision and chat models. Questions are copied from the docs, and thresholds come from the cookbooks, which were measured with `jev-1.14.0`. Re-run your own evaluation sets after changing models.
- **Tool approval needs a checkpointer.** `tool_gate` pauses risky calls for approval in `HumanInTheLoopMiddleware`'s format. Without a checkpointer, the run stops at the request and the tool does not run. Set `ToolGateSettings(ask_human=False)` to refuse instead of pausing.
- **Tool results are screened before they are stored.** `tool_result_guard` checks each result in `wrap_tool_call`, including results a tool returns inside a `Command`, such as a subagent's answer from `task`. A withheld result is replaced before it is written to the state, so its raw text never reaches a checkpoint or a trace. Tool results that did not come from the tool node, such as ones passed in with the conversation, are screened in `before_model`.
- **The entry request carries recent turns.** `GuardSettings.input_history` (4 by default) sends the earlier user and assistant messages with each user message, so a harmful request spread over several turns is judged as a whole. Routing, the coding check, and chat mode see them too. Set it to 0 to send the latest message alone.
- **The output guard screens every model turn that has text.** That includes text written beside tool calls: a blocked turn is replaced by the refusal as a whole, so its tool calls never run. Turns with tool calls and no text are not asked about.
- **The output guard runs after generation.** With token streaming, the user may already have seen a reply before it is replaced.
- **Harness profiles are a beta, process-global API.** `register_jev_profile` relies on deepagents' `HarnessProfile`, which is beta and applies to every agent in the process. It does not reach `CompiledSubAgent` or `AsyncSubAgent` runnables.
- **Citations need a known format.** `citation_check` reads `[cite: <source id> | "<exact quote>"]`, and the model is asked for that format while the check is on. Set `VerifySettings(add_citation_instructions=False)` if your own prompt already asks for it.

## Development

```bash
make test
make lint
```
