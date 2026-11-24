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

## Custom blueprints

A `JevBlueprint` adds a blueprint of your own. It holds its questions, the stage where they are asked (the front), and the actions to take, each at its own position (the back). The stack asks the questions at the stage, hands the answers back under the ids you gave them, and carries out every action where it says, so what is asked and what is acted on always match.

```python
from langchain_jev import JevBlueprint, JevConfig, Refuse, SystemNote, noul, register_jev_profile

personal_data = JevBlueprint(
    name="personal_data",
    stage="tool_result",  # the front: ask about each result of the tools below
    tools=["read_ticket"],
    questions={
        "has_personal_data": noul(
            "Does `tool_output` contain personal data such as phone numbers?",
            yes="It contains personal data.",
            no="It contains no personal data.",
        )
    },
    when={"has_personal_data": 0.7},  # a yes/no answer at or above its threshold triggers `act`
    act=[
        # The backs: from then until the run ends, refuse external email and warn the model.
        Refuse("No email after reading personal data.", at="tool_call", tools=["send_email"]),
        SystemNote("Do not repeat personal data to the user.", at="model_call"),
    ],
)
register_jev_profile(
    ["openai"],
    JevConfig(blueprints={"personal_data": "on"}, custom_blueprints=[personal_data]),
)
```

| Stage (front) | Hook | What Jev reads | Built-ins at the same stage |
| --- | --- | --- | --- |
| `entry` | `before_agent`, inside the entry request | the user's message and earlier turns | `input_guard`, the routers, `chat_mode` |
| `model_output` | `after_model` | the text of each model reply | `output_guard` |
| `tool_call` | `wrap_tool_call`, before the tool runs | the call and the recent messages | `tool_gate` |
| `tool_result` | `wrap_tool_call`, after the tool returns | the tool's result | `tool_result_guard` |
| `final_answer` | `after_agent` | the request and the final answer | the answer checks |

| Position (`at`, back) | Hook | Actions; the last ones in each row run your own function |
| --- | --- | --- |
| `before_agent` | `before_agent`, from `entry` only | `Refuse` |
| `model_call` | `wrap_model_call` | `SystemNote`, `UseModel`, `NoTools`, `DropTools`, `ShapeModelCall` |
| `before_model` | `before_model` | `Refuse` |
| `model_output` | `after_model` | `Refuse`, `RewriteReply` |
| `tool_call` | `wrap_tool_call`, before the tool runs | `Refuse`, `AskHuman`, `ShapeToolCall` |
| `tool_result` | `wrap_tool_call`, after the tool returns | `Warn`, `Redact`, `Refuse`, `RewriteToolResult` |
| `final_answer` | `after_agent` | `Mark`, `RewriteReply`, `Notify` |

- **Where an action acts.** Without `at`, an action acts at its usual position for the stage: the stage's own when it can, such as `Refuse` at `entry` (in `before_agent`), otherwise the only one it can reach, such as `SystemNote` at `model_call`. An action that can reach several, such as `RewriteReply` from `entry`, needs `at`. An action at the stage's own position acts on what Jev was just asked about. An action at a later position acts from then until the run ends, not into the next run. A position before the stage, such as `model_call` for a `final_answer` blueprint, raises when the blueprint is defined.
- **Tools at the tool positions.** An action at `tool_call` or `tool_result` acts on the tools in its own `tools`. It needs them unless it acts on the blueprint's own tools: a `tool_result` blueprint's own results, or a `tool_call` blueprint's calls and their results.
- **How the back knows.** Entry answers stay in the entry request's record. Answers from the other stages that trigger a later action are recorded on the message Jev was asked about, in `additional_kwargs["jev_answers"]`, so they show in traces. The back decides again from those answers, so it always agrees with the front.
- **One entry request.** Entry questions join the request the built-in entry blueprints already send, so they add no request.
- **The built-in switches.** A custom blueprint is off until `blueprints` turns it on by name. `JEV_MODES` and the run's `context` switch it, and at the tool positions a part narrows it to one tool, such as `external_email.send_email`. Reword `<name>.<question id>` in `questions`, tune `<name>.<question id>` (with `when`) or `<name>.<threshold>` (with `decide`) in `thresholds`, give it a model in `models`, and list it in `fail_closed`. In shadow mode it reports what it would do and where, and changes nothing.
- **`decide` for everything else.** Instead of `when` and `act`, `decide(answers, thresholds)` reads Choice and Score answers or combines questions, and returns an action, a list of actions, or `None`. `acts_at` lists the positions its actions may use, so the stack builds them; without it they may use the stage's usual ones (`before_agent` and `model_call` for `entry`, the stage's own position otherwise). An action elsewhere raises. Name the values it compares in `thresholds={...}` so they can be tuned without code changes.
- **Your own functions.** `ShapeModelCall` gets the `ModelRequest` and returns it changed. `ShapeToolCall` gets the call and returns its arguments. `RewriteToolResult` and `RewriteReply` get text and return text. `Notify(answers, final_text)` runs once when the run ends, even after a refusal. `decide` and every function but `Notify` may run more than once a run, so keep them free of side effects. A function that raises, or returns the wrong kind, fails the run.
- **Events.** The front reports its decision, such as `refuse+note`, with each answer and the positions in `details["at"]`. A back reports what it did, such as `refuse@tool_call` with the tool. Backs at `model_call` act on every model call and do not report each one.
- **Built-ins act first.** At each position a custom blueprint acts after the built-in ones. A refusal from `input_guard`, `output_guard`, or `tool_gate` comes first, and a result `tool_result_guard` withheld is not asked about again. Custom blueprints act in the order they are listed. A model picked by `UseModel` wins over the routers.
- **When Jev cannot answer.** A custom blueprint in `fail_closed` refuses at `entry` and `model_output`, does not run the call at `tool_call`, withholds the result at `tool_result`, and marks the answer unchecked at `final_answer`. Otherwise the agent continues.
- **Marks.** `Mark` adds the blueprint's name and reason to `additional_kwargs["jev_checks"]`, beside the built-in checks. With `VerifySettings(annotate=True)`, one note at the end of the reply names every failed check, and each check reads the answer as the model wrote it.
- **Checked when defined.** A bad name, an unknown stage or position, a `when` naming a Choice or Score, an action that cannot act at its position, or a tool position without tools raises when the `JevBlueprint` is built. A name the package already uses raises when the config is built.
- **Limits.** Entry questions are asked about the user's message, because the entry request shares one state; ask about other text at another stage. The state each stage sends Jev is fixed. `default_questions(custom=[...])` lists custom questions with the built-in ones.

[`examples/quickstart_customer_jev.py`](examples/quickstart_customer_jev.py) uses every stage and several later positions.

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

## Seeing what Jev was asked

To tune a question, look at what Jev read, how it answered, and which threshold the answer was compared with. `JevConfig(trace=JevTrace())` prints a block for every request the stack sends to Jev, and a line for every decision:

```text
[jev request] entry: input_guard, chat_mode · h2o-lightning-4b
  read: message = "Delete /tmp/a.txt"
  input_guard.jailbreak                yes 0.12
  input_guard.drop_safeguards          yes 0.00
  input_guard.harmful_request          yes 0.00
  input_guard.severity                 score 0 of 3
  chat_mode.mode                       agent 0.90 | chat 0.10
  thresholds: input_guard.review 0.35, input_guard.action 0.7, input_guard.severity_block 2, chat_mode.min_probability 0.8
[jev decision] input_guard on pass
[jev decision] chat_mode on agent
```

- **What the lines are.** `read` is the state the blueprint sent Jev, clipped. Each question line starts with the key `JevConfig.questions` rewords it by, then gives the answer: the probability of yes for a Noul, each option's probability for a Choice, and the level for a Score. `thresholds` lists the values the blueprints compare those answers with, after the guard policy, the model's own thresholds, and `JevConfig.thresholds`.
- **More detail.** `JevTrace(full=True)` also prints each question's wording and the whole state.
- **A file to work through.** `JevTrace(path="jev_trace.jsonl")` appends one JSON object per request (`"type": "request"`, with the state, questions, answers, thresholds, model, `cached`, and `error`) and one per decision (`"type": "decision"`). Run a batch of your own inputs, then compare the answers before you change a question or a threshold.
- **Without code changes.** `JevTrace.from_env()` reads `JEV_TRACE=1` (short), `JEV_TRACE=full`, and `JEV_TRACE_FILE`, and returns `None` when none is set. Every quickstart passes it, so `JEV_TRACE=1 python examples/quickstart.py` shows each request.
- **Every request counts.** Answers served from the cache are marked `cached`, and a failed request shows why it failed. A trace that raises is logged and never stops the run. Subagents that share the stack's runtime, as `register_jev_profile` sets up, are traced too.
- **Decisions printed elsewhere.** If `on_event` already prints the decisions, `JevTrace(decisions=False)` leaves them out of the printout; the file keeps them.

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
- **The tool gate checks every tool.** `tool_gate` asks Jev before each tool call, including your own tools, MCP tools, and the deepagents `delete` tool. When Jev is attached with `register_jev_profile`, that includes the tools subagents call; `middleware=jev_middleware(config)` only covers the main agent. It skips only the tools in `ToolGateSettings.skip`, which by default are `ls`, `read_file`, `glob`, and `grep`, since they only read local files. Each checked call waits for one Jev request. Set `skip=()` to check every call, or `tools=[...]` to check only the tools you list. With `ToolGateSettings.policy` set, every checked call is compared with the policy, so the policy should grant each operation the agent may do, not only the risky ones.
- **Tool approval needs a checkpointer.** `tool_gate` pauses risky calls for approval in `HumanInTheLoopMiddleware`'s format. Without a checkpointer, the run stops at the request and the tool does not run. Set `ToolGateSettings(ask_human=False)` to refuse instead of pausing. A call that would leak data, that follows instructions from a tool result, or that falls outside `ToolGateSettings.policy` is refused without asking. The leak check counts credentials, personal data, private files, and the company's internal data sent outside the company as a leak, even when the user asked or the address is the user's own; name the company's domains in the policy so Jev can tell which addresses are inside.
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
