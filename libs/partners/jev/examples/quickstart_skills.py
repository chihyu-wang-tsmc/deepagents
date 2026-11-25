"""用 anthropics/skills 測 skill_suggestion：Jev 能不能替每個請求挑對 skill。

skill_suggestion 在每次執行第一次呼叫模型前問 Jev，照 Typesafe 的 skill suggestion cookbook 問兩次：
    第 1 次：這個請求需不需要 skill？所有 skill 裡哪一個最像？（排出前 3 名）
    第 2 次：讀前 3 名的完整說明和 SKILL.md 的開頭再挑一次，並問每一個合不合（fits）
挑中的 skill 名稱會用 <skill_relevance> 提示交給模型；都不合時，也會告訴模型沒有合適的 skill。
這支程式也打開另外兩個藍圖：
    skill_pinning     Jev 很有把握時（fits 至少 0.8），直接把那個 SKILL.md 放進對話
    skill_load_check  模型自己去讀某個 SKILL.md 時，再問一次合不合，不合就在內容前加警告

每一題印出 Jev 挑的 skill、fits 和預期答案，最後統計對了幾題。
題目在下面的 CASES，可以換成自己的。

skills 放在 examples/skills/ 底下。anthropics/skills 下載到 examples/skills/anthropic/，
git 不追蹤這個資料夾（大約 13 MB），所以第一次要先在 libs/partners/jev/examples/ 執行：
    git clone --depth 1 https://github.com/anthropics/skills skills/anthropic
agent 從 /skills/ 讀 skills，這個目錄唯讀；agent 寫的其他檔案只放在記憶體，不會寫到磁碟。
啟動時 deepagents 會警告 claude-api 的說明超過 1024 字，只用前 1024 字，不影響執行。

執行前要設好的環境變數：
    H2O_BASE_URL   h2o-lightning shim 的網址（不設就用 http://localhost:8741）
可選：
    LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
        設了就用真的模型回答（OpenAI 相容 API，需要 langchain-openai），可以看模型有沒有
        照提示去讀 SKILL.md；每題最多呼叫模型 MAX_MODEL_CALLS 次（預設 3），免得一題做太久。
        沒設時用不花錢的替身模型：只有 Jev 會被呼叫，回覆寫出模型收到的提示。
    SKILLS_DIR     換一個 skills 目錄（每個 skill 一個資料夾，裡面有 SKILL.md），例如公司自己的
    JEV_MODEL      換 Jev 模型，例如 h2o-lightning-31b 或 typesafe
    JEV_MODES      例如 JEV_MODES=skill_pinning=off：只看建議，不把 SKILL.md 放進對話
    JEV_TRACE=1    每次問 Jev 都印出 Jev 讀到的內容、每題的鍵和回答、比較的門檻
    JEV_TRACE=full 再加上每題的原文和 Jev 讀到的完整內容
    JEV_TRACE_FILE=jev_trace.jsonl  每次請求和每個決定各存成一行 JSON
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import override

from langchain_jev import EnvModeSource, JevConfig, JevEvent, JevTrace, jev_middleware

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage

EXAMPLES = Path(__file__).resolve().parent
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR") or EXAMPLES / "skills" / "anthropic" / "skills")
MAX_MODEL_CALLS = int(os.environ.get("MAX_MODEL_CALLS") or 3)

if not SKILLS_DIR.is_dir():
    msg = (
        f"找不到 skills：{SKILLS_DIR}\n先在 {EXAMPLES} 執行：\n"
        "    git clone --depth 1 https://github.com/anthropics/skills skills/anthropic"
    )
    raise SystemExit(msg)


def linked_outside(folder: Path) -> list[Path]:
    """連到 folder 外面的 skill 資料夾（symlink）。

    deepagents 讀檔時不會跟著連結離開掛載的目錄，所以會略過這些 skill，而且不會警告。
    """
    root = folder.resolve()
    return [p for p in folder.iterdir() if p.is_symlink() and not p.resolve().is_relative_to(root)]


def warn_links(links: list[Path]) -> None:
    print(f"⚠ 這些 skill 是連到 {SKILLS_DIR} 外面的捷徑（symlink），deepagents 會略過：")
    print(f"    {', '.join(sorted(p.name for p in links))}")
    targets = {p.resolve().parent for p in links}
    if len(targets) == 1:
        print(f"  改用真正的資料夾：SKILLS_DIR={targets.pop()} python quickstart_skills.py")
    else:
        print("  把它們複製進來（cp -rL），或讓 SKILLS_DIR 指到真正的資料夾")


# 1. 題目：請求和預期的 skill；None 表示不需要 skill
class Case(NamedTuple):
    request: str
    expected: str | None


CASES = [
    Case("幫我做一個 Excel 預算表：每個月的收入、支出和結餘，最後一列用公式加總。", "xlsx"),
    Case("把這一季的業績重點做成 5 頁的 PowerPoint 簡報，每頁一個重點。", "pptx"),
    Case("把 report1.pdf 和 report2.pdf 合併成一個 PDF，再把第 3 頁轉成橫向。", "pdf"),
    Case("幫我做一份 Word（.docx）會議紀錄範本，要有標題、與會者和待辦事項的表格。", "docx"),
    Case("我想用 Python 寫一個 MCP server，讓 LLM 可以查我們的工單系統。", "mcp-builder"),
    Case(
        "用 Playwright 測試我本機 http://localhost:3000 的登入頁，截圖確認登入按鈕有出現。",
        "webapp-testing",
    ),
    Case("用 p5.js 做一個有隨機種子、可以調參數的流場（flow field）生成藝術。", "algorithmic-art"),
    Case("做一個給 Slack 用的小動畫 GIF：一顆會跳動的愛心。", "slack-gif-creator"),
    Case("幫我寫這週要發給全公司的週報公告，用公司慣用的格式。", "internal-comms"),
    Case("台灣最高的山是哪一座？", None),
    Case("把這句話翻成英文：今天天氣很好。", None),
]


# 2. 回答用的模型：有設 LLM_BASE_URL 就用真的，沒設就用替身
class StandIn(BaseChatModel):
    """不呼叫任何 API 的替身模型，只回覆它收到的 skill 提示。"""

    @property
    def _llm_type(self) -> str:
        return "stand-in"

    @override
    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> StandIn:
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        hint = next((m.text for m in reversed(messages) if "<skill_relevance>" in m.text), "")
        body = hint.replace("<skill_relevance>", "").replace("</skill_relevance>", "").strip()
        reply = AIMessage(content=f"（替身模型，沒有呼叫 API）收到的提示：{body or '沒有'}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def llm() -> BaseChatModel:
    if not os.environ.get("LLM_BASE_URL"):
        return StandIn()
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 - 只有用真的模型時才需要安裝

    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    )


# 3. skills 掛在 /skills/，直接讀磁碟但不能寫；agent 寫的其他檔案只放在記憶體
class Once(logging.Filter):
    """同一個警告只印一次：每一題都會重新載入 skills，警告也會重複。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        first = message not in self.seen
        self.seen.add(message)
        return first


logging.getLogger("deepagents.middleware.skills").addFilter(Once())
backend = CompositeBackend(
    default=StateBackend(),
    routes={"/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
)
read_only = FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")

# 4. 只開三個 skill 藍圖
events: list[JevEvent] = []
config = JevConfig(
    model="default",  # 讀 JEV_MODEL；沒設就是 h2o-lightning-4b
    blueprints={
        "skill_suggestion": "on",  # 挑 skill，提示模型
        "skill_pinning": "on",  # 很有把握時，直接把 SKILL.md 放進對話
        "skill_load_check": "on",  # 模型自己讀 SKILL.md 時再檢查一次
    },
    mode_source=EnvModeSource(),  # 讀 JEV_MODES，每次執行都會重讀
    on_event=events.append,
    # JEV_TRACE：印出每次問 Jev 的內容和回答；決定已經由 report 印出，所以 decisions=False
    trace=JevTrace.from_env(decisions=False),
)
agent = create_deep_agent(
    model=llm(),
    backend=backend,
    skills=["/skills/"],
    permissions=[read_only],
    # 直接傳 jev_middleware，skill_suggestion 才拿得到 backend、讀得到 SKILL.md 的開頭
    middleware=[*jev_middleware(config), ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS)],
    name="quickstart_skills",
)


# 5. 每一題印出 Jev 挑的 skill，和預期答案比對
def problem() -> str | None:
    """這一題 skill_suggestion 沒有挑出結果的原因；有結果時回傳 None。"""
    actions = {e.action: e.details for e in events if e.blueprint == "skill_suggestion"}
    if "error" in actions:  # Jev 回了錯誤、逾時或連不上
        return f"Jev 請求失敗：{actions['error']['error']}: {actions['error']['message']}"
    if actions.get("skip", {}).get("reason") == "no_skills":
        return "deepagents 沒有載入任何 skill（看上面的警告：SKILL.md 格式不對，或 skill 是捷徑）"
    if not any(action.startswith("suggest:") for action in actions):
        return "skill_suggestion 沒有執行，請確認 JEV_MODES 沒有把它關掉"
    return None


def suggestion() -> tuple[str | None, str]:
    """這一題 Jev 挑的 skill（None 是不需要 skill）和要印出來的說明。"""
    event = next(
        e for e in events if e.blueprint == "skill_suggestion" and e.action.startswith("suggest:")
    )
    name, fits = event.action.removeprefix("suggest:"), float(event.details.get("fits", 0.0))
    if name != "none":
        return name, f"{name}（fits {fits:.2f}）"
    if fits:
        return None, f"不需要 skill（最合的 fits {fits:.2f}，不到門檻）"
    return None, "不需要 skill（第 1 次請求就判斷不用）"


def report(case: Case, result: dict) -> bool | None:
    """印出這一題的結果；回傳挑得對不對，沒有結果時回傳 None。"""
    reason = problem()
    if reason is not None:
        print(f"    Jev 挑：沒有結果。{reason}")
        return None
    name, shown = suggestion()
    ok = name == case.expected
    print(f"    Jev 挑：{shown}  預期 {case.expected or '不需要 skill'} {'✓' if ok else '✗'}")
    for event in events:
        if event.blueprint == "skill_pinning":
            print(f"    放進對話：{event.action.removeprefix('pin:')} 的 SKILL.md")
        elif event.blueprint == "skill_load_check":
            skill, score = event.details["skill"], event.details["fits"]
            print(f"    模型讀了 {skill} 的 SKILL.md：{event.action}（fits {score:.2f}）")
    print(f"    回答：{str(result['messages'][-1].content)[:80]}")
    return ok


names = sorted(path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md"))
print(f"skills：{SKILLS_DIR}（{len(names)} 個）\n    {', '.join(names)}")
if links := linked_outside(SKILLS_DIR):
    warn_links(links)
outcomes: dict[int, bool | None] = {}
for number, case in enumerate(CASES, 1):
    events.clear()
    print(f"\n[{number}] {case.request}")
    result = agent.invoke({"messages": [{"role": "user", "content": case.request}]})
    outcomes[number] = report(case, result)

wrong = ", ".join(f"[{n}]" for n, ok in outcomes.items() if ok is False)
missing = ", ".join(f"[{n}]" for n, ok in outcomes.items() if ok is None)
print("\n=== skill_suggestion 對了幾題 ===")
print(f"    {sum(ok is True for ok in outcomes.values())}/{len(CASES)}")
if wrong:
    print(f"    挑錯的題目：{wrong}，用 JEV_TRACE=1 看每個 skill 的分數")
if missing:
    print(f"    沒有結果的題目：{missing}，原因印在每一題底下")
