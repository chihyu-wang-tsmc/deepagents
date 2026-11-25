"""照 Jev 挑的 skill 真的把檔案做出來：Excel、PowerPoint 存到 examples/outputs/。

quickstart_skills.py 只看 Jev 挑得對不對；這支讓 agent 照挑中的 skill 把事情做完：
    1. skill_suggestion 挑 skill，skill_pinning 把那個 SKILL.md 放進對話
    2. agent 寫腳本、在你的機器上執行，把檔案存到 examples/outputs/
    3. 每題做完，列出 outputs/ 裡新產生的檔案

⚠ 這支會在你的機器上執行指令。agent 用 deepagents 的 LocalShellBackend：指令直接在本機、
用你的帳號權限執行，沒有沙箱。所以：
    每個指令執行前都會停下來問你：[y] 執行  [n] 不執行  [a] 這一題之後都執行
    agent 的檔案工具只碰得到 examples/outputs/（它的工作區）和唯讀的 skills；
    但指令本身不受這個限制，核准前請看清楚
    傳給指令的環境變數只有 PATH、HOME 這類，不含 LLM_API_KEY 之類的機密
examples/outputs/ 不會進 git。
LocalShellBackend 要照 deepagents 原本的寫法透過 shell 執行指令；deepagents 的
backends/local_shell.py 若改成 shell=False，帶參數的指令都會失敗，程式會在那一題提醒。

skill 需要的工具：anthropics/skills 假設都已經裝好，在你的機器上要自己裝。
    Excel       pip install openpyxl pandas
                重算公式要 LibreOffice（soffice），沒有也能產生檔案
    PowerPoint  Node.js 加 pptxgenjs（skill 的主要做法），或 pip install python-pptx
啟動時會檢查並列出缺的；agent 也可能要求安裝（一樣要你核准）。

skills 的位置跟 quickstart_skills.py 一樣：examples/skills/anthropic/，或用 SKILLS_DIR 指定；
skill 資料夾一樣不能是連到外面的捷徑（symlink）。

執行前要設好的環境變數：
    LLM_BASE_URL、LLM_API_KEY、LLM_MODEL   真的模型（OpenAI 相容 API，需要 langchain-openai）
    H2O_BASE_URL   h2o-lightning shim 的網址（不設就用 http://localhost:8741）
可選：
    MAX_MODEL_CALLS  每題最多呼叫模型幾次（預設 30）
    SKILLS_DIR、JEV_MODEL、JEV_MODES、JEV_TRACE  同 quickstart_skills.py

用法：
    python quickstart_skills_run.py                      # 跑下面兩題：Excel、PowerPoint
    python quickstart_skills_run.py "幫我做一個 ..."      # 只跑你自己的一題
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    FileUploadResponse,
    WriteResult,
)
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import override

from langchain_jev import EnvModeSource, JevConfig, JevEvent, JevTrace, jev_middleware

EXAMPLES = Path(__file__).resolve().parent
OUTPUTS = EXAMPLES / "outputs"
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR") or EXAMPLES / "skills" / "anthropic" / "skills")
MAX_MODEL_CALLS = int(os.environ.get("MAX_MODEL_CALLS") or 30)
SHELL_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV", "CONDA_PREFIX", "NODE_PATH")

if not SKILLS_DIR.is_dir():
    msg = (
        f"找不到 skills：{SKILLS_DIR}\n先在 {EXAMPLES} 執行：\n"
        "    git clone --depth 1 https://github.com/anthropics/skills skills/anthropic"
    )
    raise SystemExit(msg)
if not os.environ.get("LLM_BASE_URL"):
    msg = "這支要用真的模型才做得出檔案：請設定 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL"
    raise SystemExit(msg)
OUTPUTS.mkdir(exist_ok=True)


# 1. 題目：請求和預期的 skill；命令列給一句話時，只跑那一句
class Case(NamedTuple):
    request: str
    expected: str | None


CASES = [
    Case(
        "幫我做一個 Excel 預算表 budget.xlsx：1 到 6 月的收入分別是 120、135、128、150、160、158"
        "（萬元），支出是 80、90、85、95、100、98，算出每個月的結餘，最後一列用公式加總。",
        "xlsx",
    ),
    Case(
        "幫我做一份 5 頁的 PowerPoint 簡報 q3_review.pptx，介紹第三季業績：營收成長 12%、"
        "毛利率 48%、新客戶 35 家、主要風險是匯率，最後一頁是第四季目標。",
        "pptx",
    ),
]
if len(sys.argv) > 1:
    CASES = [Case(" ".join(sys.argv[1:]), None)]


# 2. 檢查 skill 會用到的工具：Python 套件用目前的 Python 檢查，其他看 PATH 上有沒有
def check_tools() -> None:
    packages = {"openpyxl": "openpyxl", "pandas": "pandas", "python-pptx": "pptx"}
    found = {
        name: importlib.util.find_spec(module) is not None for name, module in packages.items()
    }
    found.update({name: shutil.which(name) is not None for name in ("node", "npm", "soffice")})
    marks = "  ".join(f"{'✓' if ok else '✗'} {name}" for name, ok in found.items())
    print(f"skill 會用到的工具：{marks}")
    if not all(found.values()):
        print("    ✗ 是缺的：agent 可能會要求安裝（要你核准），或改用別的做法")


# 3. agent 的工作區是 outputs/：檔案工具的 / 就是 outputs/，指令也在那裡執行；skills 唯讀
class ReadOnly(FilesystemBackend):
    """只能讀的資料夾：寫入、修改、刪除都回錯誤，agent 改不到 skills。"""

    @override
    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=f"{file_path}: skills are read-only")

    @override
    def edit(self, file_path: str, *args: object, **kwargs: object) -> EditResult:
        return EditResult(error=f"{file_path}: skills are read-only")

    @override
    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=f"{file_path}: skills are read-only")

    @override
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


backend = CompositeBackend(
    default=LocalShellBackend(
        root_dir=OUTPUTS,
        virtual_mode=True,  # 檔案工具只碰得到 outputs/
        env={name: os.environ[name] for name in SHELL_ENV if name in os.environ},
    ),
    routes={"/skills/": ReadOnly(root_dir=SKILLS_DIR, virtual_mode=True)},
)
SYSTEM_PROMPT = f"""You work on the user's own machine, inside a workspace folder.
- In file tools the workspace is `/`. Shell commands run inside it, where its real path is
  {OUTPUTS}. Save every file you make in the workspace.
- Skills are read-only. File tools see them under /skills/; shell commands see them at
  {SKILLS_DIR}, so run a skill's scripts by that path.
- The user approves each shell command before it runs, so keep commands few and say what
  each one does.
- If a package a skill needs is missing, say which one; install it only if the user approves.
- When you are done, list the files you made."""

# 4. Jev：跟 quickstart_skills.py 一樣只開三個 skill 藍圖
events: list[JevEvent] = []
config = JevConfig(
    model="default",  # 讀 JEV_MODEL；沒設就是 h2o-lightning-4b
    blueprints={"skill_suggestion": "on", "skill_pinning": "on", "skill_load_check": "on"},
    mode_source=EnvModeSource(),
    on_event=events.append,
    # JEV_TRACE：印出每次問 Jev 的內容和回答；決定已經由 show_pick 印出，所以 decisions=False
    trace=JevTrace.from_env(decisions=False),
)
agent = create_deep_agent(
    model=ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
    ),
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
    skills=["/skills/"],
    interrupt_on={"execute": True},  # 每個指令執行前停下來，等你核准
    # 直接傳 jev_middleware，skill_suggestion 才拿得到 backend、讀得到 SKILL.md 的開頭
    middleware=[*jev_middleware(config), ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS)],
    checkpointer=InMemorySaver(),  # 暫停等核准時要記住進度
    name="quickstart_skills_run",
)


# 5. 每一題：印出 Jev 挑的 skill，逐一核准指令，最後列出產生的檔案
def show_pick(case: Case) -> None:
    mine = {e.action: e.details for e in events if e.blueprint == "skill_suggestion"}
    if "error" in mine:
        print(f"    Jev 挑：沒有結果。Jev 請求失敗：{mine['error']['message']}")
        return
    name = next((a.removeprefix("suggest:") for a in mine if a.startswith("suggest:")), None)
    if name is None:
        print("    Jev 挑：沒有結果（沒有載入任何 skill，或 skill_suggestion 被關掉了）")
        return
    fits = float(mine[f"suggest:{name}"].get("fits", 0.0))
    mark = (
        f"  預期 {case.expected} {'✓' if name == case.expected else '✗'}" if case.expected else ""
    )
    shown = "不需要 skill" if name == "none" else f"{name}（fits {fits:.2f}）"
    print(f"    Jev 挑：{shown}{mark}")
    for event in events:
        if event.blueprint == "skill_pinning":
            print(f"    放進對話：{event.action.removeprefix('pin:')} 的 SKILL.md")


def decide(request: dict, *, approve_all: bool) -> tuple[dict, bool]:
    """問你要不要執行這個指令；回傳決定，以及這一題之後是不是都執行。"""
    print(f"    ▶ 要執行的指令：{request['args'].get('command', request['args'])}")
    if approve_all:
        return {"type": "approve"}, True
    try:
        answer = input("      執行？[y] 是  [n] 否  [a] 這一題之後都執行：").strip().lower()
    except EOFError:  # 沒有可以輸入的終端機時，一律不執行
        answer = "n"
    if answer in {"y", "a"}:
        return {"type": "approve"}, answer == "a"
    return {"type": "reject", "message": "使用者沒有核准這個指令。"}, False


def run(case: Case, thread: dict) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": case.request}]}, thread)
    show_pick(case)  # Jev 在第一次呼叫模型前就挑好了
    approve_all = False
    while pending := result.get("__interrupt__"):
        resume = {}
        for interrupt in pending:
            decisions = []
            for request in interrupt.value["action_requests"]:
                decision, approve_all = decide(request, approve_all=approve_all)
                decisions.append(decision)
            resume[interrupt.id] = {"decisions": decisions}
        result = agent.invoke(Command(resume=resume), thread)
    return result


def snapshot() -> dict[Path, int]:
    files = (p for p in OUTPUTS.rglob("*") if p.is_file() and "node_modules" not in p.parts)
    return {p: p.stat().st_mtime_ns for p in files if p.name != ".gitignore"}


def show_files(before: dict[Path, int]) -> None:
    made = sorted(p for p, stamp in snapshot().items() if before.get(p) != stamp)
    print("    產生的檔案：" + ("" if made else "沒有"))
    for path in made:
        print(f"      {path.relative_to(EXAMPLES)}（{path.stat().st_size:,} bytes）")


def check_shell(result: dict) -> None:
    """指令整串被當成一個程式名稱時，說明是 LocalShellBackend 沒有透過 shell 啟動指令。"""
    outputs = [str(m.content) for m in result["messages"] if getattr(m, "name", "") == "execute"]
    if any("FileNotFoundError" in text and "No such file or directory" in text for text in outputs):
        print("    ⚠ 指令沒有執行：這份 deepagents 的 LocalShellBackend 沒有透過 shell 啟動指令")
        print("      （deepagents/backends/local_shell.py 的 shell=False），帶參數的指令都會失敗。")
        print("      deepagents 原本是 shell=True；要不要改回來，請依你們的資安規定決定。")


check_tools()
print(f"skills：{SKILLS_DIR}\n輸出：{OUTPUTS}")
for number, case in enumerate(CASES, 1):
    events.clear()
    print(f"\n[{number}] {case.request}")
    before = snapshot()
    result = run(case, {"configurable": {"thread_id": f"skills-run-{number}"}})
    show_files(before)
    check_shell(result)
    print(f"    回答：{str(result['messages'][-1].content)[:300]}")
