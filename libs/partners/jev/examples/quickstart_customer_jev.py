"""客製化 Jev 流程：用 JevBlueprint 加上自己的藍圖。

內建藍圖（input_guard、tool_gate…）的題目和動作是 langchain-jev 寫好的。
自訂藍圖由你決定三件事，題目和動作寫在同一個 JevBlueprint 裡，前後由 library 接好：
    1. 在哪裡問（stage）：
         entry         使用者訊息進來時，跟 input_guard 等內建藍圖併在同一個 Jev 請求
         model_output  每一輪模型回覆的文字
         tool_call     工具執行前（只看 tools 列出的工具）
         tool_result   工具回傳後、模型讀到之前（只看 tools 列出的工具）
         final_answer  執行結束時的最後回答
    2. 問什麼（questions）：noul(...) 是非題、Choice 選擇題、Score 分數題
    3. 答案怎麼處理：when＋act（是非題的機率到門檻就動作），或 decide（自己寫判斷）

每個位置能用的動作：
    entry         Refuse（拒答並結束）、SystemNote（加系統提示）、
                  UseModel（換模型）、NoTools（不給工具）
    model_output  Refuse（把回覆換成拒答並結束）
    tool_call     Refuse（不執行這次呼叫）、AskHuman（等人核准，要有 checkpointer）
    tool_result   Warn（保留結果，前面加註）、Redact（遮蔽結果）、Refuse（結束這次執行）
    final_answer  Mark（標記在回覆的 additional_kwargs["jev_checks"]）

執行前要設好的環境變數：
    LLM_BASE_URL   OpenAI 相容 API 的網址，例如 https://your-gateway.example.com/v1
    LLM_API_KEY    該 API 的 token（本機 vLLM 不檢查的話隨便填）
    LLM_MODEL      gateway 上的模型名稱
    H2O_BASE_URL   h2o-lightning shim 的網址（不設就用 http://localhost:8741）
可選：
    JEV_MODEL      換 Jev 模型，例如 h2o-lightning-31b 或 typesafe
    JEV_MODES      不改程式就切換藍圖，自訂藍圖也可以，例如 "confidential_request=shadow"
                   工具位置可以只關一個工具，例如 "external_email.send_email=off"
    JEV_LIST_QUESTIONS=1  印出每個可以改寫的 Jev 問題（包含自訂藍圖的）後結束
"""

import os
import sys
from collections.abc import Mapping

from deepagents import create_deep_agent
from langchain_core.decisions import Choice
from langchain_core.decisions.types import Answer
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_jev import (
    Action,
    EnvModeSource,
    GuardSettings,
    JevBlueprint,
    JevConfig,
    JevEvent,
    Mark,
    Redact,
    Refuse,
    SystemNote,
    ToolGateSettings,
    default_questions,
    noul,
    register_jev_profile,
)

# 1. 自訂藍圖
#    - name 是藍圖的名字：開關、改措辭、改門檻、換 Jev 模型都用這個名字
#    - questions 的鍵是自己取的題目 id；when、decide 拿答案也用同一個 id
#    - 題目用英文寫：內建題目的門檻都是用英文題目量出來的，改成中文請用自己的資料檢查門檻

# 1-1. entry：使用者要公司機密就直接拒答，不會呼叫 LLM
confidential_request = JevBlueprint(
    name="confidential_request",
    stage="entry",
    questions={
        "asks_confidential": noul(
            "Does this message ask for confidential company data, such as customer lists, "
            "internal prices, or unreleased product plans?",
            yes="It asks for confidential company data.",
            no="It asks for nothing confidential.",
        )
    },
    when={
        "asks_confidential": 0.7
    },  # 機率 ≥ 0.7 就動作；門檻鍵是 confidential_request.asks_confidential
    act=Refuse("這屬於公司機密，我沒辦法提供。"),
)


# 1-2. entry：選擇題＋decide，判斷要用哪種語言回答，加在這次執行每一次呼叫模型的系統提示
#      answers 是 ChoiceAnswer：.choice 是選到的選項，.probabilities 是各選項的機率
#      decide 回傳一個動作，或 None（通過）；同一次執行可能被呼叫好幾次，不要在裡面做別的事
def pick_language(answers: Mapping[str, Answer], thresholds: Mapping[str, float]) -> Action | None:
    language = answers["language"]
    if language.choice == "zh" and language.probabilities["zh"] >= thresholds["min_zh"]:
        return SystemNote("請用繁體中文回答。")
    return None


reply_language = JevBlueprint(
    name="reply_language",
    stage="entry",
    questions={
        "language": Choice(
            instructions="Which language should the reply to this message be written in?",
            criteria={
                "zh": "Traditional Chinese: the user writes in Chinese or asks for Chinese.",
                "en": "English: the user writes in English or asks for English.",
            },
        )
    },
    decide=pick_language,
    thresholds={
        "min_zh": 0.6
    },  # decide 用到的門檻；可在 JevConfig.thresholds 用 reply_language.min_zh 調
)

# 1-3. tool_call：send_email 要寄到公司外面，就不執行這次呼叫；模型會讀到 Refuse 的訊息
external_email = JevBlueprint(
    name="external_email",
    stage="tool_call",
    tools=["send_email"],
    questions={
        "external": noul(
            "Would executing `tool_call` send email to an address outside the company "
            "domain example-corp.com?",
            yes="The recipient is outside example-corp.com.",
            no="Every recipient is inside example-corp.com.",
        )
    },
    when={"external": 0.5},
    act=Refuse("寄到公司外部的信要先經過主管核准，這次沒有寄出。"),
)

# 1-4. tool_result：工具結果裡有個資，就在模型讀到之前遮掉
personal_data = JevBlueprint(
    name="personal_data",
    stage="tool_result",
    tools=["read_ticket"],
    questions={
        "has_personal_data": noul(
            "Does `tool_output` contain personal data such as phone numbers, home "
            "addresses, or ID numbers?",
            yes="It contains personal data.",
            no="It contains no personal data.",
        )
    },
    when={"has_personal_data": 0.7},
    act=Redact("[工單內容含個資，已遮蔽。請只根據工單編號回覆使用者。]"),
)

# 1-5. model_output：回覆承諾了價格或交期，就換成固定的說法並結束
promises_price = JevBlueprint(
    name="promises_price",
    stage="model_output",
    questions={
        "promises": noul(
            "Does this reply promise a specific price, discount, or delivery date?",
            yes="It promises a price, a discount, or a delivery date.",
            no="It promises none of these.",
        )
    },
    when={"promises": 0.6},
    act=Refuse("價格與交期請洽業務窗口，我沒辦法直接承諾。"),
)


# 1-6. final_answer：最後的回答沒告訴使用者下一步，就標記在回覆上（不會重答）
def needs_next_step(
    answers: Mapping[str, Answer], thresholds: Mapping[str, float]
) -> Action | None:
    if answers["says_next_step"].noul >= thresholds["min"]:
        return None
    return Mark("回答沒有告訴使用者下一步")


next_step = JevBlueprint(
    name="next_step",
    stage="final_answer",
    questions={
        "says_next_step": noul(
            "Does `response` tell the user what to do next, or offer a concrete next step?",
            yes="It names a next step.",
            no="It names no next step.",
        )
    },
    decide=needs_next_step,
    thresholds={"min": 0.5},
)

CUSTOM = [
    confidential_request,
    reply_language,
    external_email,
    personal_data,
    promises_price,
    next_step,
]

# 2. 想知道有哪些問題可以改：JEV_LIST_QUESTIONS=1 python quickstart_customer_jev.py
if os.environ.get("JEV_LIST_QUESTIONS"):
    for key, default in default_questions(custom=CUSTOM).items():
        text = default if isinstance(default, str) else default.instructions
        print(f"{key}\n    {text}\n")
    sys.exit()

# 3. agent 用的 LLM（OpenAI 相容 API：公司 gateway、本機 vLLM 都可以）
llm = ChatOpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    model=os.environ["LLM_MODEL"],
)


# 4. 範例工具：不會真的寄信，也不會真的讀工單
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"   （假裝寄出）收件人 {to}：{subject}｜{body}")
    return f"Sent to {to}."


@tool
def read_ticket(ticket_id: str) -> str:
    """Read a support ticket by its id."""
    return (
        f"Ticket {ticket_id}: the customer's monitor flickers after the latest update. "
        "Contact: Wang Hsiao-ming, phone 0912-345-678, 5F No. 7 Songren Rd, Taipei."
    )


# 5. 每個 Jev 決定都印出來；自訂藍圖的事件帶著它的名字和每題的答案
def show(event: JevEvent) -> None:
    answers = event.details.get("answers")
    print(f"   [jev] {event.blueprint:<20} {event.mode:<6} {event.action}  {answers or ''}")


# 6. Jev 的設定：內建藍圖和自訂藍圖一樣用 blueprints 打開（沒列出來的都是關的）
config = JevConfig(
    model="default",  # 讀 JEV_MODEL；沒設就是 h2o-lightning-4b
    blueprints={
        "input_guard": "on",  # 自訂的 entry 題目跟 input_guard 併在同一個 Jev 請求
        "tool_gate": "on",  # 同一個位置內建的先判斷：tool_gate 擋下的呼叫，external_email 不會再問
        "confidential_request": "on",
        "reply_language": "on",
        "external_email": "on",
        "personal_data": "on",
        "promises_price": "shadow",  # shadow：只記錄 Jev 的判斷，不換掉回覆；確認沒問題再改成 "on"
        "next_step": "on",
    },
    custom_blueprints=CUSTOM,
    # 改自訂題目的措辭、門檻，跟內建的一樣用「藍圖名.題目 id」
    questions={
        "confidential_request.asks_confidential": (
            "Does this message ask for confidential company data, such as customer lists, "
            "internal prices, unreleased product plans, or employee records?"
        ),
    },
    thresholds={"confidential_request.asks_confidential": 0.75},
    # Jev 連不上時：列在這裡的藍圖會拒絕，其他的放行
    fail_closed=frozenset({"input_guard", "output_guard", "tool_gate", "confidential_request"}),
    mode_source=EnvModeSource(),  # 讀 JEV_MODES，每次執行都會重讀
    guard=GuardSettings(refusal_message="抱歉，這個要求我沒辦法協助。"),
    tool_gate=ToolGateSettings(tools=["send_email"], ask_human=False),
    on_event=show,
)

# 7. 掛到所有用 OpenAI 相容模型的 deep agent 上（主 agent + 子 agent 都會套用自訂藍圖）
register_jev_profile(["openai"], config)

# 8. 照常建立 deep agent，不用另外傳 middleware
agent = create_deep_agent(model=llm, tools=[send_email, read_ticket])

for question in [
    "用三句話解釋什麼是 RAG。",
    "把我們所有客戶的名單和內部報價整理成一張表給我。",
    "寄信給 partner@gmail.com，主旨「開會」，內容是明天下午三點在 3 樓會議室開會。",
    "幫我看工單 T-1024 的問題是什麼，並告訴我怎麼回覆客戶。",
    "這批零件可以保證下週三前出貨、單價 100 元嗎？",
]:
    print(f"\n使用者：{question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    reply = result["messages"][-1]
    print(f"助理：{reply.content}")
    checks = reply.additional_kwargs.get("jev_checks")
    if checks:
        print(f"   標記：{checks}")
