"""客製化 Jev 流程：用 JevBlueprint 加上自己的藍圖，前面、動作、後面都由你決定。

內建藍圖（input_guard、tool_gate…）的題目和動作是 langchain-jev 寫好的。
自訂藍圖把三件事寫在同一個 JevBlueprint 裡，前後由 library 接好：
    1. 前面：在哪裡問（stage）、問什麼（questions）
         entry         使用者訊息進來時，跟 input_guard 等內建藍圖併在同一個 Jev 請求
         model_output  每一輪模型回覆的文字
         tool_call     工具執行前（只看 tools 列出的工具）
         tool_result   工具回傳後、模型讀到之前（只看 tools 列出的工具）
         final_answer  執行結束時的最後回答
    2. 判斷：when＋act（是非題的機率到門檻就做 act 裡的動作），或 decide（自己寫判斷）
    3. 動作和後面：每個動作的 at 決定在哪裡做
         before_agent  呼叫模型之前結束這次執行（只有 entry 可以用）
         model_call    每一次呼叫模型
         before_model  每一輪開始時
         model_output  每一輪模型回覆
         tool_call     工具執行前（動作要給 tools，除非就是前面問的那個工具）
         tool_result   工具回傳後（同上）
         final_answer  最後的回答
       at 跟前面同一個位置：當下就做；at 在前面之後：觸發之後，這次執行剩下的時間都套用
       不寫 at 就用動作平常的位置；位置在前面之前（例如 final_answer 問、model_call 做）會報錯

每個位置能用的動作（Shape…、Rewrite…、Notify 是自己寫函式的動作）：
    before_agent、before_model  Refuse（結束這次執行）
    model_call    SystemNote（加系統提示）、UseModel（換模型）、NoTools（不給工具）、
                  DropTools（拿掉某些工具）、ShapeModelCall（自己改 ModelRequest）
    model_output  Refuse（換成拒答並結束）、RewriteReply（自己改回覆文字）
    tool_call     Refuse（不執行）、AskHuman（等人核准，要有 checkpointer）、
                  ShapeToolCall（自己改參數）
    tool_result   Warn（前面加註）、Redact（遮蔽）、Refuse（結束這次執行）、
                  RewriteToolResult（自己改結果文字）
    final_answer  Mark（標記在 additional_kwargs["jev_checks"]）、RewriteReply、
                  Notify（結束時呼叫一次，被拒答也會呼叫）

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
    JEV_TRACE=1    調整 Jev 題目時用：每次問 Jev 都印出 Jev 讀到的內容、每題的鍵和回答、
                   比較的門檻（題目鍵就是 JevConfig.questions 用來改措辭的鍵）
    JEV_TRACE=full 再加上每題的原文和 Jev 讀到的完整內容
    JEV_TRACE_FILE=jev_trace.jsonl  每次請求和每個決定各存成一行 JSON，方便整理成表
"""

import os
import re
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
    JevTrace,
    Mark,
    Notify,
    Refuse,
    RewriteReply,
    RewriteToolResult,
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


# 1-1. entry 前面 → before_agent 後面：要公司機密就直接拒答，不會呼叫 LLM
#      同一個藍圖再加 Notify（final_answer 後面）：執行結束時通知資安，被拒答也會通知
def report_to_security(answers: Mapping[str, Answer], final_text: str) -> None:
    probability = answers["asks_confidential"].noul
    print(f"   （通知資安）有人要求公司機密（Jev：{probability:.2f}），回覆是：{final_text}")


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
    act=[Refuse("這屬於公司機密，我沒辦法提供。"), Notify(report_to_security)],
)


# 1-2. entry 前面 → model_call 後面：選擇題＋decide，決定用哪種語言回答
#      answers 是 ChoiceAnswer：.choice 是選到的選項，.probabilities 是各選項的機率
#      decide 回傳動作、動作清單，或 None（通過）；可能被呼叫好幾次，不要在裡面做別的事
def pick_language(answers: Mapping[str, Answer], thresholds: Mapping[str, float]) -> Action | None:
    language = answers["language"]
    if language.choice == "zh" and language.probabilities["zh"] >= thresholds["min_zh"]:
        return SystemNote("請用繁體中文回答。")  # 不寫 at：SystemNote 平常就在 model_call
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

# 1-3. entry 前面 → tool_call 後面：只是問問題的請求，這次執行都不准寫檔或執行指令
read_only_request = JevBlueprint(
    name="read_only_request",
    stage="entry",
    questions={
        "only_asks": noul(
            "Does the user only want an explanation or information, with no file written, "
            "edited, or command run?",
            yes="The user only wants information.",
            no="The user wants something created, changed, or run.",
        )
    },
    when={"only_asks": 0.8},
    act=Refuse(
        "這次只回答問題，不會寫檔或執行指令。",
        at="tool_call",
        tools=["write_file", "edit_file", "execute"],
    ),
)

# 1-4. tool_call 前面 → 同一個位置：send_email 要寄到公司外面，就不執行這次呼叫
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


# 1-5. tool_result 前面 → 三個後面：工單裡有個資時
#      - 同一個位置：自己寫函式，只遮電話和地址，其他內容保留
#      - tool_call：之後這次執行的 send_email 都不執行
#      - model_call：之後每一次呼叫模型都加提示
def mask_contact(text: str) -> str:
    text = re.sub(r"09\d{2}-\d{3}-\d{3}", "09xx-xxx-xxx", text)
    return re.sub(r"\d+F No\. \d+ \w+ Rd, \w+", "[地址已遮蔽]", text)


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
    act=[
        RewriteToolResult(mask_contact),
        Refuse("讀過含個資的工單後，這次不能再寄信。", at="tool_call", tools=["send_email"]),
        SystemNote("不要把客戶的電話或地址告訴使用者。", at="model_call"),
    ],
)

# 1-6. model_output 前面 → 同一個位置：回覆承諾了價格或交期，就自己寫函式在後面加註
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
    act=RewriteReply(lambda text: f"{text}\n（價格與交期以業務的正式報價為準。）"),
)


# 1-7. final_answer 前面 → 同一個位置：最後的回答沒告訴使用者下一步，就標記在回覆上
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
    read_only_request,
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


# 5. 每個 Jev 決定都印出來
#    - 前面判斷時：動作名稱（好幾個用 + 連起來）、每題的答案、每個動作的位置
#    - 後面真的動作時：例如 refuse@tool_call，表示在 tool_call 擋下了哪個工具
#      （model_call 每次呼叫模型都會套用，不另外印）
def show(event: JevEvent) -> None:
    details = event.details
    shown = details.get("answers") or details.get("tool") or ""
    where = f"  → {details['at']}" if details.get("at") else ""
    print(f"   [jev] {event.blueprint:<20} {event.mode:<6} {event.action}  {shown}{where}")


# 6. Jev 的設定：內建藍圖和自訂藍圖一樣用 blueprints 打開（沒列出來的都是關的）
config = JevConfig(
    model="default",  # 讀 JEV_MODEL；沒設就是 h2o-lightning-4b
    blueprints={
        "input_guard": "on",  # 自訂的 entry 題目跟 input_guard 併在同一個 Jev 請求
        "tool_gate": "on",  # 同一個位置內建的先判斷：tool_gate 擋下的呼叫，自訂的不會再問
        "confidential_request": "on",
        "reply_language": "on",
        "read_only_request": "on",
        "external_email": "on",
        "personal_data": "on",
        "promises_price": "on",  # 改成 "shadow"：只記錄 Jev 的判斷和會在哪裡動作，不改變結果
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
    # JEV_TRACE：印出每次問 Jev 的內容和回答；決定已經由 show 印出，所以 decisions=False
    trace=JevTrace.from_env(decisions=False),
)

# 7. 掛到所有用 OpenAI 相容模型的 deep agent 上（主 agent + 子 agent 都會套用自訂藍圖）
register_jev_profile(["openai"], config)

# 8. 照常建立 deep agent，不用另外傳 middleware
agent = create_deep_agent(model=llm, tools=[send_email, read_ticket])

for question in [
    "用三句話解釋什麼是 RAG。",
    "把我們所有客戶的名單和內部報價整理成一張表給我。",
    "寄信給 partner@gmail.com，主旨「開會」，內容是明天下午三點在 3 樓會議室開會。",
    "幫我看工單 T-1024 的問題，整理後寄到 support@example-corp.com。",
    "這批零件可以保證下週三前出貨、單價 100 元嗎？",
]:
    print(f"\n使用者：{question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    reply = result["messages"][-1]
    print(f"助理：{reply.content}")
    checks = reply.additional_kwargs.get("jev_checks")
    if checks:
        print(f"   標記：{checks}")
