import os
import re
from collections import defaultdict, deque
from fastapi import FastAPI, Request, BackgroundTasks
from openai import OpenAI
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)

app = FastAPI()

# ดึงค่าจาก Environment Variables ของ Render
LINE_ACCESS_TOKEN = os.environ.get("LINE_ACCESS_TOKEN", "").strip()
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://thaillm.or.th/api/v1").strip()
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "").strip()

configuration = Configuration(access_token=LINE_ACCESS_TOKEN)
llm_client = OpenAI(base_url=MODEL_API_URL, api_key=MODEL_API_KEY)
group_history = defaultdict(lambda: deque(maxlen=5))

with open("chatbot_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


def clean_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def analyze_message(context_list: list, current_msg: str) -> str:
    context_str = "\n".join(context_list) if context_list else "(ไม่มีข้อความก่อนหน้า)"
    user_content = f"[บริบท]:\n{context_str}\n\n[ข้อความที่ต้องประเมิน]:\n\"{current_msg}\""
    response = llm_client.chat.completions.create(
        model="pathumma-thaillm-qwen3-8b-think-3.0.0",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    return clean_think_tags(response.choices[0].message.content)


def process_and_reply(chat_id: str, reply_token: str, msg_text: str):
    result = analyze_message(list(group_history[chat_id]), msg_text)
    group_history[chat_id].append(msg_text)

    print(f"ข้อความ: {msg_text} -> ผลลัพธ์: {result}")

    if "ไม่พบความเสี่ยง" in result:
        return

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        try:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=result)],
                )
            )
        except Exception as e:
            print(f"Reply failed ({e}), falling back to push_message...")
            line_bot_api.push_message(
                PushMessageRequest(
                    to=chat_id,
                    messages=[TextMessage(text=result)],
                )
            )


@app.get("/")
async def health_check():
    return {"status": "running"}


@app.post("/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    for event in body.get("events", []):
        if event.get("type") == "message" and event["message"].get("type") == "text":
            msg_text = event["message"]["text"]
            reply_token = event["replyToken"]
            source = event["source"]
            chat_id = source.get("groupId") or source.get("roomId") or source.get("userId")
            background_tasks.add_task(process_and_reply, chat_id, reply_token, msg_text)
    return {"status": "ok"}
