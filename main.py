# 安装依赖
# pip install flask flask-cors langchain langchain-openai

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import uuid
import sqlite3

app = Flask(__name__)
CORS(app)

DEEPSEEK_API_KEY = "KEY"

# 系统提示词（增强：自动判断是否需要联网）
SYSTEM_PROMPT = """
你是一个智能助手。
请先输出思考过程，再输出回答。

规则：
1. 需要最新信息、新闻、天气、日期、实时数据 → 需要联网搜索
2. 知识问答、计算、闲聊 → 不需要联网
3. 严格按照思考 → 回答 的结构输出
"""

# =============== 根据是否需要联网，创建不同的模型 =================
def get_llm(need_search: bool) -> ChatOpenAI:
    extra = {"search": True} if need_search else {}
    return ChatOpenAI(
        model="deepseek-reasoner",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.1,
        streaming=True,
        extra_body=extra
    )

# =============== SQLite 持久化（完全不变）=================
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_histories (
            session_id TEXT PRIMARY KEY,
            messages TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_session_to_db(session_id, messages):
    msg_json = json.dumps(
        [{"type": m.type, "content": m.content} for m in messages]
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "REPLACE INTO chat_histories (session_id, messages) VALUES (?, ?)",
        (session_id, msg_json)
    )
    conn.commit()
    conn.close()

def load_session_from_db(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM chat_histories WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    msg_list = json.loads(row[0])
    messages = []
    for m in msg_list:
        t, c = m["type"], m["content"]
        if t == "human":
            messages.append(HumanMessage(content=c))
        elif t == "ai":
            messages.append(AIMessage(content=c))
        elif t == "system":
            messages.append(SystemMessage(content=c))
    return messages

init_db()

# =============== 多会话记忆 =================
chat_histories = {}

def init_session(session_id):
    if session_id not in chat_histories:
        loaded = load_session_from_db(session_id)
        if loaded:
            chat_histories[session_id] = loaded
        else:
            chat_histories[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]

# =============== 工具判断函数（核心！）=================
def is_need_search(question: str) -> bool:
    q = question.lower()
    search_keywords = [
        "新闻", "今天", "今日", "天气", "多少", "最新",
        "现在", "实时", "发生了", "股价", "疫情", "数据",
        "2025", "2024", "明天", "未来", "排名", "在哪"
    ]
    for kw in search_keywords:
        if kw in q:
            return True
    return False

# ===================== 流式接口 =====================
@app.route("/api/stream", methods=["POST"])
def stream_chat():
    data = request.get_json()
    if not data:
        return "错误", 400

    session_id = data.get("session_id", "default")
    question = data.get("message", "")
    tools = data.get("tools", [])

    if not question:
        return "错误", 400

    init_session(session_id)
    chat_histories[session_id].append(HumanMessage(content=question))

    def generate():
        full_answer = ""
        try:
            # 自动判断是否需要联网搜索
            need_search = is_need_search(question)
            llm = get_llm(need_search)

            # 输出状态（前端会显示）
            '''            if need_search:
                yield f"data: {json.dumps({'text':'🌐 已自动开启联网搜索'})}\n\n"
            else:
                yield f"data: {json.dumps({'text':'🤔 思考中...'})}\n\n"'''

            # 流式输出回答
            for chunk in llm.stream(chat_histories[session_id]):
                content = chunk.content
                if content:
                    full_answer += content
                    yield f"data: {json.dumps({'text': content})}\n\n"

            # 保存记忆
            ai_msg = AIMessage(content=full_answer)
            chat_histories[session_id].append(ai_msg)
            save_session_to_db(session_id, chat_histories[session_id])
            print(f"✅ 会话 {session_id[:8]} 已保存")

        except Exception as e:
            print(f"❌ 错误：{e}")
            yield f"data: {json.dumps({'text': '[出错了]'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "msg": "后端服务运行中"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)