import asyncio
import chainlit as cl
import chainlit.data as cl_data
import json
import os
import plotly.graph_objects as go
import plotly.io as pio
from chainlit.input_widget import Select, Switch, Slider, TextInput
from chainlit.types import ThreadDict
from chainlit.server import app
from dataclasses import dataclass
from fastapi.routing import APIRoute
from fastapi.responses import Response, JSONResponse
from logger import get_logger
from typing import Optional, Dict
from text_2_sql_agent import Text2SQLAgentClient, Text2SQLAgentEventType, Text2SQLAgentStepType
from urllib.parse import urlparse
from fastapi import Request, Response
from account import authenticate_account
from pydantic import BaseModel, Field
import requests

logger = get_logger("Chainlit")

agent_sessions_cache = {
}

settings = {
}


async def init_sqlite():
    try:
        parsed = urlparse(SQLITE_DATABASE_URL)
        file_path = parsed.path
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured SQLite directory '{directory}'.")
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(sqlite_base.metadata.create_all)
    except Exception as e:
        logger.exception(f"Unable to connect sqlite db: {str(e)}")


# 註冊 data layer
db_type = os.getenv("DATABASE_TYPE", "sqlite")
db_url = os.getenv("DATABASE_URL")
if db_type == "postgres" and db_url:
    from postgres_data_layer import PostgresDataLayer

    cl_data._data_layer = PostgresDataLayer()
else:
    from sqlite_data_layer import SQLITE_DATABASE_URL, engine as sqlite_engine, Base as sqlite_base, SQLiteDataLayer

    asyncio.run(init_sqlite())
    cl_data._data_layer = SQLiteDataLayer()

queues: Dict[str, asyncio.Queue] = {}
queues_lock = asyncio.Lock()
consumer_tasks: Dict[str, asyncio.Task] = {}

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL")


@dataclass
class MyChatProfile(cl.ChatProfile):
    agent_id: Optional[str] = None
    is_prometheus_agent: bool = False


chat_profiles = []


def setup_chat_profiles(agents=None):
    global chat_profiles
    chat_profiles = []

    if not agents:
        text_2_sql_agent_client = Text2SQLAgentClient(base_url=AGENT_BASE_URL)
        agents = text_2_sql_agent_client.list_agents()

    if not agents:
        return

    for agent in agents:
        starters = []
        if "question_samples" in agent and isinstance(agent["question_samples"], list):
            for question in agent["question_samples"]:
                starters.append(
                    cl.Starter(
                        label=question["label"] if "label" in question else None,
                        message=question["message"] if "message" in question else None,
                        icon=question.get("icon_url") or "/public/assets/idea.svg"
                    ),
                )
        chat_profiles.append(
            MyChatProfile(
                name=agent.get("name") if agent.get("name") else f'agent-{agent["agent_id"][:5]}',
                icon=agent.get("icon_url") or "/public/assets/database.svg",
                markdown_description=agent["description"] if "description" in agent else None,
                starters=starters,
                agent_id=agent["agent_id"],
                is_prometheus_agent=("prometheus_url" in agent.get("agent_config", {}))
            )
        )
    if chat_profiles and not any(chat_profile.default for chat_profile in chat_profiles):
        chat_profiles[0].default = True


agent_status = "Healthy"
agent_error = "N/A"
if not bool(AGENT_BASE_URL):
    agent_status = "Unhealthy"
    agent_error = "Agent base url is not configured."
else:
    text_2_sql_agent_client = Text2SQLAgentClient(base_url=AGENT_BASE_URL)

    agents = text_2_sql_agent_client.list_agents()
    if not agents:
        agent_status = "Unhealthy"
        agent_error = "No agent is available."
    else:
        setup_chat_profiles(agents)


def get_user_session_agent_id() -> str:
    selected_chat_profile = cl.user_session.get("chat_profile")
    agent_id = None
    default_agent_id = None
    for chat_profile in chat_profiles:
        if chat_profile.name == selected_chat_profile:
            agent_id = chat_profile.agent_id
            break
        if chat_profile.default:
            default_agent_id = chat_profile.agent_id
    return agent_id or default_agent_id


def is_prometheus_agent(agent_id: str) -> bool:
    for chat_profile in chat_profiles:
        if agent_id == chat_profile.agent_id:
            return chat_profile.is_prometheus_agent
    return False


async def consumer_coroutine(step_name: str, step_id: str, component: str = "step"):
    queue = queues[step_id]

    logger.info(f"Consumer for step_id ({step_id}) is starting.")
    if component == "message":
        message = cl.Message(content="")
        await message.send()
        while True:
            delta = await queue.get()
            if delta is None:
                # None 代表結束
                logger.info(f"Consumer for step_id ({step_id}) received stop signal.")
                break
            await message.stream_token(delta)
        await message.update()
    elif component == "figure":
        """
          這邊先送一個 cl.Message 是為了讓此 Plotly element 的 Message 擁有 parentId，
          可以與其它擁有一樣 parentId 的 Steps 元件 group 在一起。
        """
        message = cl.Message(content="")
        await message.send()
        fig_str = ""
        while True:
            delta = await queue.get()
            if delta is None:
                # None 代表結束
                logger.info(f"Consumer for step_id ({step_id}) received stop signal.")
                break
            fig_str += delta
            await message.stream_token("")
        try:
            figure_obj = to_plotly_figure(fig_str)
            plotly_element = cl.Plotly(name="chart", figure=figure_obj, display="inline")
            message.elements = [plotly_element]
            message.content = ""
            await message.update()
        except Exception as e:
            logger.exception(f"Unable to generate plotly figure: {str(e)}")
            figure_obj = go.Figure()
            figure_obj.add_annotation(
                text="無法產生 Plotly Figure!",  # The error message in Chinese
                xref="paper", yref="paper",  # Reference to the entire plotting area
                x=0.5, y=0.5,  # Positioning at the center
                showarrow=False,  # No arrow pointing to the text
                font=dict(size=20, color="red"),  # Font size and color
                align="center"  # Center alignment
            )

            # Update layout to remove axes and grid for a cleaner look
            figure_obj.update_layout(
                title="錯誤訊息",  # Title of the figure
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white"  # Background color
            )
            plotly_element = cl.Plotly(name="chart", figure=figure_obj, display="inline")
            message.elements = [plotly_element]
            message.content = ""
            await message.update()
    else:
        async with cl.Step(name=step_name, type="llm") as step:
            try:
                while True:
                    delta = await queue.get()
                    if delta is None:
                        # None 代表結束
                        logger.info(f"Consumer for step_id ({step_id}) received stop signal.")
                        break
                    await step.stream_token(delta)
            except Exception as e:
                logger.exception(f"Error in consuming step messages: {str(e)}")
            await step.update()


async def producer(stream):
    global queues, consumer_tasks

    try:
        async for part in stream:
            event = part.get("event", {})
            payload = event.get("payload", {})
            event_type = payload.get("event_type")
            step_id = payload.get("step_id", None)
            delta = payload.get("delta", "")

            if event_type == Text2SQLAgentEventType.TURN_START.value:
                turn_id = payload.get("turn_id", None)
                logger.info(f"TURN_START({turn_id})...")
            elif event_type == Text2SQLAgentEventType.TURN_COMPLETE.value:
                turn_id = payload.get("turn", {}).get("turn_id", None)
                logger.info(f"TURN_COMPLETE({turn_id})!")
            elif event_type == Text2SQLAgentEventType.STEP_START.value:
                step_type = payload.get("step_type", None)
                step_name = payload.get("info", {}).get("name", None)
                logger.info(f"STEP_START({step_type}, {step_id}, {step_name})")
                if step_type == Text2SQLAgentStepType.END.value or step_type == Text2SQLAgentStepType.ERROR.value:
                    if step_id not in queues:
                        queues[step_id] = asyncio.Queue()
                        consumer_tasks[step_id] = asyncio.create_task(
                            consumer_coroutine(step_name, step_id, component="message"))
                        logger.info(f"Started consumer task for step_id: {step_id}(message)")
                elif step_type == Text2SQLAgentStepType.FIGURE.value:
                    if step_id not in queues:
                        queues[step_id] = asyncio.Queue()
                        consumer_tasks[step_id] = asyncio.create_task(
                            consumer_coroutine(step_name, step_id, component="figure"))
                        logger.info(f"Started consumer task for step_id: {step_id}(figure)")
                else:
                    if step_id not in queues:
                        queues[step_id] = asyncio.Queue()
                        consumer_task = asyncio.create_task(consumer_coroutine(step_name, step_id, component="step"))
                        consumer_tasks[step_id] = consumer_task
                        logger.info(f"Started consumer task for step_id: {step_id}(step)")
            elif event_type == Text2SQLAgentEventType.STEP_PROGRESS.value:
                if step_id in queues:
                    if isinstance(delta, str):
                        await queues[step_id].put(delta)
                    else:
                        await queues[step_id].put(str(delta))
                        logger.warning(f"Received STEP_PROGRESS for invalid delta: {str(delta)}")
                else:
                    logger.warning(f"Received STEP_PROGRESS for unknown step_id: {step_id}")
            elif event_type == Text2SQLAgentEventType.STEP_COMPLETE.value:
                logger.info(f"STEP_COMPLETE({step_id})")
                if step_id in queues:
                    await queues[step_id].put(None)
                else:
                    logger.warning(f"Received STEP_COMPLETE for unknown step_id: {step_id}")
    except Exception as e:
        logger.exception(f"Error in on_message: {str(e)}")
        error_message = f"發生錯誤: {str(e)}"
        await cl.Message(content=error_message).send()

    async with queues_lock:
        for q in queues.values():
            await q.put(None)


def to_plotly_figure(json_str) -> go.Figure:
    try:
        figure = pio.from_json(json_str)
    except Exception as e:
        raise ValueError(f"创建 Plotly Figure 时出错: {e}")
    return figure


# 1. 定義一個 Pydantic 模型來接收前端傳來的 JSON 資料
#    這能確保資料格式正確並提供自動驗證
class CreateAgentRequest(BaseModel):
    agentName: str
    llmModelName: str
    endpoint: str
    llmApiKey: str
    embeddingModalName: str
    embeddingEndpoint: str  # 您的 API 結構中似乎共用 endpoint
    embeddingApiKey: str    # 您的 API 結構中似乎共用 llmApiKey
    dbConnectionString: str
    # 欄位名稱要和 React formik 的 initialValues 完全一樣


# 2. 建立一個新的 FastAPI POST 端點
@app.post("/api/create-agent")
async def create_agent_endpoint(request: CreateAgentRequest):
    """
    這個端點會接收從前端 Modal 傳來的資料來建立新的 Agent
    """
    logger.info(f"Received request to create agent: {request.agentName}")
    try:
        # 3. 將前端傳來的資料，組裝成 create_agent 方法需要的格式
        agent_payload = {
            "agent_config": {
                "api_type": "openai",  # 根據您的需求，可能需要從前端傳來
                "api_key": request.llmApiKey,
                "base_url": request.endpoint,
                "model": request.llmModelName,
                "sampling_params": {
                    "max_tokens": 1600,
                    "temperature": 0.7
                },
                 "embedding_model": {
                    "api_key": request.embeddingApiKey,  # <-- 修改處：使用獨立的 embedding key
                    "base_url": request.embeddingEndpoint,  # <-- 修改處：使用独立的 embedding endpoint
                    "model": request.embeddingModalName,
                    "api_type": "openai"
                },
                "db_uri": request.dbConnectionString
            },
            "name": request.agentName,
            "question_samples": []  # 您可以預設一些範例問題
        }

        # 4. 呼叫 client 的 create_agent 方法
        text_2_sql_agent_client = Text2SQLAgentClient(base_url=AGENT_BASE_URL)
        new_agent = text_2_sql_agent_client.create_agent(config=agent_payload)

        # 5. 【非常重要】刷新 Chat Profiles 讓新 Agent 顯示在 UI 上
        setup_chat_profiles()

        logger.info(f"Agent '{request.agentName}' created successfully.")
        return JSONResponse(
            content={"message": "Agent created successfully", "agent": new_agent},
            status_code=201
        )

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text
        logger.error(f"Failed to create agent. Status: {e.response.status_code}, Detail: {error_detail}")
        return JSONResponse(
            content={"error": "Failed to create agent", "detail": error_detail},
            status_code=e.response.status_code
        )
    except Exception as e:
        logger.exception("An unexpected error occurred while creating agent.")
        return JSONResponse(
            content={"error": "An internal server error occurred.", "detail": str(e)},
            status_code=500
        )


# --- 修改結束 ---
@app.head("/health")
async def agent_status_check():
    """
        因為 Chainlit 有較高的優先順序去處理所有 GET method 的 endpoints，用以 serving 前端 UI，
        Route: {'GET'} /{full_path:path}, Name: serve
        所以這邊以 HEAD method 新增 /health endpoint 來做 agent 狀態檢查。
    """
    return Response(
        headers={"X-Agent-Status": agent_status, "X-Agent-Error": agent_error},
        status_code=200
    )


logger.info(f"Available routes:")
for route in app.router.routes:
    if isinstance(route, APIRoute):
        logger.info(f"Route: {route.methods} {route.path}, Name: {route.name}")
    else:
        logger.info(f"Route: {route.path}, Name: {route.name} (type: {type(route)})")

"""
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    logger.info(f"password_auth_callback {username}")

    if username == "guest":
        setup_chat_profiles()
        return cl.User(
            identifier="guest",
            metadata={"role": "guest", "provider": "credentials"}
        )

    try:
        account_info = authenticate_account(username, password)
        if not account_info is None:
            setup_chat_profiles()
            return cl.User(
                identifier=account_info.get("identifier"),
                metadata={
                    "role": account_info.get("role") or "user",
                    "provider": account_info.get("provider") or "credentials"
                }
            )
    except Exception as e:
        logger.exception(f"Error in auth_callback: {str(e)}")

    logger.info(f"無效的登入動作(username: {username})")
    return None
"""


@cl.on_logout
def handle_logout(request: Request, response: Response):
    logger.info(f"on_logout")

    global chat_profiles
    chat_profiles = []


@cl.on_chat_start
async def start():
    """
    @cl.on_chat_start 和 cl_data._data_layer.create_step (type: "run", name: "on_chat_start") 同步執行，不可以互相依賴。
    """
    logger.info(f"on_chat_start")

    agent_id = get_user_session_agent_id()
    is_prometheus = is_prometheus_agent(agent_id)
    logger.info(f"is_prometheus: {is_prometheus}")
    if is_prometheus:
        await cl.ChatSettings([]).send()
    elif agent_id:
        docs = await text_2_sql_agent_client.async_get_document(agent_id)
        initial = "\n".join(str(doc) for doc in docs) if docs else ""
        await cl.ChatSettings(
            [
                TextInput(
                    id="docs",
                    label="documents",
                    multiline=True,
                    initial=initial
                ),
            ]
        ).send()


@cl.on_settings_update
async def setup_agent(settings):
    logger.info(f"on_settings_update", settings)

    selected_chat_profile = cl.user_session.get("chat_profile")
    logger.info(f"setup_agent: {selected_chat_profile}")

    agent_id = get_user_session_agent_id()
    if agent_id:
        settings_docs = settings["docs"]
        await text_2_sql_agent_client.async_update_document(agent_id, settings_docs)


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return chat_profiles


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    logger.info(f'on_chat_resume: {thread["name"]}({thread["id"]}) with {len(thread["steps"])} steps.')
    agent_id = get_user_session_agent_id()
    is_prometheus = is_prometheus_agent(agent_id)
    logger.info(f"is_prometheus: {is_prometheus}")
    if is_prometheus:
        await cl.ChatSettings([]).send()
    elif agent_id:
        docs = await text_2_sql_agent_client.async_get_document(agent_id)
        initial = "\n".join(str(doc) for doc in docs) if docs else ""
        await cl.ChatSettings(
            [
                TextInput(
                    id="docs",
                    label="documents",
                    multiline=True,
                    initial=initial
                ),
            ]
        ).send()

    thread_id = thread.get("id")
    session = agent_sessions_cache.get(thread_id)
    if not session:
        logger.info(f"Check if session({thread_id}) was created before!")
        session = await text_2_sql_agent_client.async_get_session(thread_id)
        agent_sessions_cache[thread_id] = session
    if not session:
        current_user = cl.user_session.get("user")
        if current_user:
            logger.info(f"Create agent session({thread_id}) for current user({current_user.identifier})...")
            session = await text_2_sql_agent_client.async_create_session(
                agent_id,
                current_user.identifier,
                "undefined",
                session_id=str(thread_id)
            )
            logger.info(f"Session created: {session}")
            agent_sessions_cache[thread_id] = session
    elif "agent_id" in session and session["agent_id"] != agent_id:
        # Workaround for that Chainlit might save wrong chat_profile in metadata.
        for chat_profile in chat_profiles:
            if chat_profile.agent_id == session["agent_id"]:
                logger.info(f"Revise current chat profile: {chat_profile.name}")
                cl.user_session.set("chat_profile", chat_profile)
                break


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"on_message")

    session = agent_sessions_cache.get(message.thread_id)
    if not session:
        logger.info(f"Check if session({message.thread_id}) was created before!")
        session = await text_2_sql_agent_client.async_get_session(message.thread_id)
        agent_sessions_cache[message.thread_id] = session
    if not session:
        current_user = cl.user_session.get("user")
        if current_user:
            logger.info(f"Create agent session({message.thread_id}) for current user({current_user.identifier})...")
            agent_id = get_user_session_agent_id()
            session = await text_2_sql_agent_client.async_create_session(
                agent_id,
                current_user.identifier,
                "undefined",
                session_id=str(message.thread_id)
            )
            logger.info(session)
            agent_sessions_cache[message.thread_id] = session

    if session:
        stream = text_2_sql_agent_client.async_create_turn(
            message.thread_id,
            messages=[{
                "role": "user",
                "content": str(message.content),
                "context": "string"
            }],
            stream=True
        )
        await producer(stream)
    else:
        error_message = "發生錯誤：未建立 agent session！"
        await cl.Message(content=error_message).send()
