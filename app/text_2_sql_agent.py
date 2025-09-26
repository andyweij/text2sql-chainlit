from enum import Enum
from io import StringIO
from logger import get_logger
from typing import List, Dict, Optional, Callable
import aiohttp
import json
import requests

logger = get_logger("Text2SQLAgent")

class Text2SQLAgentStepType(Enum):
    INFERENCE = "inference"
    REVISE = "revise"
    END = "end"
    FIGURE = "figure"
    ERROR = "error"
    SUMMARY = "summary"

class Text2SQLAgentEventType(Enum):
    TURN_START = "turn_start"
    TURN_COMPLETE = "turn_complete"
    STEP_START = "step_start"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETE = "step_complete"

class Text2SQLAgentClient:
    """Client for interacting with the Text2SQL Agent API."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def list_agents(self) -> List[Dict]:
        response = requests.get(f"{self.base_url}/agent")
        response.raise_for_status()
        return response.json()

    def create_agent(self, config: Dict) -> Dict:
        response = requests.post(f"{self.base_url}/agent", json=config)
        response.raise_for_status()
        data = response.json()
        return data

    def update_agent(self, agent_id: str, config: Dict) -> Dict:
        response = requests.patch(f"{self.base_url}/agent/{agent_id}", json=config)
        response.raise_for_status()
        return response.json()

    def delete_agent(self, agent_id: str):
        response = requests.delete(f"{self.base_url}/agent/{agent_id}")
        response.raise_for_status()
    
    def get_document(
        self,
        agent_id: str,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> list:
        logger.info(f"get_document with agent({agent_id})")
        response = requests.get(f"{self.base_url}/agent/{agent_id}/document")
        response.raise_for_status()
        docs = []
        jsonl = response.text.strip()
        for line in jsonl.splitlines():
            docs.append(json.loads(line))
        if callback and callable(callback):
            callback(docs)

    async def async_get_document(
        self,
        agent_id: str,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> list:
        logger.info(f"async_get_document with agent({agent_id})")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/agent/{agent_id}/document"
            ) as response:
                response.raise_for_status()
                docs = []
                jsonl = await response.text()
                jsonl = jsonl.strip()
                for line in jsonl.splitlines():
                    docs.append(line)
                if callback and callable(callback):
                    callback(docs)
                return docs

    def update_document(
        self,
        agent_id: str,
        data: str,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        logger.info(f"update_document with agent({agent_id})")
        file_like_object = StringIO(data)
        response = requests.patch(f"{self.base_url}/agent/{agent_id}/document", files={'file': ('doc.jsonl', file_like_object)})
        response.raise_for_status()
        if callback and callable(callback):
            callback()

    async def async_update_document(
        self,
        agent_id: str,
        input: str,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        logger.info(f"async_update_document with agent({agent_id})")
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                'file',
                StringIO(input),
                filename='doc.jsonl',
                content_type='application/octet-stream')
            async with session.patch(
                f"{self.base_url}/agent/{agent_id}/document", data=data
            ) as response:
                response.raise_for_status()
                if callback and callable(callback):
                    callback()

    def get_session(
        self,
        session_id: str,
        offset: int = 0,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        logger.info(f"get_session with id({session_id}), offset({offset})")

        try:
            url = f"{self.base_url}/agent/session/{session_id}"
            params = {"offset": str(offset)}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            data = None

        if callback and callable(callback):
            callback(data)

        return data

    async def async_get_session(
        self,
        session_id: str,
        offset: int = 0,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        logger.info(f"async_get_session with id({session_id}), offset({offset})")

        url = f"{self.base_url}/agent/session/{session_id}"
        params = {"offset": str(offset)}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                try:
                    response.raise_for_status()
                    data = await response.json()
                except aiohttp.ClientResponseError as e:
                    data = None

                if callback and callable(callback):
                    callback(data)

                return data

    def create_session(
        self,
        agent_id: str,
        user_identifier: str,
        session_name: str,
        session_id: str = None,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        logger.info(f"create_session: agent({agent_id}), user({user_identifier}), name: {session_name}, session_id: {session_id}")
        response = requests.post(
            f"{self.base_url}/agent/{agent_id}/session", params={"user_id": user_identifier, "session_name": session_name, "session_id": session_id}
        )
        response.raise_for_status()
        data = response.json()
        if callback and callable(callback):
            callback(data)
        return data

    async def async_create_session(
        self,
        agent_id: str,
        user_identifier: str,
        session_name: str,
        session_id: str = None,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        logger.info(f"async_create_session: agent({agent_id}), user({user_identifier}), name: {session_name}, session_id: {session_id}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/agent/{agent_id}/session", params={"user_id": user_identifier, "session_name": session_name, "session_id": session_id}
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if callback and callable(callback):
                    callback(data)
                return data

    def create_turn(
        self,
        session_id: str,
        messages: List[Dict],
        stream: bool = True,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        logger.info(f"create_turn with session({session_id}) and messages: {messages}")
        response = requests.post(
            f"{self.base_url}/agent/session/{session_id}/turn", 
            json=messages, 
            params={"stream": stream}
        )
        response.raise_for_status()
        data = response.json()
        if callback and callable(callback):
            callback(data)
        return data

    async def async_create_turn(
        self,
        session_id: str,
        messages: List[Dict],
        stream: bool = True,
        callback: Callable[[Dict], None] = None
    ):
        logger.info(f"async_create_turn with session({session_id}) and messages: {messages}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/agent/session/{session_id}/turn", 
                json=messages, 
                params={"stream": str(stream)}
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    buffer += chunk.decode("utf-8", errors="replace")
                    
                    # 檢查緩衝區中是否有完整的行 (以 \n 為分隔)
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data:"):
                            json_str = line[5:].strip()
                            if json_str:
                                try:
                                    data = json.loads(json_str)
                                except json.JSONDecodeError as e:
                                    # 若解析失敗，表示該行資料可能還沒完整，將該行重新加回緩衝區並跳出循環，等待更多資料
                                    buffer = line + "\n" + buffer
                                    break
                                if callback and callable(callback):
                                    callback(data)
                                yield data                
