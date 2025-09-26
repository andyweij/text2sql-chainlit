import json
import os
import uuid
import chainlit as cl
import chainlit.data as cl_data
from bidict import bidict
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict, FeedbackDict
from chainlit.types import (
    PaginatedResponse,
    PageInfo,
    Pagination,
    ThreadDict,
    ThreadFilter,
    Feedback,
)
from datetime import datetime
from logger import get_logger
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, Integer, select, text
from typing import List, Dict, Optional, Union

logger = get_logger("SQLiteDataLayer")

# 若環境變數中未設定 SQLITE_DATABASE_URL，預設使用 SQLite
SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite+aiosqlite:////app/data/chainlit.db")
engine = create_async_engine(SQLITE_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
logger.info(f"SQLITE_DATABASE_URL: {SQLITE_DATABASE_URL}")

Base = declarative_base()

def parse_json_field(value, expected_type):
    """
    將傳入的 JSON 轉成 expected_type。
    - 若 value 為 None，則回傳 empty instance (例如 {} 或 [])。
    - 若 value 為字串，嘗試使用 json.loads 轉換。
    - 若轉換後符合 expected_type，則直接回傳；否則回傳 empty instance。
    """
    if value is None:
        return expected_type()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as e:
            logger.warning(f"Error parsing JSON field: {e}")
            return expected_type()
    if isinstance(value, expected_type):
        return value
    return expected_type()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    identifier = Column(String, unique=True, nullable=False)
    createdAt = Column(String)
    metadata_json = Column(JSON, name="metadata") # 用 JSON 儲存 metadata 資料，預期為 dict。

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, User):
            return self.id == other.id
        return False

class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    createdAt = Column(String)
    userId = Column(String, ForeignKey("users.id"))
    userIdentifier = Column(String)
    tags = Column(JSON) # 用 JSON 儲存 tags 資料，預期為 list。
    metadata_json = Column(JSON, name="metadata") # 用 JSON 儲存 metadata 資料，預期為 dict。

class Step(Base):
    __tablename__ = "steps"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    type = Column(String)
    threadId = Column(String, ForeignKey("threads.id"))
    tags = Column(JSON) # 用 JSON 儲存 tags 資料，預期為 list。
    input = Column(String)
    output = Column(String)
    createdAt = Column(String)
    parentId = Column(String)
    streaming = Column(Boolean, default=False)
    waitForAnswer = Column(Boolean, default=False)
    isError = Column(Boolean, default=False)
    metadata_json = Column(JSON, name="metadata") # 用 JSON 儲存 metadata 資料，預期為 dict。

class ElementModel(Base):
    __tablename__ = "elements"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    threadId = Column(String, ForeignKey("threads.id", ondelete="CASCADE"))
    type = Column(String)
    url = Column(String)
    chainlitKey = Column(String)
    name = Column(String, nullable=False)
    display = Column(String)
    objectKey = Column(String)
    size = Column(String)
    page = Column(Integer)
    language = Column(String)
    forId = Column(String)
    mime = Column(String)

    def __repr__(self):
        return (
            f"<Element(id={self.id}, threadId={self.threadId}, name={self.name}, "
            f"type={self.type}, url={self.url}, language={self.language})>"
        )

class FeedbackModel(Base):
    __tablename__ = "feedbacks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    forId = Column(String, nullable=False)
    threadId = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    value = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)

class UsersCache:
    def __init__(self):
        self.data = bidict()

    def set(self, user_id, identifier, value):
        self.data[(user_id, identifier)] = value

    def get_by_id(self, user_id):
        for (id_, identifier), value in self.data.items():
            if id_ == user_id:
                return value
        return None

    def get_by_identifier(self, identifier):
        for (id_, ident), value in self.data.items():
            if ident == identifier:
                return value
        return None

    def delete(self, user_id, identifier):
        self.data.pop((user_id, identifier), None)

class SQLiteDataLayer(cl_data.BaseDataLayer):
    def __init__(self):
        self.users_cache = UsersCache()
        self.threads_cache = {}

    async def get_user(self, identifier: str):
        logger.info(f"get_user: {identifier}")
        # 優先從 cache 中查詢
        user = self.users_cache.get_by_identifier(identifier)
        if user:
            logger.info(f"Cache hit for user({identifier}, id: {user.id})")
            return cl.PersistedUser(
                id=user.id,
                identifier=user.identifier,
                createdAt=user.createdAt if user.createdAt else datetime.now().isoformat(),
                metadata=parse_json_field(user.metadata_json, dict)
            )
        # 從資料庫中查詢
        async with SessionLocal() as session:
            try:
                result = await session.execute(
                    text('SELECT id, identifier, createdAt, metadata FROM users WHERE identifier = :identifier'),
                    {"identifier": identifier},
                )
                user_row = result.fetchone()
                if not user_row:
                    logger.info(f"User {identifier} not found in database")
                    return None

                logger.info(f"Found user({identifier}, id: {user_row.id}) in database.")
                parsed_metadata = parse_json_field(user_row.metadata, dict)
                user = User(
                    id=user_row.id,
                    identifier=user_row.identifier,
                    createdAt=user_row.createdAt,
                    metadata_json=parsed_metadata
                )
                self.users_cache.set(user.id, user.identifier, user)
                return cl.PersistedUser(
                    id=user.id,
                    identifier=user.identifier,
                    createdAt=user.createdAt if user.createdAt else datetime.now().isoformat(),
                    metadata=parsed_metadata
                )
            except Exception as e:
                logger.warning(f"Error in get_user: {e}")
                return None

    async def create_user(self, user: cl.User):
        logger.info(f"create_user: {user}")
        async with SessionLocal() as session:
            try:
                created_at = getattr(user, 'createdAt', None)
                new_user = User(
                    id=str(uuid.uuid4()),
                    identifier=user.identifier,
                    createdAt=created_at if created_at else datetime.now().isoformat(),
                    metadata_json=user.metadata
                )
                session.add(new_user)
                await session.commit()
                self.users_cache.set(new_user.id, new_user.identifier, new_user)
                return cl.PersistedUser(
                    id=new_user.id,
                    createdAt=new_user.createdAt,
                    identifier=new_user.identifier,
                    metadata=parse_json_field(new_user.metadata_json, dict)
                )
            except Exception as e:
                logger.warning(f"Error in create_user: {e}")
                return None

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        logger.info(f"list_threads: {filters}")
        async with SessionLocal() as session:
            try:
                conditions = []
                if filters.userId is not None:
                    conditions.append(Thread.userId == filters.userId)
                if filters.search:
                    conditions.append(Thread.name.ilike(f"%{filters.search}%"))
                if filters.feedback is not None:
                    subq = (
                        select(FeedbackModel.threadId)
                        .where(FeedbackModel.value == filters.feedback)
                        .distinct()
                    )
                    subq = subq.scalar_subquery()
                    conditions.append(Thread.id.in_(subq))

                query = select(Thread).where(*conditions).order_by(Thread.createdAt.asc())
                result = await session.execute(query)
                threads = result.scalars().all()

                thread_dicts = []
                for thread in threads:
                    steps = await self.get_steps_by_thread(thread.id)
                    elements = await self.get_elements_by_thread(thread.id)
                    logger.info(f"Thread({thread.name}, {thread.id}) has {len(steps)} steps and {len(elements)} elements.")

                    is_thread_empty = True
                    for step in steps:
                        if step["type"] != "run":
                            is_thread_empty = False
                            break
                    if not is_thread_empty or len(elements) > 0:
                        thread_dicts.append(
                            ThreadDict(
                                id=thread.id,
                                createdAt=thread.createdAt,
                                name=thread.name,
                                userId=thread.userId,
                                userIdentifier=thread.userIdentifier,
                                tags=parse_json_field(thread.tags, list),
                                metadata=parse_json_field(thread.metadata_json, dict),
                                steps=steps,
                                elements=elements,
                            )
                        )
                logger.info(f"list_threads return {len(thread_dicts)} threads.")
                return PaginatedResponse(
                    data=thread_dicts,
                    pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
                )
            except Exception as e:
                logger.warning(f"Error in list_threads: {e}")
                return PaginatedResponse(data=[], pageInfo=PageInfo(False, None, None))

    async def create_step(self, step_dict: StepDict):
        step_type = step_dict.get("type", None)
        step_name = step_dict.get("name", None)
        thread_id = step_dict["threadId"]
        logger.info(f"create_step: type({step_type}), name({step_name}), thread_id({thread_id})")
        async with SessionLocal() as session:
            if step_type == "run":
                if step_name == "on_chat_start":
                    current_user = cl.user_session.get("user")
                    if current_user:
                        logger.info(f"Create new thread({thread_id}) for user({current_user.identifier})...")
                        user = await self.get_user(current_user.identifier)
                        user_id = user.id if user else None
                        await self.update_thread(
                            thread_id=thread_id,
                            name="...",
                            user_id=user_id
                        )

            logger.info(f'Create new step({step_dict["id"]})...')
            try:
                new_step = Step(
                    id=step_dict["id"],
                    name=step_dict["name"],
                    type=step_dict["type"],
                    threadId=step_dict["threadId"],
                    input=step_dict.get("input"),
                    output=step_dict.get("output"),
                    createdAt=step_dict["createdAt"],
                    parentId=step_dict["parentId"] if step_dict.get("parentId") else None
                )
                session.add(new_step)
                await session.commit()
                logger.info(f"Step({new_step.id}) was created successfully.")
            except Exception as e:
                logger.warning(f"Error in create_step: {e}")

    async def delete_step(self, step_id: str):
        async with SessionLocal() as session:
            try:
                await session.execute(
                    text('DELETE FROM steps WHERE id = :step_id'),
                    {"step_id": step_id},
                )
                await session.commit()
            except Exception as e:
                logger.warning(f"Error in delete_step: {e}")

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        logger.info(f"get_thread: {thread_id}")
        if not thread_id:
            return None

        async with SessionLocal() as session:
            try:
                thread = await session.get(Thread, thread_id)
                if not thread:
                    logger.info(f"Thread {thread_id} not found in the database.")
                    return None

                steps = await self.get_steps_by_thread(thread.id)
                elements = await self.get_elements_by_thread(thread.id)

                thread_dict = ThreadDict(
                    id=thread.id,
                    createdAt=thread.createdAt,
                    name=thread.name,
                    userId=thread.userId,
                    userIdentifier=thread.userIdentifier,
                    tags=parse_json_field(thread.tags, list),
                    metadata=parse_json_field(thread.metadata_json, dict),
                    steps=steps,
                    elements=elements
                )

                self.threads_cache[thread_id] = thread_dict
                return thread_dict
            except Exception as e:
                logger.warning(f"Error in get_thread: {e}")
                return None

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        logger.info(f"update_thread({thread_id})\nname: {name}\nuser id: {user_id}\nmetadata: {metadata}\ntags: {tags}")
        async with SessionLocal() as session:
            try:
                thread = await session.get(Thread, thread_id)
                if not thread:
                    logger.info(f"Thread id {thread_id} not found, create it...")
                    user_identifier = None
                    if user_id:
                        user = self.users_cache.get_by_id(user_id)
                        if user:
                            user_identifier = user.identifier

                    new_thread = Thread(
                        id=thread_id,
                        name=name,
                        createdAt=datetime.now().isoformat(),
                        userId=user_id,
                        userIdentifier=user_identifier,
                        metadata_json=metadata,
                        tags=tags
                    )
                    session.add(new_thread)
                    await session.commit()
                    logger.info(f"Thread({thread_id}) was created.")

                    thread_dict = ThreadDict(
                        id=new_thread.id,
                        createdAt=new_thread.createdAt,
                        name=new_thread.name,
                        userId=new_thread.userId,
                        userIdentifier=new_thread.userIdentifier,
                        tags=parse_json_field(new_thread.tags, list),
                        metadata=parse_json_field(new_thread.metadata_json, dict),
                        steps=[],
                        elements=[],
                    )
                    self.threads_cache[thread_id] = thread_dict
                else:
                    if name:
                        thread.name = name
                    if user_id:
                        thread.userId = user_id
                    if metadata:
                        thread.metadata_json = {**metadata}
                    if tags:
                        thread.tags = tags

                    await session.commit()

                    thread_dict = ThreadDict(
                        id=thread.id,
                        createdAt=thread.createdAt,
                        name=thread.name,
                        userId=thread.userId,
                        userIdentifier=thread.userIdentifier,
                        tags=parse_json_field(thread.tags, list),
                        metadata=parse_json_field(thread.metadata_json, dict),
                        steps=[],
                        elements=[],
                    )
                    self.threads_cache[thread_id] = thread_dict
            except Exception as e:
                logger.warning(f"Error in update_thread: {e}")

    async def delete_thread(self, thread_id: str):
        logger.info(f"delete_thread: {thread_id}")
        async with SessionLocal() as session:
            try:
                await session.execute(
                    text('DELETE FROM threads WHERE id = :thread_id'),
                    {"thread_id": thread_id},
                )
                await session.commit()

                if thread_id in self.threads_cache:
                    del self.threads_cache[thread_id]
            except Exception as e:
                logger.warning(f"Error in delete_thread: {e}")

    async def get_thread_author(self, thread_id: str):
        async with SessionLocal() as session:
            try:
                result = await session.execute(
                    text('SELECT userIdentifier FROM threads WHERE id = :thread_id'),
                    {"thread_id": thread_id},
                )
                author = result.scalar()
                return author
            except Exception as e:
                logger.warning(f"Error in get_thread_author: {e}")
                return None

    async def build_debug_url(self) -> str:
        return "http://localhost:15432/debug"

    async def save_chart_content_to_file(self, file_name: str, content: Optional[Union[bytes, str]] = None):
        logger.info(f"Save chart content to file {file_name}")
        root_dir = Path(__file__).resolve().parent
        chart_dir = root_dir / 'public' / 'chart'
        chart_dir.mkdir(parents=True, exist_ok=True)
        file_path = chart_dir / file_name
        try:
            if isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(content)
            elif isinstance(content, str):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                logger.warning(f"Content is None or unsupported type for file {file_path}. Skipping write.")
                return
            logger.info(f"Content successfully saved to {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save content to {file_path}: {e}")

    async def create_element(self, element: Element):
        logger.info(f"create_element: {element}")
        async with SessionLocal() as session:
            thread = await session.get(Thread, element.thread_id)
            if not thread:
                logger.info(f"Thread({element.thread_id}) doesn't exist!")
                return

            try:
                element_id = element.id if element.id else str(uuid.uuid4())
                plotly_chart_file_name = f"{element_id}.json"
                plotly_chart_url = f"/public/chart/{plotly_chart_file_name}"
                new_element = ElementModel(
                    id=element_id,
                    threadId=element.thread_id,
                    type=element.type,
                    url=plotly_chart_url if isinstance(element, cl.Plotly) else element.url,
                    chainlitKey=element.chainlit_key,
                    name=element.name,
                    display=element.display,
                    objectKey=element.object_key,
                    size=element.size,
                    page=None if isinstance(element, cl.Plotly) else element.page,
                    language=element.language,
                    forId=element.for_id if element.for_id else None,
                    mime=element.mime,
                )
                if element.content:
                    await self.save_chart_content_to_file(plotly_chart_file_name, element.content)

                session.add(new_element)
                await session.commit()

                logger.info(f"Element({new_element.id}) created successfully!")
            except Exception as e:
                logger.warning(f"Error in create_element: {e}")
                await session.rollback()

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        logger.info(f"get_element: {element_id}")
        async with SessionLocal() as session:
            try:
                element = await session.get(ElementModel, element_id)
                if not element:
                    logger.info(f"Element {element_id} not found")
                    return None
                return ElementDict(
                    id=element.id,
                    threadId=element.threadId,
                    type=element.type,
                    chainlitKey=element.chainlitKey,
                    url=element.url,
                    objectKey=element.objectKey,
                    name=element.name,
                    display=element.display,
                    size=element.size,
                    language=element.language,
                    page=element.page,
                    autoPlay=None,
                    playerConfig=None,
                    forId=element.forId,
                    mime=element.mime,
                )
            except Exception as e:
                logger.warning(f"Error in get_element: {e}")
                return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        logger.info(f"delete_element: {element_id}")
        async with SessionLocal() as session:
            try:
                await session.execute(
                    text('DELETE FROM elements WHERE id = :element_id'),
                    {"element_id": element_id},
                )
                await session.commit()
            except Exception as e:
                logger.warning(f"Error in delete_element: {e}")

    async def update_step(self, step_dict: StepDict):
        step_id = step_dict["id"]
        step_name = step_dict["name"]
        step_type = step_dict["type"]
        step_input = step_dict.get("input")
        step_output = step_dict.get("output")
        step_parent_id = step_dict.get("parentId")
        step_metadata = step_dict.get("metadata")
        step_tags = step_dict.get("tags")
        logger.info(f'update_step({step_id})\nname: {step_name}\ntype: {step_type}\ninput: {step_input}\noutput: {step_output}\nparentId: {step_parent_id}')
        if not step_type or step_type == "run":
            logger.info(f"Skip updating step!")
            return

        async with SessionLocal() as session:
            try:
                step = await session.get(Step, step_id)
                if not step:
                    logger.info(f"Step with id {step_id} not found!")
                    return
                else:
                    if step_name:
                        step.name = step_name
                    if step_type:
                        step.type = step_type
                    if step_input:
                        step.input = step_input
                    if step_output:
                        step.output = step_output
                    if step_parent_id:
                        step.parentId = step_parent_id
                    if step_metadata:
                        step.metadata_json = step_metadata
                    if step_tags:
                        step.tags = step_tags

                    await session.commit()
            except Exception as e:
                logger.warning(f"Error in update_step: {e}")

    async def delete_feedback(self, feedback_id: str) -> bool:
        async with SessionLocal() as session:
            try:
                feedback = await session.get(FeedbackModel, feedback_id)
                if not feedback:
                    logger.info(f"Feedback({feedback_id}) not found!")
                    return False

                await session.delete(feedback)
                await session.commit()
                logger.info(f"Feedback({feedback_id}) deleted successfully.")
                return True
            except Exception as e:
                logger.warning(f"Error deleting feedback {feedback_id}: {e}")
                await session.rollback()
                return False

    async def upsert_feedback(self, feedback: Feedback) -> str:
        async with SessionLocal() as session:
            try:
                feedback_id = feedback.id if feedback.id else None
                if feedback_id:
                    existing_feedback = await session.get(FeedbackModel, feedback_id)
                    if existing_feedback:
                        existing_feedback.forId = feedback.forId
                        existing_feedback.threadId = feedback.threadId
                        existing_feedback.value = feedback.value
                        existing_feedback.comment = feedback.comment
                        await session.commit()
                        logger.info(f"Feedback({existing_feedback.id}) updated.")
                        return existing_feedback.id

                new_feedback_id = str(uuid.uuid4())
                new_obj = FeedbackModel(
                    id=new_feedback_id,
                    forId=feedback.forId,
                    threadId=feedback.threadId,
                    value=feedback.value,
                    comment=feedback.comment
                )
                session.add(new_obj)
                await session.commit()
                logger.info(f"New feedback({new_feedback_id}) inserted.")
                return new_feedback_id
            except Exception as e:
                logger.warning(f"Error in upsert_feedback: {e}")
                await session.rollback()
                return ""

    async def get_steps_by_thread(self, thread_id: str) -> List[StepDict]:
        logger.info(f"get_steps_by_thread: {thread_id}")
        async with SessionLocal() as session:
            try:
                steps_query = select(Step).where(Step.threadId == thread_id).order_by(Step.createdAt.asc())
                result = await session.execute(steps_query)
                steps = result.scalars().all()

                step_dicts = []
                for step in steps:
                    if step.type == "assistant_message":
                        feedback_query = select(FeedbackModel).where(FeedbackModel.forId == step.parentId)
                        result = await session.execute(feedback_query)
                        feedback = result.scalars().first()
                    else:
                        feedback = None
                    step_dicts.append(StepDict(
                        id=step.id,
                        createdAt=step.createdAt,
                        name=step.name,
                        type=step.type,
                        threadId=step.threadId,
                        parentId=step.parentId if step.parentId else None,
                        streaming=step.streaming,
                        waitForAnswer=step.waitForAnswer,
                        isError=step.isError,
                        metadata=parse_json_field(step.metadata_json, dict),
                        tags=parse_json_field(step.tags, list),
                        input=step.input,
                        output=step.output,
                        start=None,
                        end=None,
                        generation=None,
                        showInput=True,
                        language=None,
                        indent=None,
                        feedback=FeedbackDict(
                            forId=feedback.forId,
                            id=feedback.id,
                            value=feedback.value,
                            comment=feedback.comment
                        ) if feedback else None
                    ))
                return step_dicts
            except Exception as e:
                logger.warning(f"Error in get_steps_by_thread: {e}")
                return []

    async def get_elements_by_thread(self, thread_id: str) -> List[ElementDict]:
        logger.info(f"get_elements_by_thread: {thread_id}")
        async with SessionLocal() as session:
            try:
                elements_query = select(ElementModel).where(ElementModel.threadId == thread_id)
                result = await session.execute(elements_query)
                elements = result.scalars().all()

                elements_dicts = []
                for element in elements:
                    logger.info(f"Element name: {element.name}")
                    logger.info(f"Element type: {element.type}")
                    elements_dicts.append(ElementDict(
                        id=element.id,
                        threadId=element.threadId if element.threadId else None,
                        type=element.type,
                        chainlitKey=element.chainlitKey,
                        url=element.url,
                        objectKey=element.objectKey,
                        name=element.name,
                        display=element.display,
                        size=element.size,
                        language=element.language,
                        page=element.page,
                        autoPlay=None,
                        playerConfig=None,
                        forId=element.forId if element.forId else None,
                        mime=element.mime,
                    ))
                return elements_dicts
            except Exception as e:
                logger.warning(f"Error in get_elements_by_thread: {e}")
                return []
