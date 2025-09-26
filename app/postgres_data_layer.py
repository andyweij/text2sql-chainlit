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
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, Integer
from sqlalchemy import select
from sqlalchemy import text
from typing import List, Dict, Optional, Union

logger = get_logger("PostgresDataLayer")

# 初始化 SQLAlchemy
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
logger.info(f"DATABASE_URL: {DATABASE_URL}")

Base = declarative_base()


# 定義資料表模型
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identifier = Column(String, unique=True, nullable=False)
    createdAt = Column(String)
    metadata_json = Column(JSON, name="metadata")

    def __hash__(self):
        """基於主鍵 id 生成哈希值"""
        return hash(str(self.id))

    def __eq__(self, other):
        """比較兩個 User 對象是否相等"""
        if isinstance(other, User):
            return str(self.id) == str(other.id)
        return False

class Thread(Base):
    __tablename__ = "threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    createdAt = Column(String)
    userId = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    userIdentifier = Column(String)
    tags = Column(ARRAY(String))
    metadata_json = Column(JSON, name="metadata")

class Step(Base):
    __tablename__ = "steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    type = Column(String)
    threadId = Column(UUID(as_uuid=True), ForeignKey("threads.id"))
    tags = Column(ARRAY(String))
    input = Column(String)
    output = Column(String)
    createdAt = Column(String)
    parentId = Column(UUID(as_uuid=True))
    streaming = Column(Boolean, default=False)
    waitForAnswer = Column(Boolean, default=False)
    isError = Column(Boolean, default=False)
    metadata_json = Column(JSON, name="metadata")

class ElementModel(Base):
    __tablename__ = "elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    threadId = Column(UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"))
    type = Column(String)
    url = Column(String)
    chainlitKey = Column(String)
    name = Column(String, nullable=False)
    display = Column(String)
    objectKey = Column(String)
    size = Column(String)
    page = Column(Integer)
    language = Column(String)
    forId = Column(UUID(as_uuid=True))
    mime = Column(String)

    def __repr__(self):
        return (
            f"<Element(id={self.id}, threadId={self.threadId}, name={self.name}, "
            f"type={self.type}, url={self.url}, language={self.language})>"
        )

class FeedbackModel(Base):
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    forId = Column(UUID(as_uuid=True), nullable=False)
    threadId = Column(UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
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
        for (id_, identifier_), value in self.data.items():
            if identifier_ == identifier:
                return value
        return None

    def delete(self, user_id, identifier):
        self.data.pop((user_id, identifier), None)

# 定義自定義 Data Layer
class PostgresDataLayer(cl_data.BaseDataLayer):
    def __init__(self):
        self.users_cache = UsersCache()
        self.threads_cache = {}

    async def get_user(self, identifier: str):
        logger.info(f"get_user: {identifier}")

        # 優先從 cache 中查詢
        user = self.users_cache.get_by_identifier(identifier)
        if user:
            logger.info(f"Cache hit for user({identifier}, id: {str(user.id)})")
            return cl.PersistedUser(
                id=str(user.id),
                identifier=user.identifier,
                createdAt=user.createdAt if user.createdAt else datetime.now().isoformat(),
                metadata=user.metadata_json
            )

        # 從數據庫查詢
        async with SessionLocal() as session:
            try:
                result = await session.execute(
                    text('SELECT "id", "identifier", "createdAt", "metadata" FROM users WHERE identifier = :identifier'),
                    {"identifier": identifier},
                )
                user_row = result.fetchone()
                if not user_row:
                    logger.info(f"User {identifier} not found in database")
                    return None

                logger.info(f"Found user({identifier}, id: {str(user_row.id)}) in database.")
                # 更新 cache 並返回
                user = User(
                    id=user_row.id,
                    identifier=user_row.identifier,
                    createdAt=user_row.createdAt,
                    metadata_json=user_row.metadata
                )
                logger.info(f"set cache {type(user)}")
                self.users_cache.set(str(user.id), str(user.identifier), user)

                logger.info(f"Return user({user.identifier}, id: {str(user.id)}).")
                return cl.PersistedUser(
                    id=str(user.id),
                    identifier=user.identifier,
                    createdAt=user.createdAt if user.createdAt else datetime.now().isoformat(),
                    metadata=user.metadata_json
                )
            except Exception as e:
                logger.warning(f"Error in get_user: {e}")
                return None

    async def create_user(self, user: cl.User):
        logger.info(f"create_user: {user}")
        async with SessionLocal() as session:
            try:
                # 插入用戶到數據庫
                created_at = getattr(user, 'createdAt', None)
                new_user = User(
                    id=uuid.uuid4(),
                    identifier=user.identifier,
                    createdAt=created_at if created_at else datetime.now().isoformat(),
                    metadata_json=user.metadata
                )
                session.add(new_user)
                await session.commit()

                # 更新 cache
                self.users_cache.set(str(new_user.id), new_user.identifier, new_user)
                return cl.PersistedUser(
                    id=str(new_user.id),
                    createdAt=new_user.createdAt,
                    identifier=new_user.identifier,
                    metadata=new_user.metadata_json
                )
            except Exception as e:
                logger.warning(f"Error in create_user: {e}")
                return None

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:

        logger.info(f"list_threads: {filters}")
        async with SessionLocal() as session:
            try:
                # 使用 ORM 查詢
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
                threads = result.scalars().all()  # 返回 ORM 對象串列

                thread_dicts = []
                for thread in threads:
                    steps = await self.get_steps_by_thread(str(thread.id))
                    elements = await self.get_elements_by_thread(str(thread.id))
                    logger.info(f"Thread({thread.name}, {str(thread.id)}) has {len(steps)} steps and {len(elements)} elements.")

                    is_thread_empty = True
                    for step in steps:
                        if not step["type"] == "run":
                            is_thread_empty = False
                            break
                    #logger.info(f"{len(steps)} steps in thread - {thread.name}.")
                    #logger.info(f"{len(elements)} elements in thread - {thread.name}.")
                    if not is_thread_empty or len(elements) > 0:
                        thread_dicts.append(
                            ThreadDict(
                                id=str(thread.id),
                                createdAt=thread.createdAt,
                                name=thread.name,
                                userId=str(thread.userId),
                                userIdentifier=thread.userIdentifier,
                                tags=thread.tags if thread.tags else [],
                                metadata=thread.metadata_json or {},
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
                        logger.info(f"Create new thread({thread_id}) for user({str(current_user.identifier)})...")
                        user = await self.get_user(current_user.identifier)
                        user_id = user.id
                        await self.update_thread(
                            thread_id= thread_id,
                            name= "...",
                            user_id= str(user_id)
                        )

            logger.info(f'Create new step({step_dict["id"]})...')
            try:
                new_step = Step(
                    id=step_dict["id"],
                    name=step_dict["name"],
                    type=step_dict["type"],
                    threadId=uuid.UUID(step_dict["threadId"]),
                    input=step_dict["input"] if "input" in step_dict else None,
                    output=step_dict["output"] if "output" in step_dict else None,
                    createdAt=step_dict["createdAt"],
                    parentId=uuid.UUID(step_dict["parentId"]) if step_dict.get("parentId") else None
                )
                session.add(new_step)
                await session.commit()
                logger.info(f"Step({new_step.id}) was created successfully.")
            except Exception as e:
                logger.warning(f"Error in create_step: {e}")

    async def delete_step(self, step_id: str):
        async with SessionLocal() as session:
            try:
                # 刪除指定步驟
                await session.execute(
                    "DELETE FROM steps WHERE id = :step_id",
                    {"step_id": step_id},
                )
                await session.commit()
            except Exception as e:
                logger.warning(f"Error in delete_step: {e}")


    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        logger.info(f"get_thread: {thread_id}")
        if not thread_id:
            return None

        """
        # 優先從 cache 中查詢
        if thread_id in self.threads_cache:
            logger.info(f"Cache hit for thread: {thread_id}")
            thread = self.threads_cache.get(thread_id)
            if thread:
                steps = await self.get_steps_by_thread(thread_id)
                elements = await self.get_elements_by_thread(thread_id)
                thread['steps'] = steps
                thread['elements'] = elements
                return thread
        """

        # 從數據庫查詢線程
        async with SessionLocal() as session:
            try:
                thread = await session.get(Thread, thread_id)
                if not thread:
                    logger.info(f"Thread {thread_id} not found in the database.")
                    return None

                steps = await self.get_steps_by_thread(thread_id)
                elements = await self.get_elements_by_thread(thread.id)

                thread_dict = ThreadDict(
                    id=str(thread.id),  # 转换 UUID 为字符串
                    createdAt=thread.createdAt,
                    name=thread.name,
                    userId=str(thread.userId),
                    userIdentifier=thread.userIdentifier,
                    tags=thread.tags if thread.tags else [],
                    metadata=thread.metadata_json if isinstance(thread.metadata_json, dict) else {},
                    steps=steps,
                    elements=elements
                )

                # 更新 cache 並返回
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
                # 查找要更新的線程
                thread = await session.get(Thread, thread_id)
                if not thread:
                    logger.info(f"Thread id {thread_id} not found, create it...")

                    if user_id:
                        user = self.users_cache.get_by_id(user_id)
                        if user:
                            user_identifier = user.identifier
                        else:
                            user_identifier = None

                    new_thread = Thread(
                        id=uuid.UUID(thread_id),
                        name=name,
                        createdAt=datetime.now().isoformat(),
                        userId=user_id,
                        userIdentifier=user_identifier,
                        metadata_json=metadata,
                        tags=tags
                    )
                    session.add(new_thread)
                    await session.commit()
                    logger.info(f"Thread({str(thread_id)}) was created.")

                    thread_dict = ThreadDict(
                        id=str(new_thread.id),
                        createdAt=new_thread.createdAt,
                        name=new_thread.name,
                        userId=str(new_thread.userId),
                        userIdentifier=new_thread.userIdentifier,
                        tags=new_thread.tags if new_thread.tags else [],
                        metadata = new_thread.metadata_json if isinstance(new_thread.metadata_json, dict) else {},
                        steps=[],
                        elements=[],
                    )
                    self.threads_cache[thread_id] = thread_dict
                else:
                    if name:
                        thread.name = name
                    if user_id:
                        thread.userId = uuid.UUID(user_id)
                    if metadata:
                        thread.metadata_json = {**metadata}
                    if tags:
                        thread.tags = tags

                    # 提交更改
                    await session.commit()

                    # 更新 cache
                    thread_dict = ThreadDict(
                        id=str(thread.id),
                        createdAt=thread.createdAt,
                        name=thread.name,
                        userId=str(thread.userId),
                        userIdentifier=thread.userIdentifier,
                        tags=thread.tags if thread.tags else [],
                        metadata=thread.metadata_json if isinstance(thread.metadata_json, dict) else {},
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
                # 刪除線程
                await session.execute(
                    text('DELETE FROM threads WHERE id = :thread_id'),
                    {"thread_id": thread_id},
                )
                await session.commit()

                # 刪除 cache
                if thread_id in self.threads_cache:
                    del self.threads_cache[thread_id]
            except Exception as e:
                logger.warning(f"Error in delete_thread: {e}")

    async def get_thread_author(self, thread_id: str):
        async with SessionLocal() as session:
            try:
                # 查詢線程作者
                result = await session.execute(
                    text('SELECT "userIdentifier" FROM threads WHERE id = :thread_id'),
                    {"thread_id": thread_id},
                )
                author = result.scalar()
                return author
            except NoResultFound:
                return None
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
                logger.warning(f"Content is None or of unsupported type for file {file_path}. Skipping write.")
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
                element_id = uuid.UUID(element.id) if element.id else uuid.uuid4()
                plotly_chart_file_name = f"{str(element_id)}.json"
                plotly_chart_url = f"/public/chart/{plotly_chart_file_name}"
                new_element = ElementModel(
                    id=element_id,
                    threadId=uuid.UUID(element.thread_id),
                    type=element.type,
                    url=plotly_chart_url if isinstance(element, cl.Plotly) else element.url,
                    chainlitKey=element.chainlit_key,
                    name=element.name,
                    display=element.display,
                    objectKey=element.object_key,
                    size=element.size,
                    page=None if isinstance(element, cl.Plotly) else element.page,
                    language=element.language,
                    forId=uuid.UUID(element.for_id) if element.for_id else None,
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
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        logger.info(f"delete_element: {element_id}")
        pass

    async def update_step(self, step_dict: StepDict):
        step_id=step_dict["id"]
        step_name=step_dict["name"]
        step_type=step_dict["type"]
        step_input=step_dict["input"] if "input" in step_dict else None
        step_output=step_dict["output"] if "output" in step_dict else None
        step_parent_id=step_dict["parentId"] if "parentId" in step_dict else None
        step_metadata=step_dict["metadata"] if "metadata" in step_dict else None
        step_tags=step_dict["tags"] if "tags" in step_dict else None
        logger.info(f'update_step({step_id})\nname: {step_name}\ntype: {step_type}\ninput: {step_input}\noutput: {step_output}\nparentId: {step_parent_id}')
        if not step_type or "run" == step_type:
            logger.info(f"Skip updating step!")
            return

        async with SessionLocal() as session:
            try:
                # 查找要更新的 Step
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
                        step.parentId = uuid.UUID(step_parent_id)
                    if step_metadata:
                        step.metadata_json = step_metadata
                    if step_tags:
                        step.tags = step_tags

                    # 提交更改
                    await session.commit()

                    # 更新 cache
                    """
                    thread_dict = ThreadDict(
                        id=str(thread.id),
                        createdAt=thread.createdAt,
                        name=thread.name,
                        userId=str(thread.userId),
                        userIdentifier=thread.userIdentifier,
                        tags=thread.tags if thread.tags else [],
                        metadata=thread.metadata_json.dict() if hasattr(thread.metadata_json, "dict") else {},
                        steps=[],
                        elements=[],
                    )
                    self.threads_cache[thread_id] = thread_dict
                    """
            except Exception as e:
                logger.warning(f"Error in update_step: {e}")

    async def delete_feedback(self, feedback_id: str) -> bool:
        async with SessionLocal() as session:
            try:
                feedback_uuid = uuid.UUID(feedback_id)
            except ValueError:
                logger.warning(f"Invalid feedback_id: {feedback_id}")
                return False

            try:
                feedback_query = select(FeedbackModel).where(FeedbackModel.id == feedback_uuid)
                feedbacks = await session.execute(feedback_query)
                feedback_obj = feedbacks.scalars().first()

                if not feedback_obj:
                    logger.info(f"Feedback({feedback_id}) not found!")
                    return False

                await session.delete(feedback_obj)
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
                feedback_uuid: Optional[uuid.UUID] = None
                if feedback.id:
                    try:
                        feedback_uuid = uuid.UUID(feedback.id)
                    except ValueError:
                        logger.warning(f"Invalid feedback.id: {feedback.id}")

                if feedback_uuid:
                    feedback_query = select(FeedbackModel).where(FeedbackModel.id == feedback_uuid)
                    feedbacks = await session.execute(feedback_query)
                    existing_feedback = feedbacks.scalars().first()

                    if existing_feedback:
                        existing_feedback.forId = uuid.UUID(feedback.forId)
                        existing_feedback.threadId = uuid.UUID(feedback.threadId)
                        existing_feedback.value = feedback.value
                        existing_feedback.comment = feedback.comment
                        await session.commit()
                        logger.info(f"Feedback({existing_feedback.id}) updated.")
                        return str(existing_feedback.id)

                new_feedback_id = uuid.uuid4()
                new_obj = FeedbackModel(
                    id=new_feedback_id,
                    forId=uuid.UUID(feedback.forId),
                    threadId=uuid.UUID(feedback.threadId),
                    value=feedback.value,
                    comment=feedback.comment
                )
                session.add(new_obj)
                await session.commit()
                logger.info(f"New feedback({new_feedback_id}) inserted.")
                return str(new_feedback_id)
            except Exception as e:
                logger.warning(f"Error in upsert_feedback: {e}")
                await session.rollback()
                return ""

    async def get_steps_by_thread(self, thread_id: str) -> List[StepDict]:
        """Get steps by thread ID, returning a list of StepDict."""
        """Not a abstract method from chainlit.data.BaseDataLayer"""

        logger.info(f"get_steps_by_thread: {thread_id}")
        # 使用 ORM 查詢 steps
        async with SessionLocal() as session:
            try:
                steps_query = select(Step).where(Step.threadId == thread_id).order_by(Step.createdAt.asc())
                result = await session.execute(steps_query)
                steps = result.scalars().all()

                # 將 ORM 實例轉成 StepDict
                step_dicts = []
                for step in steps:
                    if step.type == "assistant_message":
                        feedback_query = select(FeedbackModel).where(FeedbackModel.forId == step.parentId)
                        result = await session.execute(feedback_query)
                        feedback = result.scalars().first()
                    else:
                        feedback = None
                    step_dicts.append(StepDict(
                        id=str(step.id),
                        createdAt=step.createdAt,
                        name=step.name,
                        type=step.type,
                        threadId=str(step.threadId),
                        parentId=str(step.parentId) if step.parentId else None,
                        streaming=step.streaming,
                        waitForAnswer=step.waitForAnswer,
                        isError=step.isError,
                        metadata=step.metadata_json or {},
                        tags=step.tags or [],
                        input=step.input,
                        output=step.output,
                        start=None,
                        end=None,
                        generation=None,
                        showInput=True,
                        language=None,
                        indent=None,
                        feedback=FeedbackDict(
                            forId=str(feedback.forId),
                            id=str(feedback.id),
                            value=feedback.value,
                            comment=feedback.comment
                        ) if feedback else None
                    ))
                return step_dicts
            except Exception as e:
                logger.warning(f"Error in get_steps_by_thread: {e}")

    async def get_elements_by_thread(self, thread_id: str) -> List[ElementDict]:
        """Get elements by thread ID, returning a list of ElementDict."""
        """Not a abstract method from chainlit.data.BaseDataLayer"""

        logger.info(f"get_elements_by_thread: {thread_id}")
        # 使用 ORM 查詢 steps
        async with SessionLocal() as session:
            try:
                # 查询符合 thread_id 的元素
                elements_query = select(ElementModel).where(ElementModel.threadId == thread_id)
                result = await session.execute(elements_query)
                elements = result.scalars().all()

                elements_dicts = []
                for element in elements:
                    logger.info(f"Element name: {element.name}")
                    logger.info(f"Element type: {element.type}")
                    logger.info(f"{element}")

                    elements_dicts.append(ElementDict(
                        id=str(element.id),
                        threadId=str(element.threadId) if element.threadId else None,
                        type=element.type,
                        chainlitKey=element.chainlitKey,
                        url=element.url,
                        objectKey=element.objectKey,
                        name=element.name,
                        display=element.display,
                        size=element.size,
                        language=element.language,
                        page=element.page,
                        autoPlay=None,  # 默认值为 None
                        playerConfig=None,  # 默认值为 None
                        forId=str(element.forId) if element.forId else None,
                        mime=element.mime,
                    ))

                return elements_dicts
            except Exception as e:
                logger.warning(f"Error in get_steps_by_thread: {e}")
