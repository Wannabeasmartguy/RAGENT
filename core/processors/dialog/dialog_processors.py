from typing import Optional, Dict, Any, List, Type
from enum import IntEnum
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
import time
from loguru._logger import Logger
import streamlit as st

from core.strategy import BaseDialogProcessStrategy
from core.storage.db.base import Sqlstorage
from core.models.memory import AssistantRun
from utils.log.logger_config import *


class OperationPriority(IntEnum):
    """对话处理器的操作优先级"""
    NORMAL = 0
    HIGH = 1  # 高优先级操作将跳过防抖


class Operation:
    def __init__(self, method, priority=OperationPriority.NORMAL, *args, **kwargs):
        self.method = method
        self.priority = priority
        self.args = args
        self.kwargs = kwargs


class BaseDialogProcessor(BaseDialogProcessStrategy):
    """
    对话处理器，用于管理对话相关的数据库操作
    """
    def __init__(
        self,
        storage: Sqlstorage,
        debounce_delay: float = 0.5,
        max_queue_size: int = 100,
        logger: Logger = logger
    ):
        self.storage = storage
        self.debounce_delay = debounce_delay
        self.operation_queue = Queue(maxsize=max_queue_size)
        self.last_operation_time = 0.0
        self.lock = Lock()
        
        # 启动处理线程
        self.processing_thread = Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # logger
        self._logger = logger
    
    def _process_queue(self):
        """处理后端操作队列"""
        while True:
            try:
                # 获取下一个操作
                operation = self.operation_queue.get()
                if operation is None:
                    break
                
                # 解包操作信息
                method, args, kwargs = operation.method, operation.args, operation.kwargs
                
                # 执行操作
                with self.lock:
                    current_time = time.time()
                    time_diff = current_time - self.last_operation_time
                    
                    # 高优先级操作跳过防抖检查
                    if time_diff >= self.debounce_delay or operation.priority == OperationPriority.HIGH:
                        try:
                            method(*args, **kwargs)
                            self.last_operation_time = current_time
                            self._logger.info(f"Successfully executed operation: {method.__name__}")
                        except Exception as e:
                            self._logger.error(f"Error executing operation: {e}")
                    else:
                        self._logger.debug(f"Operation {method.__name__} debounced")
                
                self.operation_queue.task_done()
                
            except Exception as e:
                self._logger.error(f"Error in operation processing thread: {e}")
    
    def _enqueue_operation(self, method, priority=OperationPriority.NORMAL, *args, **kwargs):
        """将操作添加到队列"""
        try:
            operation = Operation(method, priority, *args, **kwargs)
            self.operation_queue.put(operation)
            self._logger.debug(f"Operation {method.__name__} enqueued with priority {priority}")
        except Exception as e:
            self._logger.error(f"Error enqueueing operation: {e}")
            raise
    
    def shutdown(self):
        """关闭处理器"""
        self.operation_queue.put(None)
        self.processing_thread.join()


class ClassicChatDialogProcessor(BaseDialogProcessor):
    """
    对话处理器，用于管理对话相关的数据库操作
    """
    def __init__(
        self,
        storage: Sqlstorage,
        debounce_delay: float = 0.5,
        max_queue_size: int = 100,
        logger: Logger = logger
    ):
        super().__init__(
            storage=storage,
            debounce_delay=debounce_delay,
            max_queue_size=max_queue_size,
            logger=logger
        )
    
    def update_dialog_name(self, *, run_id: str, user_id: str, new_name: str):
        """更新对话名称"""
        def _update():
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    user_id=user_id,
                    run_name=new_name,
                    updated_at=datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def update_dialog_config(
        self,
        *,
        run_id: str,
        user_id: str,
        llm_config: Dict[str, Any],
        run_data: Optional[Dict[str, Any]] = None,
        assistant_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None
    ):
        """更新对话配置"""
        def _update():
            current_run = self.storage.get_specific_run(run_id, user_id)
            if not current_run:
                raise ValueError(f"Dialog with run_id {run_id} not found")
            
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    user_id=user_id,
                    name=current_run.name,
                    run_name=current_run.run_name,
                    llm=llm_config,
                    memory=current_run.memory,
                    run_data=run_data or current_run.run_data,
                    assistant_data=assistant_data or current_run.assistant_data,
                    task_data=task_data or current_run.task_data,
                    updated_at=updated_at or datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def update_chat_history(
        self,
        *,
        run_id: str,
        user_id: str,
        chat_history: List[Dict[str, Any]],
        assistant_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        run_data: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None
    ):
        """更新对话历史"""
        def _update():
            current_run = self.storage.get_specific_run(run_id, user_id)
            if not current_run:
                raise ValueError(f"Dialog with run_id {run_id} not found")
            
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    user_id=user_id,
                    name=current_run.name,
                    run_name=current_run.run_name,
                    llm=current_run.llm,
                    memory={"chat_history": chat_history},
                    assistant_data=assistant_data or current_run.assistant_data,
                    task_data=task_data or current_run.task_data,
                    run_data=run_data or current_run.run_data,
                    updated_at=updated_at or datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def create_dialog(
        self,
        *,
        run_id: str,
        user_id: str,
        run_name: str,
        llm_config: Dict[str, Any],
        name: str = "assistant",
        run_data: Optional[Dict[str, Any]] = None,
        assistant_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        priority: OperationPriority = OperationPriority.NORMAL
    ):
        """创建新对话"""
        def _create():
            try:
                self._logger.debug(f"Creating new dialog with run_id: {run_id}")
                self._logger.debug(f"Dialog config: {llm_config}")
                
                self.storage.upsert(
                    AssistantRun(
                        name=name,
                        run_id=run_id,
                        user_id=user_id,
                        run_name=run_name,
                        llm=llm_config,
                        memory={"chat_history": []},
                        run_data=run_data,
                        assistant_data=assistant_data,
                        task_data=task_data,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                )
                self._logger.info(f"Successfully created dialog: {run_name} ({run_id})")
            except Exception as e:
                self._logger.error(f"Failed to create dialog: {e}")
                raise
        
        self._enqueue_operation(_create, priority=priority)
    
    def delete_dialog(self, run_id: str, user_id: str):
        """删除对话"""
        def _delete():
            self.storage.delete_run(run_id, user_id)
        self._enqueue_operation(_delete)
    
    def get_dialog(self, run_id: str, user_id: Optional[str] = None) -> Optional[AssistantRun]:
        """获取对话（同步操作）"""
        with self.lock:
            return self.storage.get_specific_run(
                run_id=run_id,
                user_id=user_id
            )
    
    def get_all_dialogs(self, user_id: Optional[str] = None, debug_mode: bool = False) -> List[AssistantRun]:
        """获取所有对话（同步操作）"""
        try:
            # 等待所有操作完成
            self.operation_queue.join()
            
            with self.lock:
                dialogs = self.storage.get_all_runs(
                    user_id=user_id,
                    debug_mode=debug_mode
                )
                self._logger.debug(f"Retrieved {len(dialogs)} dialogs")
                return dialogs
        except Exception as e:
            self._logger.error(f"Error getting dialogs: {e}")
            return []


class RAGChatDialogProcessor(ClassicChatDialogProcessor):
    """RAG聊天对话处理器"""
    def __init__(
        self,
        storage: Sqlstorage,
        debounce_delay: float = 0.5,
        max_queue_size: int = 100,
        logger: Logger = logger
    ):
        super().__init__(
            storage=storage,
            debounce_delay=debounce_delay,
            max_queue_size=max_queue_size,
            logger=logger
        )

    # 添加对RAG聊天对话中保存向量数据库相关数据的操作
    def update_knowledge_base_config(
        self, 
        *, 
        run_id: str, 
        user_id: str,
        knowledge_base_config: Dict[str, Any]
    ):
        """
        更新当前对话指向的知识库配置
        
        Args:
            run_id (str): 对话ID
            knowledge_base_config (Dict[str, Any]): 知识库配置
                collection_name (str): 当前对话指向的知识库名称
                is_rerank (bool): 是否启用重排序
                is_hybrid_retrieve (bool): 是否启用混合检索
                hybrid_retrieve_weight (float): 混合检索权重
                query_mode (str): 查询模式 ("collection" 或 "file")
                selected_file (str, optional): 单文件模式下选择的文件
        """
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")
                
                # 更新run_data中的knowledge_base_config
                current_run_data = current_run.run_data or {}
                current_run_data["knowledge_base_config"] = knowledge_base_config
                
                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id,
                        user_id=user_id,
                        name=current_run.name,
                        run_name=current_run.run_name,
                        llm=current_run.llm,
                        memory=current_run.memory,
                        run_data=current_run_data,
                        assistant_data=current_run.assistant_data,
                        task_data=current_run.task_data,
                        updated_at=datetime.now()
                    )
                )
                self._logger.info(f"Successfully updated knowledge base config for dialog: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update knowledge base config: {e}")
                raise
                
        self._enqueue_operation(_update)
    
    def get_knowledge_base_config(self, run_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取当前对话指向的知识库配置"""
        with self.lock:
            try:
                run = self.storage.get_specific_run(run_id, user_id)
                if run and run.run_data:
                    return run.run_data.get("knowledge_base_config", {})
                return {}
            except Exception as e:
                self._logger.error(f"Error getting knowledge base config: {e}")
                return {}


class AgenChatDialogProcessor(BaseDialogProcessor):
    """AgentChat对话处理器"""
    def __init__(
        self,
        storage: Sqlstorage,
        debounce_delay: float = 0.5,
        max_queue_size: int = 100,
        logger: Logger = logger
    ):
        super().__init__(
            storage=storage,
            debounce_delay=debounce_delay,
            max_queue_size=max_queue_size,
            logger=logger
        )
    
    def create_dialog(
        self,
        *,
        run_id: str,
        user_id: str,
        run_name: str,
        template: Optional[Dict[str, Any]] = None,
        name: str = "assistant",
        team_state: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict[str, Any]] = None,
        run_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        priority: OperationPriority = OperationPriority.NORMAL
    ):
        """创建新对话"""
        def _create():
            try:
                self._logger.debug(f"Creating new dialog with run_id: {run_id}")
                self._logger.debug(f"Agent/Team config: {template}")
                
                assistant_data = {
                    "template": template,
                    "team_state": team_state if team_state else {},
                    "agent_state": agent_state if agent_state else {}
                }
                self.storage.upsert(
                    AssistantRun(
                        name=name,
                        run_id=run_id,
                        run_name=run_name,
                        user_id=user_id,
                        llm=template.get("llm", {}),
                        memory={"chat_history": []},
                        run_data=run_data,
                        assistant_data=assistant_data,
                        task_data=task_data,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                )
                self._logger.info(f"Successfully created dialog: {run_name} ({run_id})")
            except Exception as e:
                self._logger.error(f"Failed to create dialog: {e}")
                raise
        
        self._enqueue_operation(_create, priority=priority)

    def delete_dialog(self, run_id: str, user_id: str):
        """删除对话"""
        def _delete():
            self.storage.delete_run(run_id, user_id)
        self._enqueue_operation(_delete)

    def get_dialog(self, run_id: str, user_id: Optional[str] = None) -> Optional[AssistantRun]:
        """获取对话（同步操作）"""
        with self.lock:
            return self.storage.get_specific_run(
                run_id=run_id,
                user_id=user_id
            )
        
    def get_all_dialogs(self, user_id: Optional[str] = None, debug_mode: bool = False) -> List[AssistantRun]:
        """获取所有对话（同步操作）"""
        try:
            # 等待所有操作完成
            self.operation_queue.join()
            
            with self.lock:
                dialogs = self.storage.get_all_runs(
                    user_id=user_id,
                    debug_mode=debug_mode
                )
                self._logger.debug(f"Retrieved {len(dialogs)} dialogs")
                return dialogs
        except Exception as e:
            self._logger.error(f"Error getting dialogs: {e}")
            return []

    def update_dialog_name(self, *, run_id: str, user_id: str, new_name: str):
        """更新对话名称"""
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")
            
                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id,
                        user_id=user_id,
                        run_name=new_name,
                        updated_at=datetime.now()
                    )
                )
                self._logger.info(f"Successfully updated dialog name for: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update dialog name: {e}")
                raise
        self._enqueue_operation(_update)

    def update_template(
        self, 
        *, 
        run_id: str, 
        user_id: str,
        template: Dict[str, Any]
    ):
        """
        更新当前对话指向的 Agent team 模板

        Args:
            run_id (str): 对话ID
            template (Dict[str, Any]): 模板
        """
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")

                # 更新run_data中的template
                current_assistant_data = current_run.assistant_data or {}
                current_assistant_data["template"] = template

                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id,
                        name=current_run.name,
                        run_name=current_run.run_name,
                        llm=current_run.llm,
                        memory=current_run.memory,
                        run_data=current_run.run_data,
                        assistant_data=current_assistant_data,
                        task_data=current_run.task_data,
                        updated_at=datetime.now()
                    )
                )
                self._logger.info(f"Successfully updated template for dialog: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update template: {e}")
                raise
            
        self._enqueue_operation(_update)

    def get_template(self, run_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取当前对话指向的 Agent team 模板"""
        with self.lock:
            try:
                run = self.storage.get_specific_run(run_id, user_id)
                if run and run.assistant_data:
                    return run.assistant_data.get("template", {})
                return {}
            except Exception as e:
                self._logger.error(f"Error getting template: {e}")
                return {}
    
    def update_team_state(self, run_id: str, user_id: str, team_state: Dict[str, Any]):
        """更新团队状态"""
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")
            
                current_assistant_data = current_run.assistant_data or {}
                current_assistant_data["team_state"] = team_state
                
                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id, 
                        user_id=user_id,
                        assistant_data=current_assistant_data
                    )
                )
                self._logger.info(f"Successfully updated team state for dialog: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update team state: {e}")
                raise
        
        self._enqueue_operation(_update)

    def get_team_state(self, run_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取团队状态"""
        with self.lock:
            run = self.storage.get_specific_run(run_id, user_id)
            if run and run.assistant_data:
                return run.assistant_data.get("team_state", {})
            return {}
    
    def update_run_name(self, run_id: str, user_id: str, run_name: str):
        """更新对话名称"""
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")
                
                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id, 
                        user_id=user_id, 
                        run_name=run_name
                    )
                )
                self._logger.info(f"Successfully updated run name for: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update run name: {e}")
                raise
        self._enqueue_operation(_update)

    def update_template_and_team_state(self, run_id: str, user_id: str, template: Dict[str, Any], team_state: Dict[str, Any]):
        """更新对话模板和团队状态"""
        def _update():
            try:
                current_run = self.storage.get_specific_run(run_id, user_id)
                if not current_run:
                    raise ValueError(f"Dialog with run_id {run_id} not found")
                
                current_assistant_data = current_run.assistant_data or {}
                current_assistant_data["template"] = template
                current_assistant_data["team_state"] = team_state

                self.storage.upsert(
                    AssistantRun(
                        run_id=run_id,
                        user_id=user_id,
                        assistant_data=current_assistant_data
                    )
                )
                self._logger.info(f"Successfully updated template and team state for dialog: {run_id}")
            except Exception as e:
                self._logger.error(f"Failed to update template and team state: {e}")
                raise
        self._enqueue_operation(_update)
