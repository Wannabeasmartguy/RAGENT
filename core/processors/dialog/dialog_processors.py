from typing import Optional, Dict, Any, List, Type
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
import time
from loguru import logger
from loguru._logger import Logger

from core.strategy import DialogProcessStrategy
from core.storage.db.base import Sqlstorage
from core.model.chat.assistant import AssistantRun


class DialogProcessor(DialogProcessStrategy):
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
        self.last_operation_time = 0
        self.lock = Lock()
        
        # 启动处理线程
        self.processing_thread = Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # logger
        self.logger = logger
    
    def _process_queue(self):
        """处理操作队列的后台线程"""
        while True:
            try:
                # 获取下一个操作
                operation = self.operation_queue.get()
                if operation is None:
                    break
                
                # 解包操作信息
                method, args, kwargs = operation
                
                # 执行操作
                with self.lock:
                    current_time = time.time()
                    # 检查防抖
                    if current_time - self.last_operation_time >= self.debounce_delay:
                        try:
                            method(*args, **kwargs)
                            self.last_operation_time = current_time
                            self.logger.info(f"Successfully executed operation: {method.__name__}")
                        except Exception as e:
                            self.logger.error(f"Error executing operation {method.__name__}: {e}")
                    else:
                        self.logger.debug(f"Operation {method.__name__} debounced")
                
                self.operation_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in operation processing thread: {e}")
    
    def _enqueue_operation(self, method, *args, **kwargs):
        """将操作添加到队列"""
        try:
            self.operation_queue.put((method, args, kwargs))
            self.logger.debug(f"Operation {method.__name__} enqueued")
        except Exception as e:
            self.logger.error(f"Error enqueueing operation: {e}")
            raise
    
    def update_dialog_name(self, run_id: str, new_name: str):
        """更新对话名称"""
        def _update():
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    run_name=new_name,
                    updated_at=datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def update_dialog_config(
        self,
        run_id: str,
        llm_config: Dict[str, Any],
        assistant_data: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None
    ):
        """更新对话配置"""
        def _update():
            current_run = self.storage.get_specific_run(run_id)
            if not current_run:
                raise ValueError(f"Dialog with run_id {run_id} not found")
            
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    name=current_run.name,
                    run_name=current_run.run_name,
                    llm=llm_config,
                    memory=current_run.memory,
                    assistant_data=assistant_data or current_run.assistant_data,
                    updated_at=updated_at or datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def update_chat_history(
        self,
        run_id: str,
        chat_history: List[Dict[str, Any]],
        assistant_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None
    ):
        """更新对话历史"""
        def _update():
            current_run = self.storage.get_specific_run(run_id)
            if not current_run:
                raise ValueError(f"Dialog with run_id {run_id} not found")
            
            self.storage.upsert(
                AssistantRun(
                    run_id=run_id,
                    name=current_run.name,
                    run_name=current_run.run_name,
                    llm=current_run.llm,
                    memory={"chat_history": chat_history},
                    assistant_data=assistant_data or current_run.assistant_data,
                    task_data=task_data or current_run.task_data,
                    updated_at=updated_at or datetime.now()
                )
            )
        self._enqueue_operation(_update)
    
    def create_dialog(
        self,
        run_id: str,
        run_name: str,
        llm_config: Dict[str, Any],
        name: str = "assistant",
        assistant_data: Optional[Dict[str, Any]] = None,
        task_data: Optional[Dict[str, Any]] = None
    ):
        """创建新对话"""
        def _create():
            self.storage.upsert(
                AssistantRun(
                    name=name,
                    run_id=run_id,
                    run_name=run_name,
                    llm=llm_config,
                    memory={"chat_history": []},
                    assistant_data=assistant_data,
                    task_data=task_data,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            )
        self._enqueue_operation(_create)
    
    def delete_dialog(self, run_id: str):
        """删除对话"""
        def _delete():
            self.storage.delete_run(run_id)
        self._enqueue_operation(_delete)
    
    def get_dialog(self, run_id: str) -> Optional[AssistantRun]:
        """获取对话（同步操作）"""
        with self.lock:
            return self.storage.get_specific_run(run_id)
    
    def get_all_dialogs(self) -> List[AssistantRun]:
        """获取所有对话（同步操作）"""
        with self.lock:
            return self.storage.get_all_runs()
    
    def shutdown(self):
        """关闭处理器"""
        self.operation_queue.put(None)
        self.processing_thread.join() 