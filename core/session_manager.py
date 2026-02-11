"""
会话管理器 - 管理AI助手的上下文会话
负责会话的创建、删除、查看、切换，以及会话记录的存储
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class SessionStatus(Enum):
    """会话状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class SessionStep:
    """会话执行步骤"""
    step_number: int
    timestamp: str
    user_input: str
    llm_response: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    is_completed: bool
    final_message: Optional[str] = None
    thinking: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "llm_response": self.llm_response,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_completed": self.is_completed,
            "final_message": self.final_message,
            "thinking": self.thinking
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionStep':
        """从字典创建"""
        return cls(
            step_number=data["step_number"],
            timestamp=data["timestamp"],
            user_input=data["user_input"],
            llm_response=data["llm_response"],
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            is_completed=data["is_completed"],
            final_message=data.get("final_message"),
            thinking=data.get("thinking")
        )


@dataclass
class SessionRecord:
    """会话记录"""
    session_id: str
    status: str
    created_at: str
    updated_at: str
    workdir: str
    questions: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workdir": self.workdir,
            "questions": self.questions,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """从字典创建"""
        return cls(
            session_id=data["session_id"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            workdir=data["workdir"],
            questions=data.get("questions", []),
            summary=data.get("summary")
        )


class SessionManager:
    """会话管理器"""

    def __init__(self, storage_dir: str = None):
        """
        初始化会话管理器

        Args:
            storage_dir: 会话存储目录（默认: 当前目录/.sessions）
        """
        if storage_dir is None:
            storage_dir = os.path.join(os.getcwd(), ".sessions")

        self.storage_dir = storage_dir
        self.sessions_file = os.path.join(self.storage_dir, "sessions.json")
        self._ensure_storage_dir()
        self._sessions: Dict[str, SessionRecord] = {}
        self._current_session_id: Optional[str] = None

        self._load_sessions()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_sessions(self):
        """加载会话数据"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "sessions" in data:
                        self._sessions = {
                            sid: SessionRecord.from_dict(sdata)
                            for sid, sdata in data.get("sessions", {}).items()
                        }
                        self._current_session_id = data.get("current_session_id")
                    else:
                        self._sessions = {
                            sid: SessionRecord.from_dict(sdata)
                            for sid, sdata in data.items()
                        }
                        self._current_session_id = None
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 加载会话数据失败: {e}")
                self._sessions = {}
                self._current_session_id = None
        else:
            self._sessions = {}
            self._current_session_id = None

    def _save_sessions(self):
        """保存会话数据"""
        data = {
            "sessions": {
                sid: session.to_dict()
                for sid, session in self._sessions.items()
            },
            "current_session_id": self._current_session_id
        }
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_session(self) -> str:
        """
        创建新会话

        Returns:
            session_id: 新会话的ID
        """
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        workdir = os.getcwd()

        for sid in self._sessions:
            self._sessions[sid].status = SessionStatus.ARCHIVED.value

        session = SessionRecord(
            session_id=session_id,
            status=SessionStatus.ACTIVE.value,
            created_at=timestamp,
            updated_at=timestamp,
            workdir=workdir
        )

        self._sessions[session_id] = session
        self._current_session_id = session_id
        self._save_sessions()

        return session_id

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功删除
        """
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        if self._current_session_id == session_id:
            self._current_session_id = None

        self._save_sessions()
        return True

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            SessionRecord: 会话记录，不存在返回None
        """
        return self._sessions.get(session_id)

    def get_current_session(self) -> Optional[SessionRecord]:
        """
        获取当前会话

        Returns:
            SessionRecord: 当前会话记录，不存在返回None
        """
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    def list_sessions(self, status: SessionStatus = None) -> List[SessionRecord]:
        """
        列出所有会话

        Args:
            status: 状态过滤（可选）

        Returns:
            List[SessionRecord]: 会话列表
        """
        sessions = list(self._sessions.values())

        if status:
            sessions = [s for s in sessions if s.status == status.value]

        return sorted(sessions, key=lambda x: x.created_at, reverse=True)

    def switch_session(self, session_id: str) -> bool:
        """
        切换到指定会话

        Args:
            session_id: 目标会话ID

        Returns:
            bool: 是否成功切换
        """
        if session_id not in self._sessions:
            return False

        for sid in self._sessions:
            if sid != session_id:
                self._sessions[sid].status = SessionStatus.ARCHIVED.value

        self._current_session_id = session_id
        self._sessions[session_id].status = SessionStatus.ACTIVE.value
        self._sessions[session_id].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_sessions()
        return True

    def archive_session(self, session_id: str) -> bool:
        """
        归档会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功归档
        """
        if session_id not in self._sessions:
            return False

        self._sessions[session_id].status = SessionStatus.ARCHIVED.value
        self._sessions[session_id].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self._current_session_id == session_id:
            self._current_session_id = None

        self._save_sessions()
        return True

    def activate_session(self, session_id: str) -> bool:
        """
        激活会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功激活
        """
        if session_id not in self._sessions:
            return False

        self._sessions[session_id].status = SessionStatus.ACTIVE.value
        self._sessions[session_id].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._current_session_id = session_id
        self._save_sessions()
        return True

    def add_question(
        self,
        session_id: str,
        question: str,
        steps: List[SessionStep],
        summary: str = None
    ) -> bool:
        """
        添加问题到会话

        Args:
            session_id: 会话ID
            question: 用户问题
            steps: 执行步骤
            summary: 总结

        Returns:
            bool: 是否成功添加
        """
        if session_id not in self._sessions:
            return False

        question_record = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "steps": [step.to_dict() for step in steps],
            "summary": summary
        }

        self._sessions[session_id].questions.append(question_record)

        if summary:
            self._sessions[session_id].summary = summary

        self._sessions[session_id].updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_sessions()
        return True

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话详细信息

        Args:
            session_id: 会话ID

        Returns:
            Dict: 会话详细信息
        """
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "workdir": session.workdir,
            "questions_count": len(session.questions),
            "questions": session.questions,
            "summary": session.summary,
            "is_current": session_id == self._current_session_id
        }

    def get_or_restore_session(self) -> str:
        """
        获取或恢复会话
        如果当前有激活的会话，则使用它；否则查找已有会话或创建新会话
        启动程序时会自动恢复之前激活的会话

        Returns:
            session_id: 当前会话ID
        """
        if self._current_session_id and self._current_session_id in self._sessions:
            session = self._sessions[self._current_session_id]
            if session.status == SessionStatus.ACTIVE.value:
                return self._current_session_id

        active_sessions = self.list_sessions(SessionStatus.ACTIVE)
        if active_sessions:
            self._current_session_id = active_sessions[0].session_id
            self._sessions[self._current_session_id].status = SessionStatus.ACTIVE.value
            self._save_sessions()
            return self._current_session_id

        return self.create_session()

    def auto_create_or_switch(self) -> str:
        """
        自动创建或切换到会话
        如果当前有激活的会话，则使用它；否则创建新会话

        Returns:
            session_id: 当前会话ID
        """
        return self.get_or_restore_session()

    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """
        获取会话摘要（默认返回当前激活会话）

        Args:
            session_id: 会话ID，不指定则返回当前会话

        Returns:
            Dict: 会话摘要信息
        """
        if session_id is None:
            session_id = self._current_session_id

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            question_summaries = []
            for i, q in enumerate(session.questions):
                question_summaries.append({
                    "index": i,
                    "question": q["question"],
                    "timestamp": q["timestamp"],
                    "summary": q.get("summary", ""),
                    "steps_count": len(q.get("steps", []))
                })

            return {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "questions_count": len(session.questions),
                "question_summaries": question_summaries,
                "is_current": session_id == self._current_session_id
            }

        return {
            "session_id": None,
            "status": None,
            "questions_count": 0,
            "question_summaries": [],
            "message": "没有活动的会话"
        }

    def get_step_detail(self, step_number: int) -> Dict[str, Any]:
        """
        获取指定步骤的详细信息

        Args:
            step_number: 步骤编号（从0开始）

        Returns:
            Dict: 步骤详细信息
        """
        current_session = self._current_session_id
        if not current_session or current_session not in self._sessions:
            return {"error": "没有活动的会话"}

        session = self._sessions[current_session]

        all_steps = []
        for q in session.questions:
            steps = q.get("steps", [])
            for step in steps:
                all_steps.append({
                    "question": q["question"],
                    "step": step
                })

        if step_number < 0 or step_number >= len(all_steps):
            return {"error": f"步骤 {step_number} 不存在，有效范围 0-{len(all_steps)-1}"}

        item = all_steps[step_number]
        step_data = item["step"]

        tool_calls = step_data.get("tool_calls", [])
        tool_results = step_data.get("tool_results", [])

        result_info = []
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("function", {}).get("name", tc.get("name", "unknown"))
            result = tool_results[i] if i < len(tool_results) else None
            if result:
                result_info.append(f"- {tool_name}: {result.get('content', result.get('message', ''))}")
            else:
                result_info.append(f"- {tool_name}: (无结果)")

        return {
            "step_number": step_number,
            "question": item["question"],
            "timestamp": step_data["timestamp"],
            "is_completed": step_data["is_completed"],
            "final_message": step_data.get("final_message"),
            "tool_calls_count": len(tool_calls),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "result_summary": "\n".join(result_info) if result_info else None
        }

