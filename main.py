"""
会话感知Agent引擎 - 集成会话管理的Agent
"""
import os
import argparse
from typing import Optional, Iterator, Dict, Any

from config import MODEL_CONFIG
from core.llm_engine import LLMFactory
from core.tool_executor import ToolExecutor
from core.agent_engine import AgentEngine, AgentConfig
from core.file_manager import FileManager
from core.session_manager import SessionManager, SessionStep


class SessionAwareCodeAssistant:
    """会话感知代码助手"""

    def __init__(self, workdir: str = None):
        self.workdir = os.getcwd() if workdir is None else workdir
        self.backup_dir = os.path.join(self.workdir, "backups")
        self.file_manager = FileManager(self.workdir, self.backup_dir)
        self.session_manager = SessionManager()

        self.llm_factory = LLMFactory()
        self.llm_engine = None
        self.tool_executor = None
        self.agent = None

    def setup_engine(self):
        """初始化LLM引擎"""
        self.llm_engine = self.llm_factory.create_engine(
            producer=MODEL_CONFIG.get("producer", ""),
            model=MODEL_CONFIG.get("model", ""),
            enable_thinking=MODEL_CONFIG.get("enable_thinking", ""),
        )

        self.tool_executor = ToolExecutor(
            file_manager=self.file_manager,
            workdir=self.workdir,
            session_manager=self.session_manager
        )

    def _convert_agent_steps_to_session_steps(self, agent_steps: list) -> list:
        """将Agent步骤转换为会话步骤"""
        session_steps = []
        for step in agent_steps:
            tool_results = []
            for result in step.tool_results:
                tool_results.append({
                    "tool_name": getattr(result, 'tool_name', 'unknown'),
                    "success": getattr(result, 'success', False),
                    "content": getattr(result, 'content', ''),
                    "error": getattr(result, 'error', '')
                })

            session_step = SessionStep(
                step_number=step.step_number,
                timestamp=step.timestamp,
                user_input=step.user_input,
                llm_response=step.llm_response,
                tool_calls=step.tool_calls,
                tool_results=tool_results,
                is_completed=step.is_completed,
                final_message=step.final_message,
                thinking=getattr(step, 'thinking', None)
            )
            session_steps.append(session_step)

        return session_steps

    def _get_session_context_for_prompt(self) -> str:
        """获取会话摘要用于提示词"""
        summary = self.session_manager.get_session_summary()

        if summary.get("session_id") is None:
            return ""

        if summary["questions_count"] == 0:
            return "当前会话暂无历史记录。"

        lines = ["已完成的问题和结果:", ""]

        for qs in summary["question_summaries"]:
            lines.append(f"[{qs['index']}] 问题: {qs['question']}")
            if qs['summary']:
                lines.append(f"    结果: {qs['summary']}")
            lines.append("")

        return "\n".join(lines)

    def interactive(self, max_iterations: int = 200, verbose: bool = True):
        """交互模式"""
        if not self.llm_engine:
            self.setup_engine()

        assert self.llm_engine is not None, "LLM引擎未初始化"
        assert self.tool_executor is not None, "工具执行器未初始化"

        session_context = self._get_session_context_for_prompt()

        config = AgentConfig(
            max_iterations=max_iterations,
            verbose_output=verbose
        )
        self.agent = AgentEngine(
            self.llm_engine,
            self.tool_executor,
            config,
            session_summary=session_context
        )

        print("=" * 60)
        print("  AI Agent - 智能编程助手 (会话模式)")
        print("=" * 60)
        print(f"工作目录: {self.workdir}")
        print(f"最大迭代次数: {max_iterations}")

        session_id = self.session_manager.get_or_restore_session()
        current_session = self.session_manager.get_session(session_id)
        if current_session:
            if len(current_session.questions) > 0:
                print(f"恢复会话: {session_id}")
                print(f"已有问题数: {len(current_session.questions)}")
            else:
                print(f"新建会话: {session_id}")
        print("-" * 60)
        print("输入您的任务描述，或输入 'quit'/'exit' 退出")
        print("示例: 创建一个HTTP服务器")
        print("会话管理命令:")
        print("  /new           - 创建新会话")
        print("  /show          - 列出所有会话")
        print("  /switch <id>   - 切换到指定会话")
        print("  /delete <id>   - 删除会话")
        print("=" * 60)

        while True:
            try:
                print()
                user_input = input("> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("再见！")
                    break

                if user_input.startswith('/new'):
                    self._handle_new_command(user_input)
                    continue

                if user_input.startswith('/show'):
                    self._print_session_list()
                    continue

                if user_input.startswith('/switch'):
                    self._handle_switch_command(user_input)
                    continue

                if user_input.startswith('/delete'):
                    self._handle_delete_command(user_input)
                    continue

                if user_input.startswith('/'):
                    print(f"未知命令: {user_input}")
                    continue

                print("\n" + "-" * 60)
                print(f"执行任务: {user_input}")
                print("-" * 60)

                session_id = self.session_manager.get_or_restore_session()
                current_session = self.session_manager.get_session(session_id)
                if current_session:
                    print(f"当前会话: {session_id}")

                execution_steps = []
                final_summary = None

                for step in self.agent.start(user_input):
                    execution_steps.append(step)

                    if step.is_completed:
                        if step.final_message:
                            print("\n任务完成! 最终回答:")
                            print("-" * 60)
                            print(step.final_message)
                            print("-" * 60)
                            final_summary = step.final_message
                        break

                    if step.tool_calls:
                        print(f"\n执行了 {len(step.tool_calls)} 个工具调用")

                summary = self.agent.get_execution_summary()
                print(f"\n执行统计: {summary['total_steps']} 步, {summary['successful_tool_calls']}/{summary['total_tool_calls']} 工具成功")

                session_steps = self._convert_agent_steps_to_session_steps(execution_steps)

                if final_summary is None:
                    final_summary = summary.get('final_message', f"执行完成: {summary['total_steps']}步, {summary['successful_tool_calls']}个工具调用成功")

                self.session_manager.add_question(
                    session_id=session_id,
                    question=user_input,
                    steps=session_steps,
                    summary=final_summary
                )

            except KeyboardInterrupt:
                print("\n\n检测到中断信号，正在停止...")
                self.agent.stop()
                break
            except Exception as e:
                print(f"\n错误: {e}")
                continue

    def _handle_new_command(self, command: str):
        """处理创建新会话命令"""
        session_id = self.session_manager.create_session()
        print(f"✓ 创建新会话成功: {session_id}")
        self._print_session_info(session_id)

        if self.agent:
            self.agent._session_summary = self._get_session_context_for_prompt()

    def _handle_switch_command(self, command: str):
        """处理切换会话命令"""
        parts = command.split()
        if len(parts) < 2:
            print("用法: /switch <session_id>")
            return

        session_id = parts[1]

        if self.session_manager.switch_session(session_id):
            print(f"✓ 切换到会话: {session_id}")
            self._print_session_info(session_id)

            if self.agent:
                self.agent._session_summary = self._get_session_context_for_prompt()
        else:
            print(f"✗ 会话不存在: {session_id}")

    def _handle_delete_command(self, command: str):
        """处理删除会话命令"""
        parts = command.split()
        if len(parts) < 2:
            print("用法: /delete <session_id>")
            return

        session_id = parts[1]
        if self.session_manager.delete_session(session_id):
            print(f"✓ 删除会话成功: {session_id}")
        else:
            print(f"✗ 会话不存在: {session_id}")

    def _print_session_list(self):
        """打印会话列表"""
        sessions = self.session_manager.list_sessions()
        current_id = self.session_manager._current_session_id

        print("\n会话列表:")
        print("-" * 60)
        print(f"{'ID':<10} {'状态':<10} {'问题数':<8} {'更新时间':<20}")
        print("-" * 60)

        for session in sessions:
            status_icon = "*" if session.session_id == current_id else " "
            print(f"{status_icon}{session.session_id:<9} {session.status:<10} {len(session.questions):<8} {session.updated_at:<20}")

        print("-" * 80)
        print(f"(* 表示当前会话)")
        print()

    def _print_session_info(self, session_id: str):
        """打印会话基本信息"""
        session = self.session_manager.get_session(session_id)
        if not session:
            print(f"会话不存在: {session_id}")
            return

        print(f"\n会话信息:")
        print(f"  ID: {session.session_id}")
        print(f"  状态: {'激活' if session.status == 'active' else '归档'}")
        print(f"  创建时间: {session.created_at}")
        print(f"  更新时间: {session.updated_at}")
        print(f"  工作目录: {session.workdir}")
        print(f"  问题数量: {len(session.questions)}")
        if session.summary:
            print(f"  总结: {session.summary[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent - 智能编程助手 (会话模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--workdir',
        type=str,
        nargs='?',
        default='.',
        help='工作目录路径（默认: 当前目录）'
    )

    args = parser.parse_args()

    assistant = SessionAwareCodeAssistant(workdir=args.workdir)
    assistant.setup_engine()
    assistant.interactive()


if __name__ == "__main__":
    main()
