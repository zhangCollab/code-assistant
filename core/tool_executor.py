"""
工具执行器 - 执行Tool Calling请求的工具
将LLM的工具调用请求转换为实际命令执行
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import traceback
import json
import subprocess
import os
import threading
import queue

from core.file_manager import FileManager


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    content: str
    error: Optional[str] = None
    tool_name: str = ""
    execution_time: float = 0.0
    timestamp: str = ""


@dataclass
class ToolResponse:
    """统一工具响应格式"""
    success: bool
    content: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "content": self.content
        }
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def ok(cls, content: str = "", data: Optional[Dict[str, Any]] = None) -> "ToolResponse":
        """创建成功响应"""
        return cls(success=True, content=content, data=data)

    @classmethod
    def fail(cls, error: str, data: Optional[Dict[str, Any]] = None) -> "ToolResponse":
        """创建失败响应"""
        return cls(success=False, content="", error=error, data=data)


class ToolExecutor:
    """工具执行器 - 执行LLM请求的工具调用"""

    def __init__(self, file_manager: FileManager, workdir: str = ".", session_manager=None):
        self.file_manager = file_manager
        self.workdir = Path(workdir).resolve()
        self._execution_history: List[Dict[str, Any]] = []
        self._todo_list: List[Dict[str, Any]] = []
        self._session_manager = session_manager

        self._register_tools()
    
    def _register_tools(self):
        """注册所有可用工具"""
        self.tools = {
            "read": self._execute_read,
            "write": self._execute_write,
            "edit": self._execute_edit,
            "glob": self._execute_glob,
            "grep": self._execute_grep,
            "bash": self._execute_bash,
            "todowrite": self._execute_todowrite,
            "question": self._execute_question,
            "webfetch": self._execute_webfetch,
            "skill": self._execute_skill,
            "session_detail": self._execute_session_detail,
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行工具调用"""
        import time
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                content="",
                error=f"未知工具: {tool_name}",
                tool_name=tool_name,
                execution_time=0,
                timestamp=datetime.now().isoformat()
            )
        
        try:
            func = self.tools[tool_name]
            response: ToolResponse = func(arguments)
            execution_time = time.time() - start_time
            
            result_content = self._format_result(response)

            self._execution_history.append({
                "tool": tool_name,
                "arguments": arguments,
                "success": response.success,
                "response": response.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            return ToolResult(
                success=response.success,
                content=result_content,
                error=response.error,
                tool_name=tool_name,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            self._execution_history.append({
                "tool": tool_name,
                "arguments": arguments,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return ToolResult(
                success=False,
                content="",
                error=error_msg,
                tool_name=tool_name,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    def _format_result(self, result: ToolResponse) -> str:
        """格式化工具响应结果"""
        if not result:
            return "操作完成，无返回内容"
        parts = []

        if result.content:
            parts.append(result.content)

        if result.data:
            parts.append(json.dumps(result.data))

        if result.error:
            parts.append(result.error)


        return "\n".join(parts)

    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self._execution_history.copy()
    
    def clear_history(self):
        """清空执行历史"""
        self._execution_history.clear()
    
    def _resolve_path(self, file_path: str) -> Path:
        """解析文件路径 - 强制使用workdir下的相对路径"""
        path = Path(file_path)
        
        # 如果是绝对路径，强制转换为相对于workdir的路径
        if path.is_absolute():
            try:
                # 尝试提取相对于workdir的部分
                rel_path = path.relative_to(self.workdir)
                return self.workdir / rel_path
            except ValueError:
                # 路径不在workdir下，强制使用文件名部分
                return self.workdir / path.name
        
        return self.workdir / file_path
    
    def _execute_read(self, args: Dict[str, Any]) -> ToolResponse:
        """读取文件"""
        file_path = args.get("filePath")
        limit = args.get("limit", 2000)
        offset = args.get("offset", 0)
        
        if not file_path:
            return ToolResponse.fail("缺少文件路径参数")
        
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return ToolResponse.fail(f"文件不存在: {file_path}")
            
            if path.is_dir():
                return ToolResponse.fail(f"是目录而非文件: {file_path}")
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                line_number = 0
                
                for line in f:
                    line_number += 1
                    
                    if line_number > offset:
                        if len(line) > 2000:
                            current_line = line[:2000]
                        else:
                            current_line = line
                        lines.append(current_line)
                        
                        if len(lines) >= limit:
                            break
                
                content = ''.join(lines)
                actual_line_count = len(content.split('\n'))
                
                return ToolResponse.ok(
                    content=content,
                    data={
                        "path": str(path.absolute()),
                        "size": len(content.encode('utf-8')),
                        "lines": actual_line_count,
                        "truncated": line_number > offset + limit
                    }
                )
        except Exception as e:
            return ToolResponse.fail(f"读取失败: {str(e)}")
    
    def _execute_write(self, args: Dict[str, Any]) -> ToolResponse:
        """写入文件"""
        file_path = args.get("filePath")
        content = args.get("content", "")
        
        if not file_path:
            return ToolResponse.fail("缺少文件路径参数")
        
        try:
            path = self._resolve_path(file_path)
            
            if path.exists() and path.is_dir():
                return ToolResponse.fail(f"是目录而非文件: {file_path}")
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return ToolResponse.ok(
                content=f"已写入文件: {path}",
                data={
                    "path": str(path.absolute()),
                    "size": len(content.encode('utf-8'))
                }
            )
        except Exception as e:
            return ToolResponse.fail(f"写入失败: {str(e)}")
    
    def _execute_edit(self, args: Dict[str, Any]) -> ToolResponse:
        """编辑文件"""
        file_path = args.get("filePath")
        old_text = args.get("oldString")
        new_text = args.get("newString")
        replace_all = args.get("replaceAll", False)
        
        if not file_path or not old_text or new_text is None:
            return ToolResponse.fail("缺少必要参数（filePath, oldString, newString）")
        
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return ToolResponse.fail(f"文件不存在: {file_path}")
            
            content = path.read_text(encoding='utf-8')
            
            if old_text not in content:
                return ToolResponse.fail("未找到匹配文本")
            
            if replace_all:
                new_content = content.replace(old_text, new_text)
                replacement_count = content.count(old_text)
            else:
                new_content = content.replace(old_text, new_text, 1)
                replacement_count = 1
            
            path.write_text(new_content, encoding='utf-8')
            
            return ToolResponse.ok(
                content=f"已完成替换（{replacement_count}处）",
                data={
                    "path": str(path.absolute()),
                    "size": len(new_content.encode('utf-8'))
                }
            )
        except Exception as e:
            return ToolResponse.fail(f"编辑失败: {str(e)}")

    def _execute_glob(self, args: Dict[str, Any]) -> ToolResponse:
        """文件模式匹配"""
        pattern = args.get("pattern")
        path = args.get("path", ".")

        if not pattern:
            return ToolResponse.fail("缺少模式参数")

        try:
            search_path = self._resolve_path(path)

            if not search_path.exists():
                return ToolResponse.fail(f"路径不存在: {path}")

            search_dir = search_path
            if search_path.is_file():
                search_dir = search_path.parent
            elif not search_path.is_dir():
                return ToolResponse.fail(f"不是有效的路径: {path}")

            files = list(search_dir.rglob(pattern))
            files = [str(f.relative_to(self.workdir) if f.is_relative_to(self.workdir) else f) for f in files]
            files.sort()

            return ToolResponse.ok(
                content=f"找到 {len(files)} 个匹配文件",
                data={
                    "files": files[:100],
                    "count": len(files)
                }
            )
        except Exception as e:
            return ToolResponse.fail(f"搜索失败: {str(e)}")
    
    def _execute_grep(self, args: Dict[str, Any]) -> ToolResponse:
        """内容搜索"""
        pattern = args.get("pattern")
        path = args.get("path", ".")
        include = args.get("include")
        
        if not pattern:
            return ToolResponse.fail("缺少搜索模式")
        
        try:
            import re
            search_path = self._resolve_path(path)
            
            if not search_path.exists():
                return ToolResponse.fail(f"路径不存在: {path}")
            
            regex = re.compile(pattern)
            results = []
            
            if search_path.is_file():
                files = [search_path]
            else:
                files = search_path.rglob("*") if include else search_path.glob("**/*")
                files = [f for f in files if f.is_file()]
                if include:
                    from fnmatch import fnmatch
                    files = [f for f in files if any(fnmatch(f.name, p) for p in include.split())]
            
            for file_path in files:
                try:
                    lines = file_path.read_text(encoding='utf-8').split('\n')
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            rel_path = str(file_path.relative_to(self.workdir)) if file_path.is_relative_to(self.workdir) else str(file_path)
                            results.append(f"{rel_path}:{i+1}:{line.rstrip()}")
                except Exception:
                    continue
            
            return ToolResponse.ok(
                content=f"找到 {len(results)} 条匹配结果",
                data={
                    "results": results[:100],
                    "count": len(results)
                }
            )
        except Exception as e:
            return ToolResponse.fail(f"搜索失败: {str(e)}")
    
    def _execute_bash(self, args: Dict[str, Any]) -> ToolResponse:
        """执行命令"""
        command = args.get("command")
        timeout = args.get("timeout", 300000)
        workdir = args.get("workdir")
        
        if not command:
            return ToolResponse.fail("缺少命令参数")
        
        cwd = self.workdir
        if workdir:
            cwd = self._resolve_path(workdir)
            if not cwd.exists():
                cwd = self.workdir

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd),
                env={**os.environ, "FORCE_COLOR": "1"},
                bufsize=1
            )

            output_queue = queue.Queue()
            output_lines = []
            timeout_seconds = 5.0

            def read_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_queue.put(line)
                        else:
                            break
                except Exception:
                    pass
                finally:
                    output_queue.put(None)

            reader_thread = threading.Thread(target=read_output, daemon=True)
            reader_thread.start()

            last_output_time = threading.Event()
            last_output_time.set()

            while True:
                try:
                    line = output_queue.get(timeout=timeout_seconds)
                    if line is None:
                        break
                    output_lines.append(line.rstrip())
                    print(line, end='')
                    last_output_time.set()
                except queue.Empty:
                    print(f"\n[超时] {timeout_seconds}秒内无新输出，停止服务")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break

            exit_code = process.returncode

            full_output = "\n".join(output_lines)

            if exit_code != 0 or exit_code is None:
                return ToolResponse.fail(
                    error=f"程序超时，进程已退出",
                    data={
                        "command": command,
                        "output": full_output,
                        "exit_code": exit_code
                    }
                )

            return ToolResponse.ok(
                content="服务已启动（查看上方输出获取实时日志）",
                data={
                    "command": command,
                    "output": full_output,
                    "exit_code": exit_code
                }
            )
                
        except subprocess.TimeoutExpired:
            return ToolResponse.fail(f"命令执行超时（{timeout/1000}秒）")
        except Exception as e:
            return ToolResponse.fail(f"执行失败: {str(e)}")
    
    def _execute_todowrite(self, args: Dict[str, Any]) -> ToolResponse:
        """更新任务清单"""
        todos = args.get("todos", [])
        
        if not isinstance(todos, list):
            return ToolResponse.fail("todos必须是数组")
        
        self._todo_list = todos
        
        return ToolResponse.ok(
            content=f"已更新任务清单（共{len(todos)}个任务）",
            data={"todos": self._todo_list}
        )

    
    def _execute_question(self, args: Dict[str, Any]) -> ToolResponse:
        """向用户提问"""
        questions = args.get("questions", [])
        
        if not isinstance(questions, list) or not questions:
            return ToolResponse.fail("questions必须是问题数组")
        
        return ToolResponse.ok(
            content="已向用户发送问题",
            data={
                "questions": questions,
                "pending": True
            }
        )
    
    def _execute_webfetch(self, args: Dict[str, Any]) -> ToolResponse:
        """获取网页内容"""
        url = args.get("url")
        format_type = args.get("format", "markdown")
        
        if not url:
            return ToolResponse.fail("缺少URL参数")
        
        try:
            import urllib.request
            import ssl
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CodeAssistant/1.0)'
            })
            
            with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                content = response.read().decode('utf-8', errors='ignore')
            
            if format_type == "text":
                from html import unescape
                import re
                content = re.sub(r'<[^>]+>', '', content)
                content = unescape(content)
                content = re.sub(r'\s+', ' ', content).strip()
            
            return ToolResponse.ok(
                content=content[:5000] if len(content) > 5000 else content,
                data={
                    "url": url,
                    "format": format_type,
                    "size": len(content)
                }
            )
        except Exception as e:
            return ToolResponse.fail(f"获取失败: {str(e)}")
    
    def _execute_skill(self, args: Dict[str, Any]) -> ToolResponse:
        """加载技能"""
        name = args.get("name")
        
        if not name:
            return ToolResponse.fail("缺少技能名称")
        
        skills = {
            "code-review": """## Code Review 技能

### 分析代码时：
1. 首先阅读完整代码，理解整体结构
2. 检查常见的代码问题：
   - 未使用的导入/变量
   - 硬编码的值
   - 错误处理缺失
   - 安全风险
3. 评估代码质量、可读性和可维护性
4. 提供具体的改进建议

### 输出格式：
- 问题描述
- 位置（文件:行号）
- 严重程度
- 改进建议""",
        }
        
        skill_content = skills.get(name)
        
        if not skill_content:
            return ToolResponse.fail(f"未找到技能: {name}")
        
        return ToolResponse.ok(
            content=skill_content,
            data={"skill": name}
        )

    def _execute_session_detail(self, args: Dict[str, Any]) -> ToolResponse:
        """获取步骤详情"""
        if not self._session_manager:
            return ToolResponse.fail("会话管理器未初始化")

        step_number = args.get("stepNumber")

        if step_number is None:
            return ToolResponse.fail("缺少必要参数: stepNumber")

        detail = self._session_manager.get_step_detail(step_number)

        if "error" in detail:
            return ToolResponse.fail(detail["error"])

        return ToolResponse.ok(
            content=f"步骤 {step_number} 详情",
            data={
                "step_number": detail["step_number"],
                "question": detail["question"],
                "timestamp": detail["timestamp"],
                "tool_calls": detail.get("tool_calls", []),
                "tool_results": detail.get("tool_results", []),
                "is_completed": detail.get("is_completed", False),
                "final_message": detail.get("final_message")
            }
        )
