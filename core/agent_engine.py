"""
Agent引擎 - 自主任务执行协调器
负责理解用户意图、调用工具、处理结果和决策下一步行动
"""
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import traceback

from core.llm_engine import LLMEngine, Message, MessageRole, LLMResponse
from core.tool_definitions import (
    ToolDefinition, 
    get_all_tool_definitions, 
    ToolCallingFormatter
)
from core.tool_executor import ToolExecutor, ToolResult
from config import DEFAULT_SYSTEM_PROMPT


@dataclass
class AgentStep:
    """Agent执行步骤记录"""
    step_number: int
    timestamp: str
    user_input: str
    llm_response: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[ToolResult]
    is_completed: bool
    final_message: Optional[str] = None
    thinking: Optional[str] = None


@dataclass
class AgentConfig:
    """Agent配置"""
    max_iterations: int = 200
    max_execution_time: float = 900.0
    execution_delay: float = 0.5
    verbose_output: bool = True
    auto_continue: bool = True


class AgentEngine:
    """
    Agent引擎 - 自主任务执行核心

    工作流程:
    1. 接收用户自然语言输入
    2. 发送给LLM（带工具定义）
    3. 解析LLM的工具调用请求
    4. 执行工具并收集结果
    5. 将结果反馈给LLM
    6. 重复直到任务完成
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        tool_executor: ToolExecutor,
        config: Optional[AgentConfig] = None,
        session_summary: Optional[str] = None
    ):
        self.llm_engine = llm_engine
        self.tool_executor = tool_executor
        self.config = config or AgentConfig()
        self._session_summary = session_summary

        self._conversation_history: List[Message] = []
        self._execution_steps: List[AgentStep] = []
        self._current_step = 0
        self._is_running = False
        self._start_time: Optional[float] = None

        self._tool_definitions = get_all_tool_definitions()
    
    def start(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> Iterator[AgentStep]:
        """
        开始执行用户任务
        
        Args:
            user_input: 用户自然语言输入
            system_prompt: 自定义系统提示词
            
        Yields:
            AgentStep: 执行步骤记录
        """
        self._is_running = True
        self._start_time = time.time()
        self._execution_steps.clear()
        self._current_step = 0
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        self._conversation_history = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
        ]
        
        if self._session_summary:
            self._conversation_history.append(
                Message(role=MessageRole.USER, content=f"[上下文信息 - 会话历史]\n\n{self._session_summary}")
            )
        
        self._conversation_history.append(
            Message(role=MessageRole.USER, content=user_input),
        )
        
        step_number = 0
        
        while self._is_running and self._check_limits():
            step_number += 1
            self._current_step = step_number
            
            if self.config.verbose_output:
                print(f"\n{'='*60}")
                print(f"步骤 {step_number}")
                # print(f"{'='*60}")
            
            step = self._execute_step(step_number, user_input)
            yield step
            if len(step.tool_results)>0 and step.tool_results[0].content == '已向用户发送问题':
                break

            if step.is_completed:
                break


            if self.config.execution_delay > 0:
                time.sleep(self.config.execution_delay)
        
        if not self._is_running:
            yield AgentStep(
                step_number=step_number + 1,
                timestamp=datetime.now().isoformat(),
                user_input=user_input,
                llm_response="",
                tool_calls=[],
                tool_results=[],
                is_completed=False,
                final_message="任务被中断"
            )
    
    def _execute_step(
        self,
        step_number: int,
        original_user_input: str
    ) -> AgentStep:
        """执行单个步骤"""
        tool_calls = []
        tool_results = []
        llm_response = ""
        thinking_content = None

        try:

            # 获取工具定义并传递给LLM
            tools = self.get_tool_definitions("openai")

            response = self.llm_engine.chat(
                self._conversation_history,
                tools=tools,
                temperature=0.1
            )

            if isinstance(response, LLMResponse):
                llm_response = response.content
                thinking_content = response.thinking
                tool_calls=response.tool_calls
            else:
                raise ("格式出错了")

            if self.config.verbose_output:
                if thinking_content:
                    print("思考过程:")
                    print(thinking_content)
                    print('-'*60)

            
            if tool_calls:
                
                # 添加assistant消息（包含tool_calls）- 保持原始格式
                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content="",
                    timestamp=time.time(),
                    tool_calls=tool_calls
                )
                self._conversation_history.append(assistant_message)
                
                for call in tool_calls:
                    # 提取工具名称（支持OpenAI格式）
                    if isinstance(call, dict) and "function" in call:
                        tool_name = call.get("function", {}).get("name", "")
                        # 提取参数
                        args_data = call.get("function", {}).get("arguments", {})
                        if isinstance(args_data, str):
                            try:
                                arguments = json.loads(args_data)
                            except json.JSONDecodeError:
                                arguments = {"raw": args_data}
                        else:
                            arguments = args_data
                    else:
                        tool_name = call.get("name", "")
                        arguments = call.get("arguments", {})
                    
                    result = self.tool_executor.execute_tool(tool_name, arguments)
                    tool_results.append(result)
                    
                    result_message = self._format_tool_result(result)
                    self._add_tool_message(result_message)

                    if self.config.verbose_output:
                        print(f"\n执行工具: {tool_name}")
                        print(f"参数:\n {json.dumps(arguments, ensure_ascii=False, indent=2)}")
                        print(f"执行结果:\n {result_message}")

                step = AgentStep(
                    step_number=step_number,
                    timestamp=datetime.now().isoformat(),
                    user_input=original_user_input,
                    llm_response=llm_response,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    is_completed=False,
                    thinking=thinking_content
                )
            else:
                # 没有工具调用，检查是否是最终响应
                # 如果LLM返回的是文本响应（而不是工具调用），说明任务已完成
                
                # 检查之前的步骤是否有工具执行
                has_previous_tools = any(
                    step.tool_calls for step in self._execution_steps
                )
                
                # 检查最近的工具执行结果是否成功
                recent_tool_success = False
                for msg in reversed(self._conversation_history):
                    if msg.role == MessageRole.TOOL:
                        tool_content = msg.content
                        if ("success" in tool_content.lower() or 
                            "已创建" in tool_content or
                            "已保存" in tool_content):
                            recent_tool_success = True
                            break
                
                if has_previous_tools or recent_tool_success:
                    # 之前执行过工具且成功，现在返回文本响应，任务完成
                    step = AgentStep(
                        step_number=step_number,
                        timestamp=datetime.now().isoformat(),
                        user_input=original_user_input,
                        llm_response=llm_response,
                        tool_calls=[],
                        tool_results=[],
                        is_completed=True,
                        final_message=llm_response,
                        thinking=thinking_content
                    )
                elif self._is_final_response(llm_response, original_user_input):
                    step = AgentStep(
                        step_number=step_number,
                        timestamp=datetime.now().isoformat(),
                        user_input=original_user_input,
                        llm_response=llm_response,
                        tool_calls=[],
                        tool_results=[],
                        is_completed=True,
                        final_message=llm_response,
                        thinking=thinking_content
                    )
                else:
                    # 可能是简单的问答，不需要工具
                    step = AgentStep(
                        step_number=step_number,
                        timestamp=datetime.now().isoformat(),
                        user_input=original_user_input,
                        llm_response=llm_response,
                        tool_calls=[],
                        tool_results=[],
                        is_completed=True,
                        final_message=llm_response,
                        thinking=thinking_content
                    )
            
            self._execution_steps.append(step)
            return step
            
        except Exception as e:
            error_msg = f"执行出错: {str(e)}\n{traceback.format_exc()}"
            
            step = AgentStep(
                step_number=step_number,
                timestamp=datetime.now().isoformat(),
                user_input=original_user_input,
                llm_response=llm_response,
                tool_calls=tool_calls,
                tool_results=tool_results,
                is_completed=True,
                final_message=error_msg,
                thinking=thinking_content
            )
            
            self._execution_steps.append(step)
            return step
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        解析LLM响应中的工具调用
        
        支持多种格式:
        1. OpenAI工具调用格式: {"tool_calls": [...]}
        2. Qwen工具调用格式: {"tool_calls": [...]}
        3. 文本格式: 从响应中提取JSON
        """
        # 首先尝试解析JSON
        try:
            data = json.loads(response)
            
            # 如果是工具调用格式
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                calls = []
                for call in data["tool_calls"]:
                    if isinstance(call, dict):
                        # OpenAI格式 - 保持原始格式
                        calls.append(call)
                return calls
            
            # 如果是function_calls格式（某些模型使用）
            if "function_calls" in data and isinstance(data["function_calls"], list):
                return [
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": fc.get("arguments", "")
                        }
                    }
                    for fc in data["function_calls"]
                ]
            
            # 如果响应包含"tool_calls"但不在预期的结构中，尝试直接返回空列表
            return []
            
        except json.JSONDecodeError:
            return self._extract_tool_calls_from_text(response)
    
    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取工具调用信息"""
        import re
        
        calls = []
        
        patterns = [
            r'调用工具["：:]\s*(\w+)\s*参数?\s*[:：]?\s*(\{[^}]+\})',
            r'使用工具["：:]\s*(\w+)\s*\(([^)]+)\)',
            r'"name":\s*"(\w+)"[^}]*"arguments":\s*(\{[^}]+\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    try:
                        arguments = json.loads(match[1])
                    except json.JSONDecodeError:
                        arguments = {"raw": match[1]}
                    
                    calls.append({
                        "id": "",
                        "name": match[0],
                        "arguments": arguments
                    })
        
        return calls
    
    def _add_tool_message(self, content: str):
        """添加工具执行结果到对话历史"""
        self._conversation_history.append(
            Message(role=MessageRole.TOOL, content=content)
        )
    
    def _format_tool_result(self, result: ToolResult) -> str:
        """格式化工具执行结果"""

        return f"工具 [{result.tool_name}] 执行信息:\n{result.content} {result.error}"

    
    def _is_final_response(self, response: str, user_input: str) -> bool:
        """
        检查响应是否应该是最终回答
        某些任务（如回答问题）不需要工具调用，直接返回文本即可
        """
        # 如果响应太短，可能是空的或无效的
        if len(response.strip()) < 10:
            return False
        
        # 检查响应是否包含典型的最终回答特征
        final_indicators = [
            "任务完成", "已经完成", "已完成", 
            "已经为您创建", "已经为您生成",
            "文件已创建", "代码已生成", "已成功创建",
            "代码已保存"
        ]
        
        for indicator in final_indicators:
            if indicator in response:
                return True
        
        # 检查是否是简单的问答（不涉及文件操作）
        question_keywords = ["解释", "说明", "什么是", "如何", "为什么"]
        operation_keywords = ["创建", "生成", "写", "修改", "编辑", "删除", "查看", "请求"]
        
        has_question = any(kw in user_input for kw in question_keywords)
        has_operation = any(kw in user_input for kw in operation_keywords)
        
        # 如果用户只是问问题而不是要求操作，且响应回答了问题
        if has_question and not has_operation:
            return True
        
        # 如果用户要求操作但没有工具调用，返回False（应该继续）
        if has_operation and not self._parse_tool_calls(response):
            return False
        
        # 检查工具执行结果的历史
        for msg in reversed(self._conversation_history):
            if msg.role == MessageRole.TOOL:
                tool_content = msg.content
                # 如果工具执行成功保存了文件，认为任务可能完成
                if "代码已保存" in tool_content or "已创建" in tool_content:
                    return True
        
        return False
    
    def _check_limits(self) -> bool:
        """检查执行限制"""
        if self._current_step >= self.config.max_iterations:
            if self.config.verbose_output:
                print(f"\n达到最大迭代次数 ({self.config.max_iterations})")
            return False
        
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self.config.max_execution_time:
                if self.config.verbose_output:
                    print(f"\n达到最大执行时间 ({self.config.max_execution_time}s)")
                return False
        
        return True
    
    def stop(self):
        """停止执行"""
        self._is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "is_running": self._is_running,
            "current_step": self._current_step,
            "total_steps": len(self._execution_steps),
            "elapsed_time": time.time() - self._start_time if self._start_time else 0,
            "last_step": self._execution_steps[-1].__dict__ if self._execution_steps else None
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        successful_steps = sum(1 for s in self._execution_steps if s.is_completed or any(r.success for r in s.tool_results))
        failed_steps = len(self._execution_steps) - successful_steps
        total_tool_calls = sum(len(s.tool_calls) for s in self._execution_steps)
        successful_calls = sum(
            sum(1 for r in s.tool_results if r.success) 
            for s in self._execution_steps
        )
        
        return {
            "total_steps": len(self._execution_steps),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_calls,
            "execution_time": time.time() - self._start_time if self._start_time else 0,
            "is_completed": any(s.is_completed for s in self._execution_steps),
            "final_message": None
        }
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        tool_descriptions = []
        for tool in self._tool_definitions:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        tools_text = "\n".join(tool_descriptions)

        workdir = str(self.tool_executor.workdir)

        return f"""{DEFAULT_SYSTEM_PROMPT}

# 智能编程助手系统提示词

你是一个专业的智能编程助手，能够自主理解用户需求并高效完成各种编程任务。

## 工作环境
- **工作目录**: {workdir}
- **执行模式**: 自主工具调用 + 用户协作

## 核心能力

### 可用工具箱
{tools_text}

### 任务执行策略
1. **需求理解**: 分析用户需求，明确任务边界和目标
2. **任务分解**: 将复杂任务拆解为可管理的子任务
3. **渐进式探索**: 先了解项目结构，再深入具体实现
4. **小步迭代**: 每次只修改一个方面，便于验证和回滚
5. **持续验证**: 每个步骤后验证结果是否符合预期

### 任务管理规范
- **何时分解**: 任务涉及多个文件、多个步骤或需要决策时
- **如何分解**:
  1. 使用 `todowrite` 创建任务清单
  2. 按逻辑顺序排列任务
  3. 每个任务只做一件事
  4. 完成一个更新一个状态
- **任务粒度**: 保持原子性，每个任务应该可在10分钟内完成

### 文件操作规范
- **路径规则**: 
  - 强制使用**相对路径**（相对于 {workdir}）
  - 禁止使用绝对路径
  - 示例：src/main.py、tests/test_api.py
  
- **操作时机**:
  - 创建文件 → `write`
  - 修改文件 → `edit`
  - 查看文件 → `read`
  - 搜索文件 → `glob`
  - 搜索内容 → `grep`
  - 执行命令 → `bash`

### 探索与理解策略
- **新项目**: 先 `glob` 了解结构，再 `read` 关键文件
- **陌生代码**: 从入口文件开始，追踪关键函数
- **定位问题**: 使用 `grep` 搜索相关代码，结合 `read` 理解上下文

### 错误处理机制
- **遇到错误**:
  1. 记录错误信息
  2. 分析可能原因
  3. 尝试修复或询问用户
- **验证修复**: 执行后检查结果是否符合预期
- **一致性检查**: 确保修改不会破坏其他功能

## 交互模式

### 用户协作指南
- **需要询问**: 需求不明确、可能产生负面影响、多方案选择时
- **主动反馈**: 遇到问题、做出重要决策、完成任务时
- **确认方式**: 使用 `questions` 工具明确提问

### 反馈质量要求
- **进行中**: 简要说明当前状态
- **完成时**: 总结完成的工作、文件的修改、结果的验证
- **遇到问题**: 说明问题、尝试的方案、下一步计划

## 工具调用黄金法则

### 必须调用工具的场景
- 创建/修改/读取文件
- 搜索文件或代码内容
- 执行任何命令
- 管理任务清单

### 禁止的行为
- ❌ 在响应中直接提供完整代码（应使用 `write`）
- ❌ 描述要做什么（应直接调用工具）
- ❌ 使用绝对路径
- ❌ 跳过验证步骤

### 工具调用格式
```json
{{
    "tool_calls": [
        {{
            "id": "call_1",
            "function": {{
                "name": "工具名称",
                "arguments": {{参数}}
            }}
        }}
    ]
}}
```

## 质量保证清单

在每次工具调用前检查：
- [ ] 参数是否完整正确？
- [ ] 目标路径是否正确？
- [ ] 是否会影响其他文件？
- [ ] 是否需要先验证某个前提？

在每次操作后检查：
- [ ] 操作是否成功？
- [ ] 结果是否符合预期？
- [ ] 是否需要更新任务清单？
- [ ] 是否应该通知用户？

## 执行流程

1. **接收任务** → 分析需求，判断复杂度
3. **翻阅历史记录** → 获取修改命令，未完成任务清单，了解上下文
4. **制定计划** → 分解任务，创建清单（需要时）
5. **执行步骤** → 按清单顺序执行每项任务
6. **验证结果** → 检查操作是否成功
7. **更新状态** → 标记完成，处理下一个
7. **总结汇报** → 完成后给用户清晰总结

## 重要提醒

- **工具优先**: 任何需要实际操作的任务都必须调用工具
- **路径正确**: 相对路径是强制要求
- **原子操作**: 每次只做一件事
- **及时反馈**: 主动告知进度和结果
- **安全第一**: 不确定时询问用户

请始终以工具调用为先导，通过实际操作完成任务，而非仅在响应中描述操作。
"""
    
    def get_tool_definitions(self, engine_type: str = "openai") -> List[Dict[str, Any]]:
        """获取工具定义（指定格式）"""
        return ToolCallingFormatter.format_for_engine(engine_type, self._tool_definitions)
    
    def get_conversation_history(self) -> List[Message]:
        """获取对话历史"""
        return self._conversation_history.copy()
    
    def get_execution_steps(self) -> List[AgentStep]:
        """获取执行步骤"""
        return self._execution_steps.copy()

