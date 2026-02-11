"""
工具定义层 - 将所有命令转换为标准化Tool Calling格式
支持OpenAI、Qwen、Claude等主流LLM的Tool Calling规范
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    enum: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    """工具定义 - 符合OpenAI Function Calling规范"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        required_params: Optional[List[str]] = None
    ) -> "ToolDefinition":
        """创建工具定义"""
        properties = {}
        for param in parameters:
            param_dict = {
                "type": param.type.value,
                "description": param.description
            }
            if param.enum:
                param_dict["enum"] = param.enum
            if param.properties:
                param_dict["properties"] = param.properties
            properties[param.name] = param_dict
        
        required = required_params or [p.name for p in parameters if p.required]
        
        return cls(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI函数调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_qwen_format(self) -> Dict[str, Any]:
        """转换为Qwen工具调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """转换为Claude工具调用格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


def get_all_tool_definitions() -> List[ToolDefinition]:
    """获取所有工具定义"""
    return [
        TOOL_READ,
        TOOL_WRITE,
        TOOL_EDIT,
        TOOL_GLOB,
        TOOL_GREP,
        TOOL_BASH,
        TOOL_TODOWRITE,
        TOOL_QUESTION,
        TOOL_WEBFETCH,
        TOOL_SKILL,
        TOOL_SESSION_DETAIL,
    ]


TOOL_READ = ToolDefinition.create(
    name="read",
    description="读取文件内容。默认从开头读取最多2000行，超过2000字符的行会被截断。可指定offset和limit控制读取范围。",
    parameters=[
        ToolParameter(
            name="filePath",
            type=ParameterType.STRING,
            description="要读取的文件路径（绝对路径或相对于工作目录）",
            required=True
        ),
        ToolParameter(
            name="limit",
            type=ParameterType.INTEGER,
            description="限制读取的行数，默认2000行",
            required=False
        ),
        ToolParameter(
            name="offset",
            type=ParameterType.INTEGER,
            description="从指定行号开始读取（0-based），默认0",
            required=False
        )
    ]
)

TOOL_WRITE = ToolDefinition.create(
    name="write",
    description="创建新文件或覆盖现有文件。用于写入代码、配置、文档等。会自动创建不存在的目录。",
    parameters=[
        ToolParameter(
            name="filePath",
            type=ParameterType.STRING,
            description="要创建/写入的文件路径",
            required=True
        ),
        ToolParameter(
            name="content",
            type=ParameterType.STRING,
            description="文件内容",
            required=True
        )
    ]
)

TOOL_EDIT = ToolDefinition.create(
    name="edit",
    description="编辑文件内容。精确替换指定文本片段。",
    parameters=[
        ToolParameter(
            name="filePath",
            type=ParameterType.STRING,
            description="要编辑的文件路径",
            required=True
        ),
        ToolParameter(
            name="oldString",
            type=ParameterType.STRING,
            description="要替换的旧文本（必须精确匹配）",
            required=True
        ),
        ToolParameter(
            name="newString",
            type=ParameterType.STRING,
            description="替换后的新文本",
            required=True
        ),
        ToolParameter(
            name="replaceAll",
            type=ParameterType.BOOLEAN,
            description="是否替换所有匹配项（默认False）",
            required=False
        )
    ]
)

TOOL_GLOB = ToolDefinition.create(
    name="glob",
    description="使用通配符模式搜索文件。查找匹配特定模式的文件路径。",
    parameters=[
        ToolParameter(
            name="pattern",
            type=ParameterType.STRING,
            description="文件通配符模式，如 **/*.py、*.txt、src/**/*.js 等",
            required=True
        ),
        ToolParameter(
            name="path",
            type=ParameterType.STRING,
            description="搜索路径，默认为当前工作目录",
            required=False
        )
    ]
)

TOOL_GREP = ToolDefinition.create(
    name="grep",
    description="在文件中搜索匹配正则表达式的文本行。返回包含匹配的行。",
    parameters=[
        ToolParameter(
            name="pattern",
            type=ParameterType.STRING,
            description="正则表达式搜索模式",
            required=True
        ),
        ToolParameter(
            name="path",
            type=ParameterType.STRING,
            description="搜索路径，默认为当前工作目录",
            required=False
        ),
        ToolParameter(
            name="include",
            type=ParameterType.STRING,
            description="文件类型过滤，如 *.py、*.{ts,tsx}",
            required=False
        )
    ]
)

TOOL_BASH = ToolDefinition.create(
    name="bash",
    description="执行终端/Bash命令。用于运行脚本、安装依赖、执行git命令等。",
    parameters=[
        ToolParameter(
            name="command",
            type=ParameterType.STRING,
            description="要执行的命令",
            required=True
        ),
        ToolParameter(
            name="timeout",
            type=ParameterType.INTEGER,
            description="超时时间（毫秒），默认120000（2分钟）",
            required=False
        ),
        ToolParameter(
            name="workdir",
            type=ParameterType.STRING,
            description="执行命令的工作目录",
            required=False
        )
    ]
)

TOOL_TODOWRITE = ToolDefinition.create(
    name="todowrite",
    description="用户需求过于复杂，需要将任务分解为多个步骤并创建任务清单或更新任务状态",
    parameters=[
        ToolParameter(
            name="todos",
            type=ParameterType.ARRAY,
            description="任务列表，每个任务包含id、content、status、priority",
            required=True,
            properties={
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "任务唯一标识"},
                        "content": {"type": "string", "description": "任务描述"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"], "description": "任务状态"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "优先级"}
                    }
                }
            }
        )
    ]
)


TOOL_QUESTION = ToolDefinition.create(
    name="question",
    description="向用户提问以获取更多信息、确认或选择。暂停执行等待用户输入。",
    parameters=[
        ToolParameter(
            name="questions",
            type=ParameterType.ARRAY,
            description="问题列表，每个问题包含question、header、options",
            required=True,
            properties={
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "完整问题描述"},
                        "header": {"type": "string", "description": "简短标签（最多30字符）"},
                        "options": {"type": "array", "description": "可选答案列表"}
                    }
                }
            }
        )
    ]
)

TOOL_WEBFETCH = ToolDefinition.create(
    name="webfetch",
    description="获取网页内容。用于抓取网页并转换为markdown或文本格式。",
    parameters=[
        ToolParameter(
            name="url",
            type=ParameterType.STRING,
            description="要获取的网页URL",
            required=True
        ),
        ToolParameter(
            name="format",
            type=ParameterType.STRING,
            description="返回格式：text、markdown或html，默认markdown",
            required=False,
            enum=["text", "markdown", "html"]
        )
    ]
)

TOOL_SKILL = ToolDefinition.create(
    name="skill",
    description="加载特定任务的技能说明。获取详细的操作指导。",
    parameters=[
        ToolParameter(
            name="name",
            type=ParameterType.STRING,
            description="技能标识符，如 code-review",
            required=True
        )
    ]
)

TOOL_SESSION_DETAIL = ToolDefinition.create(
    name="session_detail",
    description="获取指定步骤的详细信息。返回该步骤的工具调用和执行结果。",
    parameters=[
        ToolParameter(
            name="stepNumber",
            type=ParameterType.INTEGER,
            description="步骤编号（从0开始），指定要查看的步骤",
            required=True
        )
    ]
)


class ToolCallingFormatter:
    """Tool Calling格式转换器 - 适配不同LLM提供商"""
    
    @staticmethod
    def to_openai(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """转换为OpenAI格式"""
        return [tool.to_openai_format() for tool in tools]
    
    @staticmethod
    def to_qwen(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """转换为Qwen格式"""
        return [tool.to_qwen_format() for tool in tools]
    
    @staticmethod
    def to_anthropic(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """转换为Claude/Anthropic格式"""
        return [tool.to_anthropic_format() for tool in tools]
    
    @staticmethod
    def to_ollama(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """转换为Ollama格式（与OpenAI兼容）"""
        return ToolCallingFormatter.to_openai(tools)
    
    @staticmethod
    def format_for_engine(engine_type: str, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """根据引擎类型自动选择格式"""
        formatters = {
            "openai": ToolCallingFormatter.to_openai,
            "anthropic": ToolCallingFormatter.to_anthropic,
            "ollama": ToolCallingFormatter.to_ollama,
            "qwen": ToolCallingFormatter.to_qwen,
            "deepseek": ToolCallingFormatter.to_openai,
        }
        formatter = formatters.get(engine_type.lower(), ToolCallingFormatter.to_openai)
        return formatter(tools)


def get_tool_by_name(name: str) -> Optional[ToolDefinition]:
    """根据名称获取工具定义"""
    for tool in get_all_tool_definitions():
        if tool.name == name:
            return tool
    return None
