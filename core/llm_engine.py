"""
大模型引擎 - 支持多种模型接口
"""
import json
import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from dataclasses import dataclass as dc

from config import MODEL_CONFIG


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    timestamp: float = 0.0
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None

@dc
class LLMResponse:
    """LLM响应"""
    content: str
    thinking: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMEngine(ABC):
    """大模型引擎基类"""
    
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> str:
        """单轮对话"""
        pass
    
    @abstractmethod
    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """流式对话"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass


class OpenAIEngine(LLMEngine):
    """OpenAI兼容引擎（支持GPT-4、Ollama等）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        **kwargs
    ):
        self.api_key = api_key or MODEL_CONFIG.get("api_key", "")
        self.api_base = api_base or MODEL_CONFIG.get("api_base", "").rstrip("/")
        self.model = model
        self.temperature = kwargs.get("temperature", MODEL_CONFIG.get("temperature", 0.7))
        self.max_tokens = kwargs.get("max_tokens", MODEL_CONFIG.get("max_tokens", 4096))
        self.top_p = kwargs.get("top_p", MODEL_CONFIG.get("top_p", 0.9))

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        """转换消息格式"""
        converted = []
        for msg in messages:
            msg_dict = {
                "role": msg.role.value,
                "content": msg.content or ""
            }
            
            # 如果是ASSISTANT角色且有tool_calls，添加tool_calls
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            # 如果是TOOL角色，添加tool_call_id
            if msg.role == MessageRole.TOOL:
                # 从内容中提取tool_call_id（格式：工具 [xxx] 执行成功:）
                import re
                match = re.search(r'工具 \[(\w+)\]', msg.content)
                if match:
                    tool_name = match.group(1)
                    msg_dict["tool_call_id"] = f"call_{tool_name}"
                else:
                    msg_dict["tool_call_id"] = f"call_{len(converted)}"
            
            converted.append(msg_dict)
        
        return converted
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """发送对话请求"""
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }

        if tools:
            payload["tools"] = tools
            if "temperature" not in kwargs:
                payload["temperature"] = 0

        response = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            timeout=kwargs.get("timeout", 60)
        )
        response.raise_for_status()
        result = response.json()

        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            thinking_content = message.get("reasoning_content", None)

            if "tool_calls" in message:
                return LLMResponse(
                    content="",
                    thinking=thinking_content,
                    tool_calls=message["tool_calls"]
                )

            return LLMResponse(
                content=message.get("content", ""),
                thinking=thinking_content
            )

        return LLMResponse(content="")

    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """流式对话"""
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True,
        }
        
        response = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            stream=True,
            timeout=kwargs.get("timeout", 60)
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data != '[DONE]':
                        chunk = json.loads(data)
                        if chunk.get("choices"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
    
    def count_tokens(self, text: str) -> int:
        """估算token数量（英文约4字符/token，中文约2字符/token）"""
        chinese_chars = sum(1 for char in text if ord(char) > 127)
        english_chars = len(text) - chinese_chars
        return int(chinese_chars / 2 + english_chars / 4)


class QwenEngine(OpenAIEngine):
    """阿里千问大模型引擎"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "qwen-plus",
        **kwargs
    ):
        api_base = api_base or MODEL_CONFIG.get("qwen_api_base", "")
        super().__init__(
            api_key=api_key,
            api_base=api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
            **kwargs
        )
        self.enable_thinking = kwargs.get("enable_thinking", MODEL_CONFIG.get("enable_thinking", True))

    def _is_qwen_model(self) -> bool:
        model_lower = self.model.lower()
        return "qwen" in model_lower or "qwq" in model_lower

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }

        if tools:
            payload["tools"] = tools
            if "temperature" not in kwargs:
                payload["temperature"] = 0

        if self._is_qwen_model() and self.enable_thinking:
            payload["enable_thinking"] = True

        response = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            timeout=kwargs.get("timeout", 60)
        )
        response.raise_for_status()
        result = response.json()

        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            thinking_content = message.get("reasoning_content", None)

            if "tool_calls" in message:
                return LLMResponse(
                    content="",
                    thinking=thinking_content,
                    tool_calls=message["tool_calls"]
                )

            return LLMResponse(
                content=message.get("content", ""),
                thinking=thinking_content
            )

        return LLMResponse(content="")



class BigModelEngine(LLMEngine):
    """智谱AI BigModel引擎"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4.7",
        **kwargs
    ):
        self.api_key = api_key or MODEL_CONFIG.get("api_key", "")
        self.model = model or MODEL_CONFIG.get("model", "")
        self.temperature = kwargs.get("temperature", MODEL_CONFIG.get("temperature", 0.7))
        self.max_tokens = kwargs.get("max_tokens", MODEL_CONFIG.get("max_tokens", 4096))
        self.enable_thinking = kwargs.get("enable_thinking", MODEL_CONFIG.get("enable_thinking", True))

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        converted = []
        for msg in messages:
            msg_dict = {
                "role": msg.role.value,
                "content": msg.content or ""
            }
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.role == MessageRole.TOOL:
                import re
                match = re.search(r'工具 \[(\w+)\]', msg.content)
                if match:
                    tool_name = match.group(1)
                    msg_dict["tool_call_id"] = f"call_{tool_name}"
                else:
                    msg_dict["tool_call_id"] = f"call_{len(converted)}"
            converted.append(msg_dict)
        return converted

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools:
            payload["tools"] = tools
            if "temperature" not in kwargs:
                payload["temperature"] = 0

        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled"}

        response = self.session.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            json=payload,
            timeout=kwargs.get("timeout", 120)
        )
        response.raise_for_status()
        result = response.json()

        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            thinking_content = message.get("reasoning_content", None)

            if "tool_calls" in message:
                return LLMResponse(
                    content="",
                    thinking=thinking_content,
                    tool_calls=message["tool_calls"]
                )

            return LLMResponse(
                content=message.get("content", ""),
                thinking=thinking_content
            )

        return LLMResponse(content="")

    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }

        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled"}

        response = self.session.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            json=payload,
            stream=True,
            timeout=kwargs.get("timeout", 120)
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data != '[DONE]':
                        chunk = json.loads(data)
                        if chunk.get("choices"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content

    def count_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for char in text if ord(char) > 127)
        english_chars = len(text) - chinese_chars
        return int(chinese_chars / 2 + english_chars / 4)


class LLMFactory:
    """LLM引擎工厂"""
    
    @staticmethod
    def create_engine(producer: str = "ollama", **kwargs) -> LLMEngine:
        """创建LLM引擎"""
        engines = {
            "openai": OpenAIEngine,
            "qwen": QwenEngine,
            "bigmodel": BigModelEngine,
            "local": lambda **kw: OpenAIEngine(api_base=kw.get("api_base", ""), **kw),
        }
        
        engine_class = engines.get(producer.lower())
        if not engine_class:
            raise ValueError(f"Unknown engine type: {producer}")
        
        return engine_class(**kwargs)
    
    @staticmethod
    def get_available_engines() -> Dict[str, List[str]]:
        """获取可用的引擎列表"""
        return {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "qwen": ["qwen-plus", "qwen-turbo", "qwen-max", "qwq-32-preview"],
            "bigmodel": ["glm-4.7", "glm-4-plus", "glm-4-flash"],
        }
