"""
LLM 客户端
统一封装 GLM、Claude、OpenAI 等 LLM 调用
"""

from typing import Any, Optional
from pydantic import BaseModel
from enum import Enum
import os

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMProvider(str, Enum):
    """LLM 提供商"""
    GLM = "glm"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    KIMI = "kimi"


class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: LLMProvider = LLMProvider.GLM
    model: str = "glm-4-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 30.0


class LLMClient:
    """
    统一 LLM 客户端

    支持:
    - GLM (智谱 AI)
    - Claude (Anthropic)
    - OpenAI
    - Ollama (本地)
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        初始化 LLM 客户端

        Args:
            config: LLM 配置，如果为 None 则从环境变量读取
        """
        if config is None:
            config = self._load_config_from_env()

        self.config = config
        self._client: Any = None

        # 根据 provider 初始化客户端
        if config.provider == LLMProvider.GLM:
            self._init_glm_client()
        elif config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic_client()
        elif config.provider == LLMProvider.OPENAI:
            self._init_openai_client()
        elif config.provider == LLMProvider.OLLAMA:
            self._init_ollama_client()
        elif config.provider == LLMProvider.KIMI:
            self._init_openai_client()

    def _load_config_from_env(self) -> LLMConfig:
        """从环境变量加载配置"""
        provider = os.getenv("LLM_PROVIDER", "glm")

        if provider == "glm":
            return LLMConfig(
                provider=LLMProvider.GLM,
                model=os.getenv("GLM_MODEL", "glm-4-flash"),
                api_key=os.getenv("GLM_API_KEY"),
                base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            )
        elif provider == "anthropic":
            return LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "openai":
            return LLMConfig(
                provider=LLMProvider.OPENAI,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "ollama":
            return LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=os.getenv("OLLAMA_MODEL", "qwen2.5:72b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        elif provider == "kimi":
            return LLMConfig(
                provider=LLMProvider.KIMI,
                model=os.getenv("KIMI_MODEL", "kimi-k2.5"),
                api_key=os.getenv("KIMI_API_KEY"),
                base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
            )
        else:
            raise ValueError(f"未知的 LLM 提供商: {provider}")

    def _init_glm_client(self):
        """初始化 GLM 客户端（使用 OpenAI 兼容 API）"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://open.bigmodel.cn/api/paas/v4",
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def _init_anthropic_client(self):
        """初始化 Anthropic 客户端"""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

    def _init_openai_client(self):
        """初始化 OpenAI 客户端（支持 Kimi / OpenAI / Ollama）"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def _init_ollama_client(self):
        """初始化 Ollama 客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",  # Ollama 不需要 API key
                base_url=f"{self.config.base_url}/v1",
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def chat_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> str:
        """
        同步聊天接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数

        Returns:
            模型响应文本
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        if self.config.provider == LLMProvider.GLM:
            return self._chat_glm_sync(messages, temperature, max_tokens, **kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._chat_anthropic_sync(messages, temperature, max_tokens, **kwargs)
        elif self.config.provider in [LLMProvider.OPENAI, LLMProvider.OLLAMA, LLMProvider.KIMI]:
            return self._chat_openai_sync(messages, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"不支持的提供商: {self.config.provider}")

    def _chat_glm_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """GLM 同步聊天"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def _chat_anthropic_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Claude 同步聊天"""
        # 提取 system 消息
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
            **kwargs
        )
        return response.content[0].text

    def _chat_openai_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """OpenAI/Ollama 同步聊天"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数

        Returns:
            模型响应文本
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        if self.config.provider == LLMProvider.GLM:
            return await self._chat_glm(messages, temperature, max_tokens, **kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return await self._chat_anthropic(messages, temperature, max_tokens, **kwargs)
        elif self.config.provider in [LLMProvider.OPENAI, LLMProvider.OLLAMA, LLMProvider.KIMI]:
            return await self._chat_openai(messages, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"不支持的提供商: {self.config.provider}")

    async def _chat_glm(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """GLM 聊天"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Claude 聊天"""
        # 提取 system 消息
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
            **kwargs
        )
        return response.content[0].text

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """OpenAI/Ollama 聊天"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None
    ) -> dict[str, Any]:
        """
        带工具调用的聊天

        Args:
            messages: 消息列表
            tools: 工具定义列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            包含 content 和 tool_calls 的响应
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        timeout = timeout or self.config.timeout

        if self.config.provider == LLMProvider.GLM:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            message = response.choices[0].message
            return {
                "content": message.content,
                "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None
            }
        elif self.config.provider in [LLMProvider.OPENAI, LLMProvider.OLLAMA, LLMProvider.KIMI]:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            message = response.choices[0].message
            return {
                "content": message.content,
                "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None
            }
        else:
            # 其他 provider 的工具调用实现
            raise NotImplementedError(f"{self.config.provider} 的工具调用尚未实现")


# 全局客户端实例
_global_client: LLMClient | None = None


def get_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """
    获取 LLM 客户端实例

    Args:
        config: 可选的配置，如果提供则创建新客户端

    Returns:
        LLMClient 实例
    """
    global _global_client

    if config is not None:
        return LLMClient(config)

    if _global_client is None:
        _global_client = LLMClient()

    return _global_client
