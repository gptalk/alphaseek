from dataclasses import asdict
import tokentrim as tt
from tokentrim.model_map import MODEL_MAX_TOKENS

from .base import ChatClient, ChatConfig
import ollama


class OllamaClient(ChatClient):
    def __init__(
            self,
            config: ChatConfig,
            model: str = "deepseek-r1:latest",  # 设置本地模型为 deepseek-r1:latest
            trim_to_token_limit: bool = True,
            max_tokens: int = 16385  # 显式指定最大令牌数
    ) -> None:
        super().__init__(config)
        self._client = ollama.Client()  # 实例化 Ollama 客户端
        self._model = model
        self._trim = trim_to_token_limit
        self._max_tokens = max_tokens  # 新增：设置最大令牌数

    def chat_complete(self, content: str) -> str:
        self._add_message("user", content)
        idx = int(self._system_prompt is not None)
        messages = [asdict(msg) for msg in self._dialog[idx:]]

        # 使用 Ollama 客户端发起对话请求
        response = self._client.chat(
            model=self._model,
            messages=tt.trim(messages, self._model, self._system_prompt, max_tokens=self._max_tokens)  # 传入 max_tokens
        )

        # 正确提取消息内容
        result = response['message'].content if 'message' in response else "No message in response"

        # 确保结果是字符串，然后分割
        result = result.split('\n')  # 将内容按行分割成一个列表

        self._add_message("assistant", result)
        return result

    def _on_reset(self) -> None:
        self._start_idx = 0

    _client: ollama.Client
    _model: str


# 更新模型的最大 token 数量配置（这部分与 Ollama 无关，因此可以忽略）
_UPDATED = False


def _update_model_max_tokens():
    global _UPDATED
    if _UPDATED:
        return
    MODEL_MAX_TOKENS["deepseek-r1:latest"] = 16385  # 设置本地 Ollama 模型的最大 token 数量
