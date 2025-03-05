from typing import Optional, List
from logging import Logger
from datetime import datetime
import json
from itertools import accumulate

import fire
import torch
from openai import OpenAI

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_generic.features import target
from alphagen_llm.client import ChatClient, ChatConfig
from alphagen_llm.prompts.interaction import DefaultInteraction, DefaultReport
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen.utils import get_logger
from alphagen.utils.misc import pprint_arguments

from abc import ABCMeta, abstractmethod
from typing import Literal, List, Optional, Callable, Tuple, Union, overload
from dataclasses import dataclass
from logging import Logger

from alphagen.utils.logging import get_null_logger
import os
from dataclasses import asdict
import openai
from openai import OpenAI
import requests
import tokentrim as tt
from tokentrim.model_map import MODEL_MAX_TOKENS
import httpx
import os


class DeepSeekClient(ChatClient):
    def __init__(
        self,
        client: OpenAI,
        config: ChatConfig,
        # model: str = "gpt-3.5-turbo-0125",
        model: str = 'deepseek-chat',
        trim_to_token_limit: bool = True
    ) -> None:
        super().__init__(config)
        _update_model_max_tokens()
        self._client = client
        self._model = model
        self._trim = trim_to_token_limit


    def chat_complete(self, content: str) -> str:
        self._add_message("user", content)
        idx = int(self._system_prompt is not None)
        messages = [asdict(msg) for msg in self._dialog[idx:]]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=tt.trim(messages, self._model, self._system_prompt)    # type: ignore
        )
        result: str = response.choices[0].message.content       # type: ignore
        self._add_message("assistant", result)
        return result

    def _on_reset(self) -> None:
        self._start_idx = 0

    _client: OpenAI
    _model: str


_UPDATED = False


def _update_model_max_tokens():
    global _UPDATED
    if _UPDATED:
        return
    MODEL_MAX_TOKENS["deepseek-chat"] = 16385

def build_chat(system_prompt: str, logger: Optional[Logger] = None):
    openai_client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    openai_client.model = "deepseek-chat"

    return DeepSeekClient(
        openai_client,
        ChatConfig(
            system_prompt=system_prompt,
            logger=logger
        )

    )

def build_parser(use_additional_mapping: bool = False) -> ExpressionParser:
    mapping = {
        "Max": [Greater],
        "Min": [Less],
        "Delta": [Sub]
    }
    return ExpressionParser(
        Operators,
        ignore_case=True,
        additional_operator_mapping=mapping if use_additional_mapping else None,
        non_positive_time_deltas_allowed=False
    )


def build_test_data(instruments: str, device: torch.device, n_half_years: int) -> List[Tuple[str, StockData]]:
    halves = (("01-01", "06-30"), ("07-01", "12-31"))

    def get_dataset(i: int) -> Tuple[str, StockData]:
        year = 2022 + i // 2
        start, end = halves[i % 2]
        return (
            f"{year}h{i % 2 + 1}",
            StockData(
                instrument=instruments,
                start_time=f"{year}-{start}",
                end_time=f"{year}-{end}",
                device=device
            )
        )

    return [get_dataset(i) for i in range(n_half_years)]


def run_experiment(
    pool_size: int = 20,
    n_replace: int = 3,
    n_updates: int = 20,
    without_weights: bool = False,
    contextful: bool = False,
    prefix: Optional[str] = None,
    force_remove: bool = False,
    also_report_history: bool = False
):
    """
    :param pool_size: Maximum alpha pool size
    :param n_replace: Replace n alphas on each iteration
    :param n_updates: Run n iterations
    :param without_weights: Do not report the weights of the alphas to the LLM
    :param contextful: Keep context in the conversation
    :param prefix: Output location prefix
    :param force_remove: Force remove worst old alphas
    :param also_report_history: Also report alpha pool update history to the LLM
    """

    args = pprint_arguments()

    initialize_qlib(f"~/.qlib/qlib_data/cn_data")
    instruments = "csi300"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = str(prefix) + "-" if prefix is not None else ""
    out_path = f"./out/llm-tests/interaction/{prefix}{timestamp}"
    logger = get_logger(name="llm", file_path=f"{out_path}/llm.log")
    # Ensure directories exist
    os.makedirs(out_path, exist_ok=True)

    with open(f"{out_path}/config.json", "w") as f:
        json.dump(args, f)

    data_train = StockData(
        instrument=instruments,
        start_time="2012-01-01",
        end_time="2021-12-31",
        device=device
    )
    data_test = build_test_data(instruments, device, n_half_years=3)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_test = [QLibStockDataCalculator(d, target) for _, d in data_test]

    def make_pool(exprs: List[Expression]) -> MseAlphaPool:
        pool = MseAlphaPool(
            capacity=max(pool_size, len(exprs)),
            calculator=calculator_train,
            device=device
        )
        pool.force_load_exprs(exprs)
        return pool

    def show_iteration(_, iter: int):
        print(f"Iteration {iter} finished...")

    inter = DefaultInteraction(
        parser=build_parser(),
        client=build_chat(EXPLAIN_WITH_TEXT_DESC, logger=logger),
        pool_factory=make_pool,
        calculator_train=calculator_train,
        calculators_test=calculator_test,
        replace_k=n_replace,
        force_remove=force_remove,
        forgetful=not contextful,
        no_actual_weights=without_weights,
        also_report_history=also_report_history,
        on_pool_update=show_iteration
    )
    inter.run(n_updates=n_updates)

    with open(f"{out_path}/report.json", "w") as f:
        json.dump([r.to_dict() for r in inter.reports], f)

    cum_days = list(accumulate(d.n_days for _, d in data_test))
    mean_ic_results = {}
    mean_ics, mean_rics = [], []

    def get_rolling_means(ics: List[float]) -> List[float]:
        cum_ics = accumulate(ic * tup[1].n_days for ic, tup in zip(ics, data_test))
        return [s / n for s, n in zip(cum_ics, cum_days)]

    for report in inter.reports:
        mean_ics.append(get_rolling_means(report.test_ics))
        mean_rics.append(get_rolling_means(report.test_rics))

    for i, (name, _) in enumerate(data_test):
        mean_ic_results[name] = {
            "ics": [step[i] for step in mean_ics],
            "rics": [step[i] for step in mean_rics]
        }
    
    with open(f"{out_path}/rolling_mean_ic.json", "w") as f:
        json.dump(mean_ic_results, f)


if __name__ == "__main__":
    print(sys.path) 
    fire.Fire(run_experiment)
