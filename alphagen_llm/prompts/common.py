from enum import IntEnum
from typing import Optional, List, Tuple
from num2words import num2words
from alphagen.data.expression import Expression
from alphagen.data.expression import Operators
from alphagen.data.parser import ExpressionParser
import re


class MetricDescriptionMode(IntEnum):
    NOT_INCLUDED = 0    # Description of this metric is not included in the prompt.
    INCLUDED = 1        # Description of this metric is included in the prompt.
    SORTED_BY = 2       # Description is included, and the alphas will be sorted according to this metric.


def alpha_word(n: int) -> str:
    return "alpha" if n == 1 else "alphas"


def alpha_phrase(n: int, adjective: Optional[str] = None) -> str:
    n_word = str(n) if n > 10 else num2words(n)
    adjective = f" {adjective}" if adjective is not None else ""
    return f"{n_word}{adjective} {alpha_word(n)}"


def safe_parse(parser: ExpressionParser, expr_str: str) -> Optional[Expression]:
    try:
        return parser.parse(expr_str)
    except:
        return None


def safe_parse_list(lines: List[str], parser: ExpressionParser) -> Tuple[List[Expression], List[str]]:
    parsed, invalid = [], []

    # 假设 Operators 列表包含类类型
    Op = [op.__name__ for op in Operators]  # 将每个类转换为类名字符串

    # 匹配符合规范的表达式，格式为：操作符(参数...)
    # pattern = re.compile(r'\b(?:' + '|'.join(Op) + r')\([a-zA-Z,0-9,]+\)')
    pattern = re.compile(r'\b(?:' + '|'.join(Op) + r')\((?:[a-zA-Z0-9_,. ]+)\)')

    # 提取符合规范的表达式
    valid_expressions = []

    for line in lines:
        # 查找符合规范的表达式
        matches = pattern.findall(line)
        matches = list(set(matches))
        if matches:
            for match in matches:
                if (e := safe_parse(parser, match)) is not None:
                    parsed.append(e)
                else:
                    invalid.append(matches)
        else:
            invalid.append(matches)

    # 输出符合规范的表达式
    print("符合规范的表达式:", parsed)

    return parsed, invalid

