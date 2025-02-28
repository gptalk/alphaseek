import re

# 操作符列表
operators = [
    'Abs', 'Sign', 'Log', 'CSRank',
    'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Greater', 'Less',
    'Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min',
    'Med', 'Mad', 'Rank', 'Delta', 'WMA', 'EMA',
    'Cov', 'Corr'
]

# 使用正则表达式动态生成匹配操作符的模式
# 匹配格式：操作符(参数...)，参数中允许包含字母、数字、下划线、逗号、空格、和单位符号
operator_pattern = r'\b(?:' + '|'.join(operators) + r')\((?:[a-zA-Z0-9_,. ]+)\)'

# 编译正则表达式
pattern = re.compile(operator_pattern)

# 示例输入数据（用户提供的文本）
lines = [
    '<think>',
    'Alright, I need to help the user by generating 20 formulaic alphas based on their specifications and limits. Let me first understand what they\'re asking for.',
    '',
    'Here are 20 formulaic alphas that may indicate future stock price trends:',
    '',
    '1. `Abs(Sub(Close, Ref(EMA(Open, 30d), 14d)))`',
    '2. `Greater(Log(High), Sub(Low, 0.5))`',
    '3. `(Max(Vol, 30d) - Min(Vol, 30d)) / Mean(Vol, 30d)`',
    '4. `EMA(Delta(Close, 1d), 20d)`',
    '5. `WMA(Ref(Volume, 10d), 30d)`',
    '6. `(Greater(Max(High, 14d), Min(Low, 14d)) ? 1 : 0)`',
    '7. `(EMA(Close, 25d) - EMA(Close, 10d)) / EMA(Close, 10d)`',
    '8. `VWAP(Open, Volume, 30d)`',
    '9. `Corr(Volume, Close, 20d)`',
    '10. `(Med(High, 5d) - Med(Low, 5d)) / Mean(Close, 5d)`',
    '11. `Delta(EMA(Close, 50d), 20d)`',
    '12. `Cov(Ref(Volume, 10d), Open, 30d)`',
    '13. `(EMA(High, 14d) - EMA(Low, 14d)) / EMA(Close, 14d)`',
    '14. `(RSI(High, 14d) < RSI(Low, 14d)) ? 1 : 0`',
    '15. `EMA(EMA(Ref(Volume, 25d), 50d), 100d)`',
    '16. `Cov(Open, Close, 20d)`',
    '17. `(Med(High, 30d) - Med(Low, 30d)) / Std(Close, 30d)`',
    '18. `Corr(Volume, Open, 50d)`',
    '19. `(MAD(High, 20d) < MAD(Low, 20d)) ? 1 : 0`',
    '20. `EMA(Theta(Close), 25d)`',
    '',
    'Each alpha is designed to capture specific aspects of price action and volume dynamics that may correlate with future trends.'
]

# 提取符合规范的表达式
valid_expressions = []

for line in lines:
    # 查找符合规范的表达式
    matches = pattern.findall(line)
    if matches:
        valid_expressions.extend(matches)

# 去重并转换为列表
unique_expressions = list(set(valid_expressions))

# 输出符合规范的表达式
print("符合规范的表达式:", unique_expressions)
