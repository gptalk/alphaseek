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
operator_pattern = r'(' + '|'.join(operators) + r')\([^\)]*\)'

# 示例输入数据（用户提供的文本）
lines = ['<think>',
         'Alright, I need to help the user by generating 20 formulaic alphas based on their specifications and limits. Let me first understand what they\'re asking for.',
         '',
         'The user is an expert quant researcher developing formulaic alphas. These alphas are mathematical expressions using specific operators and input features like open, close, high, low, volume, vwap. The goal is to create indicators that might predict future stock price trends.', '', 'First, I should recall the list of allowed operators: Abs, Log, Add, Sub, Mul, Div, Greater, Less, Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Delta, WMA, EMA. Each operator has specific uses and requirements, so I need to make sure each alpha complies.', '', 'The examples given include combinations like using EMA(open,30d) or Cov(volume, open). So, I should think about common indicators that might relate to price trends.', '',
         'I\'ll start by considering trend momentum indicators. Moving Averages (MA) are fundamental. Maybe a simple MA strategy: Close today minus 20-day MA of close. That could indicate if the price is above or below its average.', '', 'Next, RSI is a popular oscillator. It measures overbought/oversold conditions. Using Ref to look back 14 days for high and low could help calculate RSI, which might predict reversals.', '', 'Bollinger Bands are another key indicator, using MA with standard deviations. The width of the bands can indicate volatility and trend strength. So, Bollinger Bandwidth could be a good alpha.', '', 'MACD is a momentum oscillator using moving averages. Calculating the difference between 25ma and 10ma on close might show momentum shifts.', '', 'VIX is often represented by IV, measuring expected volatility. Using WMA to look at open over 30 days could give an indication of market fear or calm.', '',
         'Average Directional Index (ADX) measures trend strength regardless of direction. Calculating the difference between today\'s ADX and its 25-day MA might show if the trend is strengthening or weakening.', '',
         'Elder\'s Fisher Transform is used to detect potential reversals based on volume. It could help identify changes in market sentiment.', '', 'Momentum indicators like the 10-period RSI, as I mentioned earlier, are straightforward yet effective for trend detection.', '', 'Candlestick body size (like WMA of close over 20 days) can indicate strength or weakness in a trend.', '', 'The Chande Momentum Oscillator uses high and low to measure momentum. A 3-period EMA could make it more responsive.', '', 'On-Balance Volume tracks volume changes with price movements, which might signal continuation or divergence.', '', 'The Relative Strength Index (RSI) is another oscillator; using Fast RSI on high might highlight overbought/sold conditions.', '', 'Covariance between volume and open can indicate if volume moves with price action. A 50-day Covariance might capture significant movement.', '', 'Correlation between volume and close shows how aligned they are, which could hint at market sentiment.', '', 'Median Absolute Deviation (MAD) measures volatility similar to RSI but using median, making it robust against outliers.', '', 'Theta decay in EMA can show how quickly the trend is decaying. A 20-period EMA of theta might indicate trend strength.', '', 'Variance over a period shows how spread out the data is, which could reflect market uncertainty or stability.', '', 'Lastly, combining multiple indicators like RSI and ADX with different time frames (e.g., RSI(14) on close vs. ADX 25) can create complex yet informative alphas.', '',
         'I should make sure each alpha uses valid operators and features without exceeding the allowed t values. Also, I\'ll avoid using undefined or less common indicators to ensure they\'re implementable within the given constraints.', '</think>', '',
         'Here are 20 formulaic alphas that may indicate future stock price trends:', '', '1. `Abs(Sub(Close, Ref(EMA(Open, 30d), 14d)))`  ', '2. `Greater(Log(High), Sub(Low, 0.5))`  ', '3. `(Max(Vol, 30d) - Min(Vol, 30d)) / Mean(Vol, 30d)`  ', '4. `EMA(Delta(Close, 1d), 20d)`  ', '5. `WMA(Ref(Volume, 10d), 30d)`  ', '6. `(Greater(Max(High, 14d), Min(Low, 14d)) ? 1 : 0)`  ', '7. `(EMA(Close, 25d) - EMA(Close, 10d)) / EMA(Close, 10d)`  ', '8. `VWAP(Open, Volume, 30d)`  ', '9. `Corr(Volume, Close, 20d)`  ', '10. `(Med(High, 5d) - Med(Low, 5d)) / Mean(Close, 5d)`  ', '11. `Delta(EMA(Close, 50d), 20d)`  ', '12. `Cov(Ref(Volume, 10d), Open, 30d)`  ', '13. `(EMA(High, 14d) - EMA(Low, 14d)) / EMA(Close, 14d)`  ', '14. `(RSI(High, 14d) < RSI(Low, 14d)) ? 1 : 0`  ', '15. `EMA(EMA(Ref(Volume, 25d), 50d), 100d)`  ', '16. `Cov(Open, Close, 20d)`  ', '17. `(Med(High, 30d) - Med(Low, 30d)) / Std(Close, 30d)`  ', '18. `Corr(Volume, Open, 50d)`  ', '19. `(MAD(High, 20d) < MAD(Low, 20d)) ? 1 : 0`  ', '20. `EMA(Theta(Close), 25d)`  ', '', 'Each alpha is designed to capture specific aspects of price action and volume dynamics that may correlate with future trends.']


# 匹配符合规范的表达式，格式为：操作符(参数...)
pattern = re.compile(r'\b(?:' + '|'.join(operators) + r')\([a-zA-Z,0-9,]+\)')

# 提取符合规范的表达式
valid_expressions = []

for line in lines:
    # 查找符合规范的表达式
    matches = pattern.findall(line)
    if matches:
        valid_expressions.extend(matches)

# Remove duplicates by converting the list to a set
unique_expressions = set(valid_expressions)


# 输出符合规范的表达式
print("符合规范的表达式:", unique_expressions)