import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm


# Title of the app
st.title("Risk Management Trading App")

# Sidebar for ticker symbol input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL', max_chars=10)

# Fetch company information
company_info = yf.Ticker(ticker).info
company_name = company_info.get("shortName", "Unknown Company")

# Date range buttons
time_ranges = {
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "2 Years": timedelta(days=730),
    "5 Years": timedelta(days=1825)
}


# Default time range
selected_range = "1 Year"

# Horizontal layout for buttons under the chart
st.markdown("<br>", unsafe_allow_html=True)  # Adding a line break for spacing
cols = st.columns(len(time_ranges))

# Create buttons for each time range
for i, key in enumerate(time_ranges.keys()):
    if cols[i].button(key):
        selected_range = key

# Calculate the start date based on the selected time range
end_date = datetime.now()
start_date = end_date - time_ranges[selected_range]

end = datetime.now()
start = end - timedelta(days=365*10)

# Fetching data
data = yf.download(ticker, start=start_date, end=end_date)

data2 = yf.download(ticker, start=start, end=end)


# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])

close_price = data2['Adj Close'].iloc[-1]

# Update layout for better visualization
fig.update_layout(title=f'Candlestick Chart for {company_name} ({ticker}) - {selected_range} - Last Price: $ {close_price:.2f}',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Display the chart in the Streamlit app
st.plotly_chart(fig)

# Calculate performance for specified periods
def calculate_performance(data, period_days):
    if len(data) < period_days:
        return None
    return (data['Close'][-1] - data['Close'][-period_days]) / data['Close'][-period_days] * 100

performances = {
    "1 Month": calculate_performance(data, 30),
    "3 Months": calculate_performance(data, 90),
    "6 Months": calculate_performance(data, 180),
    "1 Year": calculate_performance(data, 365)
}

# Display performances
st.markdown("### Performance")
performance_cols = st.columns(len(performances))
for i, (label, performance) in enumerate(performances.items()):
    if performance is not None:
        color = "green" if performance > 0 else "red"
        performance_cols[i].markdown(f"<span style='color:{color}'>{label}: {performance:.2f}%</span>", unsafe_allow_html=True)
    else:
        performance_cols[i].markdown(f"{label}: N/A")

# Fetching monthly data
monthly_data = yf.download(ticker, start=start, end=end, interval='1mo')

# Calculate Average Positive and Negative Monthly Returns
monthly_returns = monthly_data['Close'].pct_change()
positive_monthly_returns = monthly_returns[monthly_returns > 0]
negative_monthly_returns = monthly_returns[monthly_returns < 0]

average_positive_monthly_return = positive_monthly_returns.mean() * 100
average_negative_monthly_return = negative_monthly_returns.mean() * 100

# Calculate Percentage of Positive and Negative Months
total_months = len(monthly_returns)
positive_months = len(positive_monthly_returns)
negative_months = len(negative_monthly_returns)

percentage_positive_months = (positive_months / total_months) * 100
percentage_negative_months = (negative_months / total_months) * 100

# Calculate Risk Reward Profile
if average_negative_monthly_return != 0:
    risk_reward_ratio = abs(average_positive_monthly_return / average_negative_monthly_return)
else:
    risk_reward_ratio = np.nan  # Avoid division by zero


# Calculate the 75th and 25th percentiles
percentile_75 = monthly_returns.quantile(0.95)
percentile_25 = monthly_returns.quantile(0.05)

# Calculate the average return above the 75th percentile
average_above_75th = monthly_returns[monthly_returns > percentile_75].mean()

# Calculate the average return below the 25th percentile
average_below_25th = monthly_returns[monthly_returns < percentile_25].mean()

# Calculate the risk-reward ratio
if average_below_25th != 0:
    risk_reward_ratio_percentiles = average_above_75th / abs(average_below_25th)
else:
    risk_reward_ratio_percentiles = np.nan  # Avoid division by zero

# Calculate Mathematical Expectation
average_win = average_positive_monthly_return / 100
average_loss = abs(average_negative_monthly_return / 100)
winning_percentage = percentage_positive_months / 100
losing_percentage = percentage_negative_months / 100

expectation = (average_win * winning_percentage) - (average_loss * losing_percentage)


# Display monthly performance metrics
st.markdown("### Monthly Performance Metrics")
st.markdown(f"**Average Positive Monthly Return**: {average_positive_monthly_return:.2f}%")
st.markdown(f"**Average Negative Monthly Return**: {average_negative_monthly_return:.2f}%")
st.markdown(f"**Percentage of Positive Months**: {percentage_positive_months:.2f}%")
st.markdown(f"**Percentage of Negative Months**: {percentage_negative_months:.2f}%")
st.markdown(f"**Risk Reward Profile**: {risk_reward_ratio:.2f}x")
st.markdown(f"**CVar Risk-Reward Ratio**: {risk_reward_ratio_percentiles:.2f}")
st.markdown(f"**Mathematical Expectation (E(x))**: {expectation:.4f}")

data_spy = yf.download(tickers='SPY', start=start, end=end)
data_qqq = yf.download(tickers='QQQ', start=start, end=end)

data_spy['DailyReturn'] = data_spy['Adj Close'].pct_change().dropna()
data_qqq['DailyReturn'] = data_qqq['Adj Close'].pct_change().dropna()
data['DailyReturn'] = data['Adj Close'].pct_change().dropna()

# Calculate the correlation with SPY and QQQ
correlation_spy = data['DailyReturn'].corr(data_spy['DailyReturn'])
correlation_qqq = data['DailyReturn'].corr(data_qqq['DailyReturn'])

# Display the correlation with SPY and QQQ
st.markdown("### Correlation with Market Indices")
st.markdown(f"**Correlation with S&P 500**: {correlation_spy:.1%}")
st.markdown(f"**Correlation with Nasdaq 100**: {correlation_qqq:.1%}")

# Calculate daily returns
data2['DailyReturn'] = data2['Close'].pct_change()

# Calculate average return by day of the week
data2['DayOfWeek'] = data2.index.dayofweek
average_return_by_day_of_week = data2.groupby('DayOfWeek')['DailyReturn'].mean() * 100
average_return_by_day_of_week.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Calculate average return by day of the month
data2['DayOfMonth'] = data2.index.day
average_return_by_day_of_month = data2.groupby('DayOfMonth')['DailyReturn'].mean() * 100

# Create bar chart for average return by day of the week
fig4 = go.Figure(data=[go.Bar(x=average_return_by_day_of_week.index, y=average_return_by_day_of_week)])
fig4.update_layout(title='Average Return by Day of the Week',
                   xaxis_title='Day of the Week',
                   yaxis_title='Average Return (%)')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig4)

# Create bar chart for average return by day of the month
fig5 = go.Figure(data=[go.Bar(x=average_return_by_day_of_month.index, y=average_return_by_day_of_month)])
fig5.update_layout(title='Average Return by Day of the Month',
                   xaxis_title='Day of the Month',
                   yaxis_title='Average Return (%)')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig5)

# Calculate Historical Relative Volume
data['3MonthAvgVolume'] = data['Volume'].rolling(window=90).mean()
data['RelativeVolume'] = data['Volume'] / data['3MonthAvgVolume']

# Create relative volume chart
fig2 = go.Figure(data=[go.Bar(x=data.index, y=data['RelativeVolume'])])

# Update layout for better visualization
fig2.update_layout(title='Historical Relative Volume',
                   xaxis_title='Date',
                   yaxis_title='Relative Volume')

# Display the relative volume chart in the Streamlit app
st.plotly_chart(fig2)

# Calculate volume change for specified periods
def calculate_volume_change(data, period_days):
    if len(data) < period_days:
        return None
    return (data['Volume'][-1] - data['Volume'][-period_days]) / data['Volume'][-period_days] * 100

volume_changes = {
    "1 Month": calculate_volume_change(data, 30),
    "3 Months": calculate_volume_change(data, 90),
    "6 Months": calculate_volume_change(data, 180),
    "1 Year": calculate_volume_change(data, 365)
}

# Display volume changes
st.markdown("### Volume Changes")
volume_change_cols = st.columns(len(volume_changes))
for i, (label, change) in enumerate(volume_changes.items()):
    if change is not None:
        color = "green" if change > 0 else "red"
        volume_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
    else:
        volume_change_cols[i].markdown(f"{label}: N/A")

# Calculate 30-day annualized volatility
data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
data['30DayVolatility'] = data['LogReturns'].rolling(window=30).std() * np.sqrt(252)

# Create 30-day annualized volatility chart
fig3 = go.Figure(data=[go.Scatter(x=data.index, y=data['30DayVolatility'], mode='lines')])

# Update layout for better visualization
fig3.update_layout(title='30-Day Annualized Volatility',
                   xaxis_title='Date',
                   yaxis_title='Annualized Volatility')

# Display the 30-day annualized volatility chart in the Streamlit app
st.plotly_chart(fig3)

# Calculate current 30-day annualized volatility and its changes
current_volatility = data['30DayVolatility'].iloc[-1]*100

def calculate_volatility_change(data, period_days):
    if len(data) < period_days:
        return None
    return (data['30DayVolatility'].iloc[-1] - data['30DayVolatility'].iloc[-period_days]) / data['30DayVolatility'].iloc[-period_days] * 100

volatility_changes = {
    "1 Month": calculate_volatility_change(data, 30),
    "3 Months": calculate_volatility_change(data, 90),
    "6 Months": calculate_volatility_change(data, 180),
    "1 Year": calculate_volatility_change(data, 365)
}

# Display current volatility and changes
st.markdown("### 30-Day Annualized Volatility")
st.markdown(f"**Current 30-Day Annualized Volatility**: {current_volatility:.2f}%")

volatility_change_cols = st.columns(len(volatility_changes))
for i, (label, change) in enumerate(volatility_changes.items()):
    if change is not None:
        color = "green" if change > 0 else "red"
        volatility_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
    else:
        volatility_change_cols[i].markdown(f"{label}: N/A")

# CVaR (Conditional Value at Risk) Calculation
def calculate_cvar(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = sorted_returns[:index].mean()
    return cvar

# Calculate CVaR at 95% confidence level
cvar = calculate_cvar(data2['DailyReturn'].dropna(), confidence_level=0.95)

# Position Size Calculator Inputs
st.sidebar.header("Position Size Calculator")
capital = st.sidebar.number_input("Total Capital ($)", value=10000)
risk_percentage = st.sidebar.number_input("Percentage of Capital at Risk (%)", value=1.0)

# Calculate the position size
risk_amount = (risk_percentage / 100) * capital
position_size = risk_amount / abs(cvar) / data2['Adj Close'].iloc[-1]
stop_loss = data2['Adj Close'].iloc[-1] + (data2['Adj Close'].iloc[-1] * cvar)


# Display the CVaR and position size
st.markdown("### Position Size Calculator")
st.markdown(f"**CVaR (95% Confidence Level)**: {cvar:.2%}")
st.markdown(f"**Risk Amount**: ${risk_amount:.2f}")
st.markdown(f"**Position Size**: {position_size:.2f} units")
st.markdown(f"**Stop Loss**: {stop_loss:.2f}")
