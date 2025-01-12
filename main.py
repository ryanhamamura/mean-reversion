import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Download SPX data
spx = yf.download("^GSPC", period="5y")["Close"][-720:]

# Calculate daily returns
daily_returns = spx.pct_change()


# Function to identify streaks
def get_streaks(returns, streak_length, streak_type="wins"):
    if streak_type == "wins":
        streaks = (returns > 0).astype(int)
    else:
        streaks = (returns < 0).astype(int)

    # Initialize streak counter
    streak_count = 0
    streak_lengths = []

    # Count consecutive streaks
    for i in range(len(streaks)):
        if streaks.iloc[i].item() == 1:
            streak_count += 1
        else:
            if streak_count >= streak_length:
                streak_lengths.append(streak_count)
                streak_count = 0

    # Check final streak
    if streak_count >= streak_length:
        streak_lengths.append(streak_count)

    return streak_lengths


# Function to analyze post-standard deviation move probabilities
def analyze_std_move_probabilities(returns, std_multiple):
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=20).std()

    # Identify significant moves
    significant_up_moves = returns > (rolling_std * std_multiple)
    significant_down_moves = returns < (-rolling_std * std_multiple)

    # Calculate next day returns after significant moves
    next_day_after_up = returns.shift(-1)[significant_up_moves]
    next_day_after_down = returns.shift(-1)[significant_down_moves]

    # Calculate probabilities
    prob_up_after_up = (
        (next_day_after_up > 0).mean() if len(next_day_after_up) > 0 else 0
    )
    prob_down_after_up = (
        (next_day_after_up < 0).mean() if len(next_day_after_up) > 0 else 0
    )
    prob_up_after_down = (
        (next_day_after_down > 0).mean() if len(next_day_after_down) > 0 else 0
    )
    prob_down_after_down = (
        (next_day_after_down < 0).mean() if len(next_day_after_down) > 0 else 0
    )

    # Calculate average returns
    avg_return_after_up = next_day_after_up.mean() if len(next_day_after_up) > 0 else 0
    avg_return_after_down = (
        next_day_after_down.mean() if len(next_day_after_down) > 0 else 0
    )

    return {
        "prob_up_after_up": prob_up_after_up,
        "prob_down_after_up": prob_down_after_up,
        "prob_up_after_down": prob_up_after_down,
        "prob_down_after_down": prob_down_after_down,
        "avg_return_after_up": avg_return_after_up,
        "avg_return_after_down": avg_return_after_down,
        "count_sig_up": len(next_day_after_up),
        "count_sig_down": len(next_day_after_down),
    }


# Get win and loss streaks for different lengths
win_streaks = {
    "2-Day": get_streaks(daily_returns, 2, "wins"),
    "3-Day": get_streaks(daily_returns, 3, "wins"),
    "4-Day": get_streaks(daily_returns, 4, "wins"),
}

loss_streaks = {
    "2-Day": get_streaks(daily_returns, 2, "losses"),
    "3-Day": get_streaks(daily_returns, 3, "losses"),
    "4-Day": get_streaks(daily_returns, 4, "losses"),
}

# Create DataFrames for plotting
win_data = pd.DataFrame(
    {
        "2-Day Win Streaks": pd.Series(win_streaks["2-Day"]),
        "3-Day Win Streaks": pd.Series(win_streaks["3-Day"]),
        "4-Day Win Streaks": pd.Series(win_streaks["4-Day"]),
    }
)

loss_data = pd.DataFrame(
    {
        "2-Day Loss Streaks": pd.Series(loss_streaks["2-Day"]),
        "3-Day Loss Streaks": pd.Series(loss_streaks["3-Day"]),
        "4-Day Loss Streaks": pd.Series(loss_streaks["4-Day"]),
    }
)

# Set up the plot style
plt.style.use("seaborn-v0_8")
fig = plt.figure(figsize=(15, 20))
gs = fig.add_gridspec(4, 3)
