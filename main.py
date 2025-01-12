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

# Plot win streak histograms in first row
for idx, column in enumerate(win_data.columns):
    ax = fig.add_subplot(gs[0, idx])
    # Convert max value to int for bins
    max_val = (
        int(np.ceil(max(win_data[column].dropna())))
        if not win_data[column].empty
        else 2
    )
    bins = range(2, max_val + 2)
    sns.histplot(data=win_data[column], ax=ax, bins=bins, color="green")
    ax.set_title(f"{column}\nCount: {len(win_data[column].dropna())}")
    ax.set_xlabel("Streak Length")
    ax.set_ylabel("Frequency")

# Plot loss streak histograms in second row
for idx, column in enumerate(loss_data.columns):
    ax = fig.add_subplot(gs[1, idx])
    # Convert max value to int for bins
    max_val = (
        int(np.ceil(max(loss_data[column].dropna())))
        if not loss_data[column].empty
        else 2
    )
    bins = range(2, max_val + 2)
    sns.histplot(data=loss_data[column], ax=ax, bins=bins, color="red")
    ax.set_title(f"{column}\nCount: {len(loss_data[column].dropna())}")
    ax.set_xlabel("Streak Length")
    ax.set_ylabel("Frequency")

# Calculate std move probabilities for 1σ and 2σ
std_move_stats_1 = analyze_std_move_probabilities(daily_returns, 1)
std_move_stats_2 = analyze_std_move_probabilities(daily_returns, 2)

# Create probability bar plots
# 1σ moves
ax1 = fig.add_subplot(gs[2, :])
prob_data_1 = {
    "After +1σ Move": [
        std_move_stats_1["prob_up_after_up"].item(),
        std_move_stats_1["prob_down_after_up"].item(),
    ],
    "After -1σ Move": [
        std_move_stats_1["prob_up_after_down"].item(),
        std_move_stats_1["prob_down_after_down"].item(),
    ],
}

x1 = np.arange(len(prob_data_1))
width = 0.35

up_probs_1 = [prob_data_1[k][0] for k in prob_data_1.keys()]
down_probs_1 = [prob_data_1[k][1] for k in prob_data_1.keys()]

ax1.bar(x1 - width / 2, up_probs_1, width, label="Up Next Day", color="green")
ax1.bar(x1 + width / 2, down_probs_1, width, label="Down Next Day", color="red")

ax1.set_ylabel("Probability")
ax1.set_title("Next Day Probabilities After 1σ Moves")
ax1.set_xticks(x1)
ax1.set_xticklabels(prob_data_1.keys())
ax1.legend()

# 2σ moves
ax2 = fig.add_subplot(gs[3, :])
prob_data_2 = {
    "After +2σ Move": [
        std_move_stats_2["prob_up_after_up"].item(),
        std_move_stats_2["prob_down_after_up"].item(),
    ],
    "After -2σ Move": [
        std_move_stats_2["prob_up_after_down"].item(),
        std_move_stats_2["prob_down_after_down"].item(),
    ],
}

x2 = np.arange(len(prob_data_2))

up_probs_2 = [prob_data_2[k][0] for k in prob_data_2.keys()]
down_probs_2 = [prob_data_2[k][1] for k in prob_data_2.keys()]

ax2.bar(x2 - width / 2, up_probs_2, width, label="Up Next Day", color="green")
ax2.bar(x2 + width / 2, down_probs_2, width, label="Down Next Day", color="red")

ax2.set_ylabel("Probability")
ax2.set_title("Next Day Probabilities After 2σ Moves")
ax2.set_xticks(x2)
ax2.set_xticklabels(prob_data_2.keys())
ax2.legend()

plt.tight_layout()

# Save the plot instead of showing it if in a non-interactive environment
plt.savefig("market_analysis.png")
plt.close()

# Print summary statistics
print("\nWin Streak Summary Statistics:")
for length in ["2-Day", "3-Day", "4-Day"]:
    print(f"\n{length} win streaks:")
    print(f"Count: {len(win_streaks[length])}")
    if win_streaks[length]:
        print(f"Average length: {pd.Series(win_streaks[length]).mean():.2f} days")
        print(f"Max length: {max(win_streaks[length])} days")

print("\nLoss Streak Summary Statistics:")
for length in ["2-Day", "3-Day", "4-Day"]:
    print(f"\n{length} loss streaks:")
    print(f"Count: {len(loss_streaks[length])}")
    if loss_streaks[length]:
        print(f"Average length: {pd.Series(loss_streaks[length]).mean():.2f} days")
        print(f"Max length: {max(loss_streaks[length])} days")

print("\n1-Standard Deviation Move Analysis:")
print(f"Number of +1σ moves: {std_move_stats_1['count_sig_up']}")
print(f"Number of -1σ moves: {std_move_stats_1['count_sig_down']}")
print(f"\nAfter +1σ move:")
print(f"Probability of up next day: {std_move_stats_1['prob_up_after_up']:.2%}")
print(f"Probability of down next day: {std_move_stats_1['prob_down_after_up']:.2%}")
print(f"Average next day return: {std_move_stats_1['avg_return_after_up']:.2%}")
print(f"\nAfter -1σ move:")
print(f"Probability of up next day: {std_move_stats_1['prob_up_after_down']:.2%}")
print(f"Probability of down next day: {std_move_stats_1['prob_down_after_down']:.2%}")
print(f"Average next day return: {std_move_stats_1['avg_return_after_down']:.2%}")

print("\n2-Standard Deviation Move Analysis:")
print(f"Number of +2σ moves: {std_move_stats_2['count_sig_up']}")
print(f"Number of -2σ moves: {std_move_stats_2['count_sig_down']}")
print(f"\nAfter +2σ move:")
print(f"Probability of up next day: {std_move_stats_2['prob_up_after_up']:.2%}")
print(f"Probability of down next day: {std_move_stats_2['prob_down_after_up']:.2%}")
print(f"Average next day return: {std_move_stats_2['avg_return_after_up']:.2%}")
print(f"\nAfter -2σ move:")
print(f"Probability of up next day: {std_move_stats_2['prob_up_after_down']:.2%}")
print(f"Probability of down next day: {std_move_stats_2['prob_down_after_down']:.2%}")
print(f"Average next day return: {std_move_stats_2['avg_return_after_down']:.2%}")
