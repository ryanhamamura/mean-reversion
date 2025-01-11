import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

# Download SPX data 
spx = yf.download('^GSPC', period='1000d')['Close'][-720:]

# Calculate daily returns 
daily_returns = spx.pct_change()

# Function to identify streaks
def get_streaks(returns, sreak_length, streak_type='wins'):
    if streak_type == 'wins':
        streaks = (returns > 0).astype(int)
    else:
        streaks = (returns < 0).astype(int)

    # Initialize streak counter 
    streak_count = 0
    streak_length = []

    # Count consecutive streaks
    for i in range(len(streaks)):
        if streaks.iloc[i] == 1:
            streak_count += 1
        else:
            if streak_count >= streak_length:
                streak_lengths.append(streak_count)
                streak_count = 0

    # Check final streak 
    if streak_count >= streak_length:
        streak_lenghts.append(streak_count)

    return streak_lengths

# Function to analyze post-standard deviation move probabilities
