import pandas as pd
import numpy as np
import os

# Get the current working directory
cwd = os.getcwd()

print("Current working directory: {0}".format(cwd))
print("os.getcwd() returns an object of type {0}".format(type(cwd)))

df = pd.read_csv('loan_data.csv')

# Extract relevant columns as lists
x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)
print("Number of records: {0}".format(n))

# Initialize lists
default = [0 for _ in range(851)]
total = [0 for _ in range(851)]

# Process the data
for i in range(n):
    y[i] = int(y[i])
    default[y[i] - 300] += x[i]
    total[y[i] - 300] += 1

# Cumulative sums
for i in range(1, 551):
    default[i] += default[i - 1]
    total[i] += total[i - 1]

def log_likelihood(n, k):
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)

r = 10
dp = [[[-10**18, 0] for _ in range(551)] for _ in range(r + 1)]

for i in range(r + 1):
    for j in range(551):
        if i == 0:
            dp[i][j][0] = 0
        else:
            for k in range(j):
                if total[j] == total[k]:
                    continue
                if i == 1:
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    ll = log_likelihood(total[j] - total[k], default[j] - default[k])
                    if dp[i][j][0] < (dp[i - 1][k][0] + ll):
                        dp[i][j][0] = ll + dp[i - 1][k][0]
                        dp[i][j][1] = k

result = round(dp[r][550][0], 4)
print("Result: {0}".format(result))

k = 550
l = []

# Backtrack to find the selected values
while r >= 0:
    l.append(k + 300)
    k = dp[r][k][1]
    r -= 1

print("Selected values: {0}".format(l))
