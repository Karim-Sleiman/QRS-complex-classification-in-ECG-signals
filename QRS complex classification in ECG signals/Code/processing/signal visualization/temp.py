import pandas as pd
import matplotlib.pyplot as plt

column_index = 11 # leads 0 to 11
start_sample = 15220 # Specify the starting sample index
end_sample = 16600 # Specify the ending sample index
figsize = (10, 6) # Specify the figsize (width, height)

df = pd.read_csv(r'X\monomorph\PVCVTECGData\1067472.csv')
df_col = df.iloc[start_sample:end_sample, column_index]

plt.figure(figsize=figsize)
plt.plot(df_col)
plt.show()