#%%
# Import the necessary libraries
# These libraries need to be installed in our environment (the remote or local server)
import pandas as pd
import numpy as np

#%%
# Create a simple pandas DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randint(0, 2, 100)
})

#%%
# Show the first 5 rows of the DataFrame
print(df.head())

#%%
# Describe the DataFrame
print(df.describe())

#%%
# Group by column 'B' and calculate the mean of column 'A'
grouped = df.groupby('B')['A'].mean()
print(grouped)

# %%
