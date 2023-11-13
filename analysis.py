import numpy as np
import pandas as pd
df = []
with open("label_embedding3.txt", mode="r") as f:
    for line in f.readlines():
        df.append([float(num) for num in line.split(" ")])
df = pd.DataFrame(df, columns=[f"dim{dim}" for dim in range(1, 13)])
df[df<0.5] = 0
df[df>=0.5] = 1
# https://www.statology.org/pandas-find-duplicates/
duplicateRows = df[df.duplicated()]
example = []
counter = 0
# for i in range(4096):

# print(np.sum(df.duplicated().tolist()))
