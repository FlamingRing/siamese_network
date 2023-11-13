## https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

X = []
with open("label_embedding2.txt") as f:
    for line in f.readlines():
        X.append([float(num) for num in line.split(" ")])
X = np.array(X)
# print(X.shape)
tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15).fit_transform(X)
df = pd.DataFrame()
kmeans = KMeans(n_clusters=30).fit(tsne_results)
# print(labels.shape)
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['label'] = kmeans.labels_
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue=df['label'],
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)
plt.show()
characters_df = pd.read_csv("characters2.csv")
output = characters_df[["UTF-8", "字"]]
output["class_idx"] = df["label"]
output.to_csv("characters_analysis2.csv")