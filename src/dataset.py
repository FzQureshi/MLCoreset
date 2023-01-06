from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


df = pd.read_csv('/Users/faizqureshi/Documents/Github/Coresets/datasets/mnist_sampled_6000.csv')

y = df['label']
x = df.drop(columns='label')

k_means = KMeans(
    n_clusters=10, init='k-means++',
    n_init=10, max_iter=100,
    tol=1e-04
)

df['Cluster'] = k_means.fit_predict(x)

print(df.head())

print(df['Cluster'].value_counts())

coreset = pd.DataFrame(columns=df.columns)
for c in df['Cluster'].unique():
    c_df = df[df['Cluster'] == c]
    print(f'Cluster-{c}\nLength c_df = {len(c_df)}')
    sampled_set = c_df.loc[np.random.choice(c_df.index, size=120, replace=False)]
    print(f'Sampled length {len(sampled_set)}')
    coreset = pd.concat([coreset, sampled_set])


print(len(coreset))
print(coreset['Cluster'].value_counts())




