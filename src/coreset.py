import math
import pandas as pd
import numpy as np
import os
import time

from techniques import LSH, OLSH, Clustering, AutoEncoder


class Coreset:
    def __init__(self, dataset=None, data_path=None, technique='uniform', name=None):
        if (dataset is None) and (not data_path):
            raise ValueError('Either dataset or data_path is needed to init Coreset')
        if technique not in ['uniform', 'lsh', 'kmeans', 'olsh']:
            raise ValueError(f'Technique {technique} is not yet implemented')
        else:
            self.technique = technique

        if not data_path:
            self.dataset = dataset
        else:
            self.data_path = data_path
            self.dataset = self._load_dataset()
        self.length = None
        self.computation_time = None
        self.lsh_sorted_dataset = None
        self.clustered_dataset = None
        self.olsh_sorted_dataset = None
        self.coreset = pd.DataFrame(columns=self.dataset.columns)

    def _load_dataset(self):
        print('Loading DF from file...')
        full_df = pd.read_csv(self.data_path)
        return full_df

    def populate(self, frac=None):
        start_time = time.time()
        self.length = math.ceil(frac * len(self.dataset))

        if self.technique == 'uniform':
            print('Populating coreset using Uniform Random Sampling...')
            self.coreset = self._generate_uniform_coreset()

        elif self.technique == 'lsh':
            print('Populating coreset using Locality Sensitive Hashing...')
            self.coreset = self._generate_lsh_coreset()

        elif self.technique == 'kmeans':
            print('Populating coreset using K-means Clustering...')
            self.coreset = self._generate_kmeans_coreset()

        elif self.technique == 'olsh':
            print('Populating coreset using Optimized LSH...')
            self.coreset = self._generate_olsh_coreset()
        return round(time.time() - start_time, 2)

    def _generate_uniform_coreset(self):
        # Randomly select row indices
        indices = np.random.choice(self.dataset.index, size=self.length, replace=False)

        # Return the selected rows as a dataframe
        return self.dataset.loc[indices]

    # LSH based
    def _generate_lsh_coreset(self):
        # Generate new sorted dataset if not already cached
        if self.lsh_sorted_dataset is None:
            lsh = LSH(self.dataset, num_bits=5)
            self.lsh_sorted_dataset = lsh.importance_rank()
        return self.lsh_sorted_dataset[:self.length]

    # K-means based
    def _generate_kmeans_coreset(self):
        k = 5
        if self.clustered_dataset is None:
            km = Clustering(self.dataset, k=k)
            self.clustered_dataset = km.assign_to_clusters()

        coreset = pd.DataFrame(columns=self.dataset.columns)
        for c in self.clustered_dataset['Cluster'].unique():
            mask = self.clustered_dataset['Cluster'] == c
            c_df = self.clustered_dataset[mask]
            sampled_set = c_df.loc[np.random.choice(c_df.index, size=int(self.length/k), replace=False)]
            coreset = pd.concat([coreset, sampled_set])
        coreset = coreset.drop(columns='Cluster')
        return coreset

    # AutoEncoder based
    def _generate_ae_coreset(self):
        print('To be implemented')
        return self.coreset

    # Optimized-LSH based
    def _generate_olsh_coreset(self):
        olsh = OLSH(self.dataset, num_bits=5)
        # Generate new sorted dataset if not already cached
        if self.olsh_sorted_dataset is None:
            self.olsh_sorted_dataset = olsh.importance_rank()
        return olsh.epsilon_greedy_select(self.olsh_sorted_dataset, self.length)

    def save(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return self.coreset.to_csv(file_path, index=False, mode='w')

    def load(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coreset = pd.read_csv(file_path)
        return self.coreset

