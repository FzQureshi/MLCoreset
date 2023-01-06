import random
import numpy as np

from copy import deepcopy
from abc import ABC

from sklearn.cluster import KMeans


def dot(normal_vector, row):
    constant = normal_vector[-1]
    _normal_vector = normal_vector[:-1]
    return np.dot(_normal_vector, row) + constant


class Technique(ABC):
    def __init__(self, dataset):
        self.dataset = deepcopy(dataset)
        self.description = None

    def generate_coreset(self):
        pass


class LSH(Technique):
    def __init__(self, dataset, num_bits=5):
        super().__init__(dataset)
        self.num_bits = num_bits
        self.description = '''<Technique Desc>'''

    @staticmethod
    def _generate_hyperplane(num_dims, min_vals, max_vals):
        normal_vector = []
        for i in range(num_dims):
            normal_vector.append(random.uniform(-1 * max_vals[i], max_vals[i]))
        normal_vector.append(random.uniform(-1 * max(max_vals), max(max_vals)))
        return normal_vector

    def _compute_hash(self, normal_vector):
        # Optimize using np matmul
        self.dataset['Dot'] = self.dataset.apply(lambda row: dot(normal_vector, list(row[1:])), axis=1)
        self.dataset['Partition'] = self.dataset.apply(lambda row: 0 if row['Dot'] < 0 else 1, axis=1)
        score = self.dataset['Partition'].sum() / len(self.dataset)

        self.dataset['ImpScore'] = self.dataset.apply(
            lambda row: row['ImpScore'] + score if row['Partition'] else row['ImpScore'] + 1 - score,
            axis=1)
        self.dataset = self.dataset.drop(['Dot', 'Partition'], axis=1)
        return True

    def _sort(self):
        sorted_dataset = self.dataset.sort_values(by=['ImpScore'])
        sorted_dataset = sorted_dataset.drop('ImpScore', axis=1)
        return sorted_dataset

    def _get_min_max(self, hardmin=0, hardmax=255):
        min_vals = list(self.dataset.min())[:-1]
        max_vals = list(self.dataset.max())[:-1]

        min_vals.append(hardmin)
        max_vals.append(hardmax)
        # print('Minmax: ', len(min_vals), len(max_vals))
        return min_vals, max_vals

    def importance_rank(self):
        min_vals, max_vals = self._get_min_max()

        # Initialize ImpScore as zero for all data-points
        self.dataset['ImpScore'] = 0

        # Subtract 1 for label
        num_dims = self.dataset.shape[1] - 1

        # Generate num_bits hyperplanes and use for hashing
        for i in range(self.num_bits):
            # print(f'Iteration - {num_bits}')
            norm_vector = self._generate_hyperplane(num_dims, min_vals, max_vals)
            self._compute_hash(norm_vector)
        return self._sort()


class AutoEncoder(Technique):
    def __init__(self, dataset, length):
        super().__init__(dataset)


class Clustering(Technique):
    def __init__(self, dataset, k=5):
        super().__init__(dataset)
        self.k = k
        self.description = '''<Technique Desc>'''

    def assign_to_clusters(self):
        x = self.dataset.drop(columns='label')

        k_means = KMeans(n_clusters=self.k,
                         init='k-means++',
                         n_init='auto',
                         max_iter=100,
                         tol=1e-04)

        self.dataset['Cluster'] = k_means.fit_predict(x)
        print(self.dataset.dtypes)
        return self.dataset


class OLSH(LSH):
    def __init__(self, dataset, num_bits=4):
        super().__init__(dataset)

    @staticmethod
    def _generate_hyperplane(num_dims, num_planes, **kwargs):
        normal_vectors = np.random.rand(num_dims, 2 * num_planes) - 0.5
        return normal_vectors

    def _compute_hash(self, normal_vector, threshold=0.5):
        self.dataset['Dot'] = self.dataset.apply(lambda row: np.dot(normal_vector, row[1:-1]), axis=1)
        self.dataset['Partition'] = self.dataset.apply(lambda row: 0 if row['Dot'] < 0 else 1, axis=1)
        score = self.dataset['Partition'].sum() / len(self.dataset)

        # Accept this candidate hyperplane
        if abs(2 * score - 1) <= threshold:
            self.dataset['ImpScore'] = self.dataset.apply(
                lambda row: row['ImpScore'] + score if row['Partition'] else row['ImpScore'] + 1 - score,
                axis=1)
            accept = 1

        # Reject this candidate hyperplane
        else:
            accept = 0

        self.dataset = self.dataset.drop(['Dot', 'Partition'], axis=1)
        return accept

    def importance_rank(self):
        # Subtract 1 for label
        num_dims = self.dataset.shape[1] - 1

        # Generate 2x the needed amount of hyperplanes
        normal_vectors = self._generate_hyperplane(num_dims, 2 * self.num_bits)

        # Initialize ImpScore as zero for all data-points
        self.dataset['ImpScore'] = 0
        generated_bits = 0
        for normal_vector in normal_vectors.T:
            generated_bits += self._compute_hash(normal_vector)
            if generated_bits == self.num_bits:
                break
        return self._sort()

    @staticmethod
    def epsilon_greedy_select(sorted_dataset, length):
        return sorted_dataset[:length]

