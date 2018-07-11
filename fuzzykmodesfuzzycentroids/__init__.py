import numpy as np
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


class NaNException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FuzzyKmodesFuzzyCentroids(object):
    def __init__(self, k=2, m=1.5, n_iter=100, error=0.00005):
        self.k = k
        self.m = m
        self.n_iter = n_iter
        self.u = None
        self.clusters = None
        self.exp = 1 / ( self.m - 1 )
        self.error = error

    def fit(self, X, Y, init=None, memberships=False):
        self.p = len(X.columns)
        self.n = len(X)
        self.columns = X.columns

        if not init:
            clusters = self.init(X, self.k)
            self.X = X
            self.Y = Y
        else:
            clusters = init

        u = [[self.membership(row, cluster, clusters) for row in self.X.values]
            for cluster in clusters]
        clusters = self.update_clusters(clusters, self.X, u)

        self.iteration = 0
        u_error = 1
        while self.iteration < self.n_iter and u_error > self.error:
            try:
                new_u = [[self.membership(x, cluster, clusters) for x in
                        self.X.values]
                        for cluster in clusters]
                clusters_new = self.update_clusters(clusters, self.X, new_u)
                u_error = np.mean([abs(m - new_m) for i in range(self.k)
                            for m, new_m in zip(u[i], new_u[i])])
                u = new_u
                clusters = clusters_new
            except NaNException:
                print('nan')
                break
            finally:
                self.iteration += 1
        self.u = u
        self.clusters = clusters
        if memberships:
            return self.cluster_membership()

    def random_numbers(self, values):
        randoms = np.random.dirichlet(np.ones(len(values)), size=1)
        return dict(zip(values, randoms[0]))

    def init(self, x, k):
        return [[self.random_numbers(x[attribute].unique())
                for attribute in self.columns]
                for _ in range(self.k)]

    def dissimilarity(self, fuzzy_set, value):
        return 1 - fuzzy_set.get(value, 0)
        # return sum(0 if term == value else confidence
        #        for term, confidence in fuzzy_set)

    def distance(self, cluster, row):
        return sum(self.dissimilarity(fuzzy_set, row[column])
                for column, fuzzy_set in enumerate(cluster))

    def membership(self, row, cluster, clusters):
        membership = sum((self.distance(cluster, row) /
                          self.distance(other_cluster, row))** self.exp
                         for other_cluster in clusters)
        if np.isnan(membership):
            return 0
        return membership ** -1

    def certainty(self, column, value, cluster_u):
        new_memb = sum(cluster_u[row] ** self.m if x == value else 0
                       for row, x in enumerate(column))
        return new_memb

    def update_clusters(self, clusters, X, u):
        for j, cluster in enumerate(clusters):
            cluster_u = u[j]
            for l, fuzzy_set in enumerate(cluster):
                feature = self.columns[l]
                column = X[feature].values
                clusters[j][l] = self.normalize(
                    {term: self.certainty(column, term, cluster_u)
                     for term in fuzzy_set})
        return clusters

    def check_normal(self, clusters, iter):
        clusters = [[self.normalize(row) for row in cluster]
                    for cluster in clusters]
        for cluster in clusters:
            for row in cluster:
                if not (0.9 < sum(r for r in row.values()) < 1.1):
                    print('error', row)
                    raise NaNException(str(iter) + str(cluster))
        return clusters

    def normalize(self, row):
        summed = sum(row.values())
        return {value: x / summed for value, x in row.items()}

    def predict(self, X, y):
        u = [[self.membership(row, cluster, self.clusters) for row in X.values]
             for cluster in self.clusters]

        for i in range(len(X)):
            yield max((u[cl][i], cl) for cl in range(self.k))[-1]

    def score(self, X, y):
        best_clusters = {key: self.select_max_cluster(key) for key in
                         set(y)}
        score = 0
        for i, cluster_n in enumerate(self.predict(X, y)):
            real = y[i]
            if best_clusters[real] == cluster_n:
                score += 1
        return score / len(X)

    def cluster_membership(self, u=None, Y=None):
        '''
        Determines which cluster a data-point belongs to and adds the
        corresponding Y-value to that cluster.
        '''
        if not u:
            u = self.u
        if Y is not None:
            Y = self.Y
        elif type(self.Y) is not None:
            Y = list(range(len(u[0])))

        cluster_membership = {i: list() for i in range(self.k)}

        for i in range(len(u[0])):
            cluster = max((u[cl][i], cl) for cl in range(self.k))[-1]
            cluster_membership[cluster].append(Y[i])

        return cluster_membership

    def count_values(self, cluster_membership=None):
        if not cluster_membership:
            cluster_membership = self.cluster_membership()
        return {cl: Counter(vals) for cl, vals in cluster_membership.items()}

    def plot_clusters(self):
        cluster_membership = self.cluster_membership()
        cluster_membershp = {cl: Counter(vals)
                             for cl, vals in cluster_membership.items()}
        for cl, vals in cluster_membership.items():
            plt.subplot(1, self.k, cl+ 1 )
            if vals:
                # Filter out the NaN values.
                plt.hist([v for v in vals if v ==v])
        plt.show()

    def select_max_cluster(self, key, clusters=None):
        if not clusters:
            clusters = self.count_values()
        return sorted(((cluster[key], i) for i, cluster in clusters.items()),
                      reverse=True)[0][1]
