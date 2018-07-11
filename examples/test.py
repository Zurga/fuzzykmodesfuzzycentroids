from itertools import combinations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fuzzykmodesfuzzycentroids import FuzzyKmodesFuzzyCentroids


def select_edible_cluster(clusters):
    edible = sorted(((cluster['e'], i) for i, cluster in clusters.items()),
                  reverse=True)[0][1]
    return clusters[edible]


if __name__ == '__main__':
    df = pd.read_csv('mushrooms.csv').dropna()
    X = df.drop('class', axis=1)
    y = df['class']
    clust = FuzzyKmodesFuzzyCentroids()
    for train, test in StratifiedKFold(4).split(X, y):
        clust.fit(X.iloc[train], y[train])
        print('Score for fold', clust.score(X.iloc[test], y[train].values))
