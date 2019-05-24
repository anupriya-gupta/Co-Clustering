## Reference : https://scikit-learn.org/stable/auto_examples/bicluster/plot_bicluster_newsgroups.html
## Language : Python3

import glob
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.metrics import consensus_score
from collections import Counter, defaultdict
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.feature_extraction.text import TfidfVectorizer

def shuffle_data(data_matrix, document_names, feature_names):
    data_matrix, document_names = shuffle(data_matrix, document_names)
    data_matrix, feature_names = shuffle(data_matrix.transpose(), feature_names)

    return data_matrix.transpose() , document_names, feature_names


def view_dataset(data_matrix):
    plt.spy(data_matrix.todense())
    plt.title("Shuffled dataset")
    plt.show()
    plt.spy(data_matrix.todense(), markersize=1)
    plt.title("Enlarged Shuffled dataset")
    plt.show()


def create_matrix(filepaths, categories):
    docs = []
    for item in filepaths:
      with open(item) as file:
        docs.append(file.read())

    vectorizer = TfidfVectorizer(stop_words='english', min_df=3, use_idf=True)
    data_matrix = vectorizer.fit_transform(docs)
    document_names = [path.split("/")[-1].split('.')[0] for path in filepaths]
    feature_names = vectorizer.get_feature_names()
    
    data_matrix, document_names, feature_names = shuffle_data(data_matrix, document_names, feature_names)
    view_dataset(data_matrix)
    
    return data_matrix, document_names, feature_names


def form_biclusters(data_matrix, categories):
    bicluster = SpectralCoclustering(n_clusters=len(categories))    
    return bicluster.fit(data_matrix)


def divide_biclusters(bicluster, categories, data_matrix, document_names, feature_names):
    divided_biclusters = {}
    
    for id_ in range(len(categories)):
        n_rows, n_cols = bicluster.get_shape(id_)
        cluster_docs, cluster_words = bicluster.get_indices(id_)

        cat = defaultdict(int)
        for i in cluster_docs:
            cat[document_names[i]] += 1

        out_of_cluster_docs = bicluster.row_labels_ != id_
        out_of_cluster_docs = np.where(out_of_cluster_docs==True)[0]
        word_col = data_matrix[:, cluster_words]
        word_scores = (np.array(word_col[cluster_docs, :].sum(axis=0) -
                               word_col[out_of_cluster_docs, :].sum(axis=0))).ravel()
        important_words = list(feature_names[cluster_words[i]]
                           for i in word_scores.argsort()[:-11:-1])

        divided_biclusters[id_] = {'document_count' : n_rows,
            'word_count' : n_cols,
            'categories' : dict(Counter(cat)),
            'important_words' : important_words
            }

    return divided_biclusters


def view_biclusters(divided_biclusters, bicluster, categories, data_matrix, document_names, feature_names):
    print("Total Words:", len(feature_names))
    print("Total Docs:", len(filepaths))
    print("Biclusters:")
    print("----------------")
    
    for key, value in divided_biclusters.items():
        print("Bicluster {} : {} documents, {} words".format(key, divided_biclusters[key]['document_count'], divided_biclusters[key]['word_count']))
        print("categories   : {}".format(divided_biclusters[key]['categories']))
        print("words        : {}\n".format(', '.join(divided_biclusters[key]['important_words'])))

    fit_data = data_matrix[np.argsort(bicluster.row_labels_)]
    fit_data = fit_data[:, np.argsort(bicluster.column_labels_)]
    plt.spy(fit_data.todense())
    plt.title("Biclusters")
    plt.show()

    plt.spy(fit_data.todense(), markersize=1)
    plt.title("Enlarged Biclusters")
    plt.show()


if __name__ == '__main__':
    filepaths = glob.glob("/data/ASU/CSE571-swm/project/code/classic3/*")
    categories = ['med','cisi','cran']
    data_matrix, document_names, feature_names = create_matrix(filepaths, categories)
    bicluster = form_biclusters(data_matrix, categories)
    divided_biclusters = divide_biclusters(bicluster, categories, data_matrix, document_names, feature_names)
    view_biclusters(divided_biclusters, bicluster, categories, data_matrix, document_names, feature_names)
