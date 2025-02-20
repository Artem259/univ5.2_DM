import sklearn

from classification.OneRClassifier import OneRClassifier
from classification.NaiveBayesClassifier import NaiveBayesClassifier
from classification.DecisionTreeClassifier import DecisionTreeClassifier
from classification.KNeighborsClassifier import KNeighborsClassifier


def OneRClassifier_info(clf: OneRClassifier):
    for rule in clf.prediction_rules_.items():
        print(f"x[{clf.best_feature_index_}] == {rule[0]}: y = {clf.classes_[rule[1]]}")


def NaiveBayesClassifier_info(clf: NaiveBayesClassifier):
    info = {
        "class_log_probs_": clf.class_log_probs_,
        "feature_log_probs_": clf.feature_log_probs_,
        "feature_missing_log_probs_": clf.feature_missing_log_probs_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def sklearn_CategoricalNB_info(clf: sklearn.naive_bayes.CategoricalNB):
    info = {
        "class_log_prior_": clf.class_log_prior_,
        "feature_log_prob_": clf.feature_log_prob_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def DecisionTreeClassifier_info(clf: DecisionTreeClassifier):
    ...  # TODO


def KNeighborsClassifier_info(clf: KNeighborsClassifier, X_pred):
    neigh_dist, neigh_ind = clf.kneighbors(X_pred)
    info = {
        "neighbors_dist": neigh_dist,
        "neighbors_indices": neigh_ind,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def sklearn_KNeighborsClassifier_info(clf: sklearn.neighbors.KNeighborsClassifier, X_pred):
    neigh_dist, neigh_ind = clf.kneighbors(X_pred)
    info = {
        "neighbors_dist": neigh_dist,
        "neighbors_indices": neigh_ind,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")
