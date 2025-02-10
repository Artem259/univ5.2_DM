import numpy as np
import sklearn

from classification.OneRClassifier import OneRClassifier
from classification.NaiveBayesClassifier import NaiveBayesClassifier


def OneRClassifier_info(clf: OneRClassifier):
    for rule in clf.rules_.items():
        print(f"x[{clf.attr_i_}] == {rule[0]}: y = {clf.classes_[rule[1]]}")


def sklearn_DecisionTreeClassifier_info(clf: sklearn.tree.DecisionTreeClassifier):
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    class_labels = clf.classes_

    # Root node (node 0)
    root_feature = feature[0]
    root_threshold = threshold[0]

    # Left child (x[root_feature] <= root_threshold)
    left_class_idx = np.argmax(values[children_left[0]])
    left_class_label = class_labels[left_class_idx]
    print(f"x[{root_feature}] <= {root_threshold:.1f}: y = {left_class_label}")

    # Right child (x[root_feature] > root_threshold)
    right_class_idx = np.argmax(values[children_right[0]])
    right_class_label = class_labels[right_class_idx]
    print(f"x[{root_feature}] > {root_threshold:.1f}: y = {right_class_label}")


def NaiveBayesClassifier_info(clf: NaiveBayesClassifier):
    print(*clf.attr_probs_, sep="\n")
    print(*clf.attr_missing_probs_)
    print(clf.class_probs_)


def sklearn_CategoricalNB_info(clf: sklearn.naive_bayes.CategoricalNB):
    print(clf.feature_log_prob_)
    print(clf.class_log_prior_)
