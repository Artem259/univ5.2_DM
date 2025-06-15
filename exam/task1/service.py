import sklearn

from classification.OneRClassifier import OneRClassifier


def OneRClassifier_info(clf: OneRClassifier):
    for rule in clf.prediction_rules_.items():
        print(f"x[{clf.best_feature_index_}] == {rule[0]}: y = {clf.classes_[rule[1]]}")
