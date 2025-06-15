def convert_frequent_itemsets(frequent_itemsets):
    res_set = set()
    for itemset, count in frequent_itemsets.items():
        res_set.add((itemset, count))
    return res_set


def convert_mlxtend_frequent_itemsets(frequent_itemsets, num_transactions):
    res_set = set()
    for _, (support, itemset) in frequent_itemsets.iterrows():
        count = int(support * num_transactions)
        res_set.add((itemset, count))
    return res_set


def check_frequent_itemsets_equal(frequent_itemsets, mlxtend_frequent_itemsets, num_transactions):
    s1 = convert_frequent_itemsets(frequent_itemsets)
    s2 = convert_mlxtend_frequent_itemsets(mlxtend_frequent_itemsets, num_transactions)
    return s1 == s2


def convert_rules(rules):
    res_set = set()
    for antecedent, consequent, confidence in rules:
        res_set.add((antecedent, consequent, round(confidence, 4)))
    return res_set


def convert_mlxtend_rules(rules):
    res_set = set()
    for _, (antecedent, consequent, confidence) in rules.iterrows():
        res_set.add((antecedent, consequent, round(confidence, 4)))
    return res_set


def check_rules_equal(rules, mlxtend_rules):
    s1 = convert_rules(rules)
    s2 = convert_mlxtend_rules(mlxtend_rules)
    return s1 == s2
