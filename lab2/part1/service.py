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
