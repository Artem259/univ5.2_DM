from itertools import combinations
from collections import defaultdict


def apriori_algorithm(transactions, min_support=0.5):
    num_transactions = len(transactions)

    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    min_count = min_support * num_transactions
    frequent_itemsets = {item: count for item, count in item_counts.items() if count >= min_count}

    k = 2
    while True:
        new_candidates = defaultdict(int)
        frequent_items = list(frequent_itemsets.keys())

        for i in range(len(frequent_items)):
            for j in range(i + 1, len(frequent_items)):
                candidate = frequent_items[i] | frequent_items[j]
                if len(candidate) == k:
                    new_candidates[candidate] = sum(1 for t in transactions if candidate.issubset(t))

        new_frequent_itemsets = {k: v for k, v in new_candidates.items() if v >= min_count}

        if not new_frequent_itemsets:
            break

        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    return frequent_itemsets


def association_rules(frequent_itemsets, transactions, min_confidence=0.6):
    rules = []
    for itemset, support_count in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                antecedent_support = sum(1 for t in transactions if antecedent.issubset(t))
                confidence = support_count / antecedent_support if antecedent_support > 0 else 0
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules
