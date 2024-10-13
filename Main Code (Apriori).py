import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pyfpgrowth
import time

# storing paths to csv files for each of the stores
store_files = {
    1: ('Grocery Store', 'grocery_store.csv'),
    2: ('Pharmacy', 'pharmacy.csv'),
    3: ('Stationery Store', 'stationery_store.csv'),
    4: ('Clothing Store', 'clothing_store.csv'),
    5: ('Electronics Store', 'electronics_store.csv')
}

# store selection
print("Welcome to Apriori 2.0!")
print("User, please select your store:")
for key, value in store_files.items():
    print(f"{key}: {value[0]}")

store_choice = int(input("Enter the store number: "))
store_name, file_name = store_files[store_choice]
print(f"You have selected dataset located in {file_name}.")

# fetching the csv file for the selected store
df_transactions = pd.read_csv(file_name)
transactions = df_transactions.values.tolist()

# removing ("nan" - not a number) values from transactions
transactions = [[item for item in transaction if str(item) != 'nan'] for transaction in transactions]

# funnction to validate input percentage
def get_valid_percentage(prompt):
    while True:
        value = float(input(prompt))
        if value <= 0 or value > 100:
            print("please enter a valid percentage between 1 and 100.")
        else:
            return value / 100

# get minimum support and confidence from user with validation
min_support = get_valid_percentage("Please enter the Minimum Support in (%) you want (value from 1 to 100): ")
min_confidence = get_valid_percentage("please enter the Minimum Confidence in (%) you want (value from 1 to 100): ")

# brute force Algo - (support calculation only)
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if set(itemset).issubset(set(transaction)):
            count += 1
    return count / len(transactions)

def brute_force_frequent_itemsets(transactions, min_support):
    items = set(item for transaction in transactions for item in transaction)
    frequent_itemsets = []
    k = 1

    while True:
        candidate_itemsets = list(combinations(items, k))
        k_frequent_itemsets = []
        
        for itemset in candidate_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                k_frequent_itemsets.append((itemset, support))

        if not k_frequent_itemsets:
            break

        frequent_itemsets.extend(k_frequent_itemsets)
        k += 1

    return frequent_itemsets

# calculating brute force frequent itemsets
start_brute_force = time.time()
frequent_itemsets_brute = brute_force_frequent_itemsets(transactions, min_support)
end_brute_force = time.time()

# Apriori Algorithm
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

start_apriori = time.time()
frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
end_apriori = time.time()

# checkingif Apriori found any frequent itemsets
if frequent_itemsets_apriori.empty:
    print("No frequent itemsets found using Apriori for the given support.")
else:
    # print Apriori frequent itemsets
    print("\n---> Apriori Frequent Itemsets <---")
    for i, row in frequent_itemsets_apriori.iterrows():
        print(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.2f}")

    # generating association rules for Apriori
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
    if rules_apriori.empty:
        print("No association rules found using Apriori for the given confidence.")
    else:
        print("\nfinal Association Rules:")
        for i, row in rules_apriori.iterrows():
            antecedent = list(row['antecedents'])
            consequent = list(row['consequents'])
            confidence = row['confidence'] * 100
            support = row['support'] * 100
            print(f"Rule {i+1}: {antecedent} -> {consequent}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Support: {support:.2f}%\n")

# FP-tree Algorithm
start_fp = time.time()
patterns_fp = pyfpgrowth.find_frequent_patterns(transactions, int(min_support * len(transactions)))
end_fp = time.time()

# fp-tree Frequent Patterns
print("\n---> FP-tree Frequent Patterns <---")
for itemset, support in patterns_fp.items():
    print(f"Pattern: {itemset}, Support: {support / len(transactions):.2f}")

# handle case where no frequent patterns are found by fp-tree
if not patterns_fp:
    print("No frequent patterns found using FP-tree for the given support.")
else:
    rules_fp = pyfpgrowth.generate_association_rules(patterns_fp, min_confidence)
    if not rules_fp:
        print("No association rules found using FP-tree for the given confidence.")
    else:
        print("\nFinal fp-tree Association Rules:")
        for i, (antecedent, (consequent, confidence)) in enumerate(rules_fp.items()):
            print(f"Rule {i+1}: {antecedent} -> {consequent}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print(f"Support: {patterns_fp[antecedent] / len(transactions) * 100:.2f}%\n")

# results for Brute Force
print("\n---> Brute Force frequent Itemsets <---")
if frequent_itemsets_brute:
    for itemset, support in frequent_itemsets_brute:
        print(f"Itemset: {itemset}, Support: {support:.2f}")
else:
    print("No frequent itemsets found using Brute Force for the given support.")
print(f"Brute Force Execution Time: {end_brute_force - start_brute_force:.4f} seconds")

# comparing the performance of the three algorithms
print(f"\nPerformance Comparison:")
print(f"Brute Force: {end_brute_force - start_brute_force:.4f} seconds")
print(f"Apriori: {end_apriori - start_apriori:.4f} seconds")
print(f"FP-tree: {end_fp - start_fp:.4f} seconds")
