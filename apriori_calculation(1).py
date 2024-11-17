import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('association.csv')
data = data.applymap(lambda x: 1 if x else 0)

frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=3)

# Display the results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
