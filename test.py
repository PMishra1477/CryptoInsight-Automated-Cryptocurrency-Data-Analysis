# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Create sample dataset
# This could be replaced with your actual dataset
transactions = [
    ['bread', 'milk'],
    ['bread', 'diaper', 'beer', 'eggs'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola'],
    ['bread', 'diaper', 'milk'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola'],
    ['bread', 'milk', 'eggs'],
    ['bread', 'milk', 'eggs', 'diaper']
]

print("Sample Transaction Data:")
for i, transaction in enumerate(transactions):
    print(f"Transaction {i+1}: {transaction}")

# Convert transaction data to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print("\nOne-hot encoded transaction data (first 5 rows):")
print(df.head())

# Find frequent itemsets using the Apriori algorithm
min_support = 0.3  # 30% support threshold
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
print("\nFrequent Itemsets (min_support=0.3):")
print(frequent_itemsets.sort_values('support', ascending=False))

# Generate association rules
min_confidence = 0.7  # 70% confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
print("\nAssociation Rules (min_confidence=0.7):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Visualizing the results

# 1. Visualize support distribution
plt.figure(figsize=(10, 6))
plt.bar(range(len(frequent_itemsets)), frequent_itemsets['support'], align='center')
plt.xticks(range(len(frequent_itemsets)), [str(list(itemset)) for itemset in frequent_itemsets['itemsets']], rotation=90)
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Support for Frequent Itemsets')
plt.tight_layout()
plt.savefig('support_distribution.png')

# 2. Plot scatter plot of rules by confidence and support
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence for Association Rules')

# Annotate points with rule text
for i, rule in rules.iterrows():
    antecedent = ', '.join(list(rule['antecedents']))
    consequent = ', '.join(list(rule['consequents']))
    rule_text = f"{antecedent} → {consequent}"
    plt.annotate(rule_text, 
                 (rule['support'], rule['confidence']),
                 textcoords="offset points",
                 xytext=(5,5),
                 ha='left')
    
plt.tight_layout()
plt.savefig('support_vs_confidence.png')

# 3. Visualize lift values
plt.figure(figsize=(10, 6))
plt.bar(range(len(rules)), rules['lift'], align='center')
rule_labels = [f"{', '.join(list(rule.antecedents))} → {', '.join(list(rule.consequents))}" for i, rule in rules.iterrows()]
plt.xticks(range(len(rules)), rule_labels, rotation=90)
plt.xlabel('Rules')
plt.ylabel('Lift')
plt.title('Lift Values for Association Rules')
plt.axhline(y=1, color='r', linestyle='-', alpha=0.3, label='Lift = 1 (No correlation)')
plt.legend()
plt.tight_layout()
plt.savefig('lift_values.png')

# Function to analyze the results
def analyze_results(rules):
    print("\nAnalysis of Association Rules:")
    
    # Sort rules by different metrics
    print("\nTop 3 rules by confidence:")
    print(rules.sort_values('confidence', ascending=False).head(3)[['antecedents', 'consequents', 'confidence']])
    
    print("\nTop 3 rules by lift:")
    print(rules.sort_values('lift', ascending=False).head(3)[['antecedents', 'consequents', 'lift']])
    
    print("\nTop 3 rules by support:")
    print(rules.sort_values('support', ascending=False).head(3)[['antecedents', 'consequents', 'support']])
    
    # Identify strong rules (high confidence and lift)
    strong_rules = rules[(rules['confidence'] > 0.8) & (rules['lift'] > 1.2)]
    print("\nStrong rules (confidence > 0.8 and lift > 1.2):")
    if len(strong_rules) > 0:
        print(strong_rules[['antecedents', 'consequents', 'confidence', 'lift']])
    else:
        print("No rules meet the criteria for strong rules.")
    
    # Business insights
    print("\nBusiness Insights:")
    print("1. Product Association: Identify which products are frequently purchased together")
    print("2. Store Layout: Optimize product placement based on association rules")
    print("3. Promotional Strategies: Design bundle offers for frequently co-purchased items")
    print("4. Cross-selling: Recommend related products based on high-confidence rules")
    print("5. Inventory Management: Better stock planning for commonly associated items")

# Analyze the results
analyze_results(rules)

# Additional analysis: Real-world application example
print("\nExample Application of Market Basket Analysis:")
print("Based on our analysis, if we were to apply these findings to a retail setting, we might:")

# Find the rule with the highest confidence
highest_conf_rule = rules.loc[rules['confidence'].idxmax()]
antecedent = ', '.join(list(highest_conf_rule['antecedents']))
consequent = ', '.join(list(highest_conf_rule['consequents']))
print(f"1. Place {antecedent} and {consequent} near each other in the store since they have a strong association.")

# Find items frequently purchased together (high support)
high_support_items = frequent_itemsets[frequent_itemsets['support'] > 0.6]['itemsets']
if len(high_support_items) > 0:
    items_list = [', '.join(list(items)) for items in high_support_items]
    print(f"2. Ensure consistent inventory of highly frequent items: {' and '.join(items_list)}")

# Consider bundle promotions based on lift
high_lift_rule = rules.loc[rules['lift'].idxmax()]
ant_lift = ', '.join(list(high_lift_rule['antecedents']))
cons_lift = ', '.join(list(high_lift_rule['consequents']))
print(f"3. Create a promotional bundle for {ant_lift} and {cons_lift} which have a high lift value of {high_lift_rule['lift']:.2f}")

print("\nConclusion:")
print("Market basket analysis using the Apriori algorithm effectively reveals hidden patterns in customer purchasing behavior.")
print("By identifying these associations, retailers can optimize product placement, develop targeted marketing strategies,")
print("improve recommendation systems, and enhance overall customer experience, ultimately leading to increased sales and customer satisfaction.")