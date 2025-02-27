#############################
# ASSOCIATION RULE LEARNING
#############################

# Data Pre-Processing
# Preparing ARL Data Structures
# Inferring Association Rules
# Preparing Script of Work
# Making Suggestions to Users at the Cart Stage

#region Data Pre-Processing
import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import openpyxl

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.describe().T
df.isnull().sum()
df.shape

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

replace_with_thresholds(df, variable="Price")

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, variable="Price")
    replace_with_thresholds(dataframe, variable="Quantity")
    return dataframe

df = df_.copy()
df = retail_data_prep(df)
df.isnull().sum()
df.describe().T
#endregion

#region Preparing ARL Data Structures

df.head()
df_fr = df[df["Country"] == "France"]

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head(20)

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(["Invoice", "StockCode"]).\
    agg({"Quantity": "sum"}).\
    unstack().\
    fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_prouct_df(dataframe, id=False):
    if id:
     return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_prouct_df(df_fr)
fr_inv_pro_df = create_invoice_prouct_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr,10120)
#endregion

#region Association Rules Analysis
frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules[(rules["support"] > 0.05) & (rules["confidence"] >= 0.1) & (rules["lift"] > 5)]
check_id(df_fr, 21086)
rules[(rules["support"] > 0.05) & (rules["confidence"] >= 0.1) & (rules["lift"] > 5)]. \
    sort_values("confidence", ascending=False)
#endregion

#region Preparing Script of Work

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


def create_invoice_prouct_df(dataframe, id=False):
    if id:
     return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=False, country="France"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_prouct_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()
df = retail_data_prep(df)
rules = create_rules(df)
#endregion

#region Product Recommandation Practice

# e.g
# User Product Id: 22492

product_id = 22492
check_id(df, 22492)

sorted_rules = rules.sort_values("lift", ascending=False)

recommandation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommandation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


check_id(df, 22326)

def arl_recommander(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommandation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommandation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommandation_list[0:rec_count]

arl_recommander(sorted_rules, 22492, 3)
#endregion