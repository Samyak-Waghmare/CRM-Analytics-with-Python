###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to define customer segments and build marketing strategies based on those segments.
# Customer behaviors will be analyzed and groups will be formed based on behavioral clusters.

###############################################################
# Dataset Description
###############################################################

# The dataset contains historical shopping behavior of customers who made their last purchases
# in 2020-2021 via OmniChannel (both online and offline).

# master_id             : Unique customer ID
# order_channel         : Channel used for shopping (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel    : Channel used for the most recent purchase
# first_order_date      : Date of the customer first purchase
# last_order_date       : Date of the customer most recent purchase
# last_order_date_online  : Date of the most recent online purchase
# last_order_date_offline : Date of the most recent offline purchase
# order_num_total_ever_online  : Total number of purchases made online
# order_num_total_ever_offline : Total number of purchases made offline
# customer_value_total_ever_offline : Total amount spent on offline purchases
# customer_value_total_ever_online  : Total amount spent on online purchases
# interested_in_categories_12 : List of categories the customer shopped in the last 12 months

###############################################################
# TASK 1: Data Understanding and Preparation
###############################################################
# Import Libraries:
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)


# TASK 1.
# 1. Read flo_data_20K.csv and create a copy of the dataframe.
df_ = pd.read_csv("dataset/flo_data_20k.csv")
df = df_.copy()
df.head()


# 2. Examine the dataset:
        # a. First 10 observations
        # b. Variable names
        # c. Shape
        # d. Descriptive statistics
        # e. Missing values
        # f. Variable types

df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.dtypes
df.info()

# 3. Omnichannel means customers shop from both online and offline platforms.
# Create new variables for each customer total number of purchases and total spend.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 4. Examine variable types. Convert date columns to datetime format.
# The if date in column condition checks whether the substring date is part of the column name.
# Each identified column is then converted using pd.to_datetime().

for column in df.columns:
    if 'date' in column:
        df[column] = pd.to_datetime(df[column])
df.info()

# 5. Examine the distribution of customer count, total products ordered, and total spend by order channel.
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

# 6. List the top 10 customers by total revenue.
df.sort_values(by="customer_value_total", ascending=False).head(10)

# 7. List the top 10 customers by total number of orders.
df.sort_values(by="order_num_total", ascending=False).head(10)

# 8. Wrap the data preparation steps into a reusable function.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe

###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Set analysis date to 2 days after the most recent purchase in the dataset.
df['last_order_date'].max() # Timestamp 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)
# Alternative:
# analysis_date = df["last_order_date"].max() + pd.Timedelta(2, "D")

# Create a new RFM dataframe with customer_id, recency, frequency, and monetary values.
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################
# Convert Recency, Frequency, and Monetary metrics to scores 1-5 using qcut,
# and save them as recency_score, frequency_score, and monetary_score.

rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])

# Create RF Score by combining recency and frequency scores.
rfm["RF_Score"] = (rfm["recency_score"].astype(str)+ rfm["frequency_score"].astype(str))
rfm.head()

###############################################################
# TASK 4: Defining Segments from RF Scores
###############################################################
# Map RF scores to human-readable segment names using a regex-based segment map.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_Score'].replace(seg_map, regex=True)
rfm.head()

###############################################################
# TASK 5: Action Time!
###############################################################

# 1. Examine the average recency, frequency, and monetary values per segment.
rfm[["segment","recency","frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 2. Using RFM analysis, find customers matching 2 specific profiles and save their IDs to CSV.

# a. FLO is introducing a new premium women shoe brand.
# Target: loyal customers (champions + loyal_customers) who shopped in the women category (KADIN).
# Save customer IDs to: target_customers_new_brand.csv

target_segments_customer_ids = rfm.loc[rfm["segment"].isin(["champions", "loyal_customers"]), "customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("target_customers_new_brand.csv", index=False)
cust_ids.shape  # 2497 customers

# b. A ~40% discount is planned on Men (ERKEK) and Children (COCUK) product categories.
# Target: formerly good customers who have gone inactive + newly acquired customers.
# Segments: cant_loose, hibernating, new_customers
# Save customer IDs to: target_customers_discount.csv

target_cust_id = rfm.loc[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"]), "customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))
                                                                      | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("target_customers_discount.csv", index=False)
cust_ids.shape

###############################################################
# BONUS: Wrap the entire pipeline into a single function
###############################################################

def create_rfm(dataframe):
    # Data Preparation
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Calculate RFM Metrics
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).dt.days
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # Calculate RF and RFM Scores
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

    # Map Scores to Segment Names
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)
