##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to create a roadmap for its sales and marketing activities.
# To support medium and long-term planning, the company needs to estimate
# the potential value that existing customers will generate in the future.

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
# TASK OVERVIEW
###############################################################
# TASK 1: Data Preparation
           # 1. Read flo_data_20K.csv and create a copy of the dataframe.
           # 2. Define outlier_thresholds and replace_with_thresholds functions to cap outliers.
           #    Note: frequency values must be integers for CLTV calculation. Round upper/lower limits.
           # 3. Cap outliers in: order_num_total_ever_online, order_num_total_ever_offline,
           #    customer_value_total_ever_offline, customer_value_total_ever_online.
           # 4. Create new variables for each customer total order count and total spend
           #    (omnichannel = online + offline combined).
           # 5. Convert date columns to datetime format.

# TASK 2: Building the CLTV Data Structure
           # 1. Set analysis date to 2 days after the most recent purchase in the dataset.
           # 2. Build a new CLTV dataframe with:
           #    customer_id, recency_cltv_weekly, T_weekly, frequency, monetary_cltv_avg.
           #    Monetary = average value per purchase. Recency and tenure expressed in weekly units.


# TASK 3: Building BG/NBD and Gamma-Gamma Models, Calculating CLTV
           # 1. Fit the BG/NBD model.
                # a. Predict expected purchases in 3 months -> add as exp_sales_3_month.
                # b. Predict expected purchases in 6 months -> add as exp_sales_6_month.
           # 2. Fit the Gamma-Gamma model. Predict expected average transaction value -> exp_average_value.
           # 3. Calculate 6-month CLTV and add as cltv column.
                # a. Standardize CLTV values -> scaled_cltv.
                # b. Observe the top 20 customers by CLTV.

# TASK 4: Creating Segments Based on CLTV
           # 1. Divide all customers into 4 segments (A-D) based on 6-month CLTV. Add as cltv_segment.
           # 2. Provide brief 6-month action recommendations for 2 of the 4 segments.

# TASK 5: Wrap the entire process into a single function.


###############################################################
# TASK 1: Data Preparation
###############################################################
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Read the dataset and create a copy.
df_ = pd.read_csv("dataset/flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()


# 2. Define functions to detect and cap outliers.
# Note: frequency values must be integers. Round limits using round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# 3. Cap outliers in the 4 transaction and spend columns.
columns = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


# 4. Create combined omnichannel columns for total orders and total spend.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]



# 5. Convert date columns to datetime format.
df.info()
for column in df.columns:
    if 'date' in column:
        df[column] = pd.to_datetime(df[column])

# TASK 2: Building the CLTV Data Structure
  # 1. Set analysis date 2 days after the most recent purchase.

analysis_date = df["last_order_date"].max() + pd.Timedelta(2, "D")

# 2. Build the CLTV dataframe.
# Monetary = average value per purchase. Recency and T expressed in weekly units.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
cltv_df["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
cltv_df.head()


###############################################################
# TASK 3: Fitting BG/NBD and Gamma-Gamma Models, Calculating 6-Month CLTV
#
# BG/NBD    => Models expected number of future transactions
# Gamma-Gamma => Models expected average transaction value
###############################################################

# 1. Fit the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Predict expected purchases in 3 months and add to dataframe.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Predict expected purchases in 6 months and add to dataframe.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Check the top 10 buyers predicted at month 3 vs month 6 - any differences?
cltv_df.sort_values("exp_sales_3_month", ascending=False)[:10]
cltv_df.sort_values("exp_sales_6_month", ascending=False)[:10]


# 2. Fit the Gamma-Gamma model. Predict expected average transaction value -> exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3. Calculate 6-month CLTV and add to dataframe.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv
cltv_df.head()


# View the top 20 customers by predicted CLTV.
cltv_df.sort_values("cltv", ascending=False)[:20]

###############################################################
# TASK 4: Creating Customer Segments Based on CLTV
###############################################################

# 1. Divide all customers into 4 groups (A-D) based on 6-month CLTV. Assign as cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2. Is dividing into 4 groups sensible? Should it be fewer or more? Interpret.
cltv_df.groupby("cltv_segment").agg({"cltv": ["count", "mean", "std", "median"]})

# cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 5, labels=["E", "D", "C", "B", "A"])
# cltv_df.groupby("cltv_segment").agg({"cltv": ["count", "mean", "std", "median"]})



# 3. Provide brief 6-month action recommendations for 2 of the 4 segments.


###############################################################
# BONUS: Wrap the entire pipeline into a single function
###############################################################

def create_cltv_df(dataframe):

    # Data Preparation
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Build CLTV Data Structure
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Fit BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # Fit Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Calculate 6-month CLTV
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # Segment customers by CLTV into 4 tiers (A = highest, D = lowest)
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)
cltv_df.head(10)


