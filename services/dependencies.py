from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.types import DoubleType
import os

# Initialize Spark session
def get_spark_session():
    spark = SparkSession.builder.appName("RecommendationAPI") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .getOrCreate()
    return spark

# Load the pre-trained ALS model
def get_als_model(spark: SparkSession):
    model_path = "als_model" # Ensure this path is correct relative to where you run the FastAPI app
    if not os.path.exists(model_path):
        print(f"ALS model not found at {model_path}. Please train and save the model first.")
        return None
    try:
        als_model = ALSModel.load(model_path)
        print("ALS model loaded successfully.")
        return als_model
    except Exception as e:
        print(f"Error loading ALS model from {model_path}: {e}")
        return None

# Define preprocessing functions (copied from your notebook)
def preprocess(df):
    # Change data types
    df = df.withColumn('event_time', to_timestamp('event_time'))
    df = df.withColumn('user_id', col('user_id').cast('integer'))
    df = df.withColumn('product_id', col('product_id').cast('integer'))
    df = df.withColumn('category_id', col('category_id').cast('long'))

    # Limit the number of carts to 1 per session for each user-product pair
    cart_df = df.filter(col('event_type') == 'cart')
    df = df.filter(col('event_type') != 'cart')
    cart_df = cart_df.dropDuplicates(subset=['product_id', 'user_id', 'user_session'])
    df = df.union(cart_df)

    # Split category codes into sub categories
    df = df.withColumn('category', split(df['category_code'], '\.').getItem(0)) \
            .withColumn('sub_category', split(df['category_code'], '\.').getItem(1)) \
            .withColumn('sub_sub_category', split(df['category_code'], '\.').getItem(2))

    return df

def product_features(df):
    # Calculate several metrics for products with the aggregate function
    df = df.groupby('product_id').agg(first('category_id').alias('category_id'),
                                      first('category_code').alias('category_code'),
                                      count(when(df['event_type'] == 'view', True)).alias('views'),
                                      count(when(df['event_type'] == 'cart', True)).alias('carts'),
                                      count(when(df['event_type'] == 'purchase', True)).alias('purchases'),
                                      mean('price').alias('price'),
                                      min('event_time').alias('first_date'),
                                      max('event_time').alias('last_date'))

    # Calculate interaction rates
    df = df.withColumn('purchase_per_view', df['purchases'] / df['views'])
    df = df.withColumn('cart_per_view', df['carts'] / df['views'])
    df = df.withColumn('purchase_per_cart', when(df['carts'] == 0, df['purchases']).otherwise(df['purchases'] / df['carts']))

    return df

def category_features(df):
    # Calculate the average product price for each category
    products = df.dropDuplicates(subset=['product_id'])
    products = products.groupby('category_id').agg(avg('price').alias('average_price'))

    # Calculate several metrics for categories with the aggregate function
    df = df.groupby('category_id').agg(first('category_code').alias('category_code'),
                                       countDistinct('product_id').alias('number_of_products'),
                                       count(when(df['event_type'] == 'view', True)).alias('views'),
                                       count(when(df['event_type'] == 'cart', True)).alias('carts'),
                                       count(when(df['event_type'] == 'purchase', True)).alias('purchases'))

    # Calculate interaction rates
    df = df.withColumn('purchase_per_view', df['purchases'] / df['views'])
    df = df.withColumn('cart_per_view', df['carts'] / df['views'])
    df = df.withColumn('purchase_per_cart', when(df['carts'] == 0, df['purchases']).otherwise(df['purchases'] / df['carts']))

    df = df.join(products, on='category_id')

    return df

def user_features(df):
    # Calculate several metrics for users with the aggregate function
    df = df.groupby('user_id').agg(count(when(df['event_type'] == 'view', True)).alias('views'),
                                   count(when(df['event_type'] == 'cart', True)).alias('carts'),
                                   count(when(df['event_type'] == 'purchase', True)).alias('purchases'),
                                   countDistinct(when(df['event_type'] == 'view', col('product_id'))).alias('distinct_products_viewed'),
                                   countDistinct(when(df['event_type'] == 'cart', col('product_id'))).alias('distinct_products_carted'),
                                   countDistinct(when(df['event_type'] == 'purchase', col('product_id'))).alias('distinct_products_purchased'),
                                   mean(when(df['event_type'] == 'view', col('price'))).alias('average_price_viewed'),
                                   mean(when(df['event_type'] == 'purchase', col('price'))).alias('average_price_purchased'),
                                   mean(when(df['event_type'] == 'view', col('relative_price'))).alias('avg_relative_price_viewed'),
                                   mean(when(df['event_type'] == 'purchase', col('relative_price'))).alias('avg_relative_price_purchased'),
                                   min('event_time').alias('first_date'),
                                   max('event_time').alias('last_date'))

    # Calculate interaction rates
    df = df.withColumn('purchase_per_view', when(df['views'] == 0, df['purchases']).otherwise(df['purchases'] / df['views']))
    df = df.withColumn('cart_per_view', when(df['views'] == 0, df['carts']).otherwise(df['carts'] / df['views']))
    df = df.withColumn('purchase_per_cart', when(df['carts'] == 0, df['purchases']).otherwise(df['purchases'] / df['carts']))

    return df

def category_smoothener(categories, mean, attr, rate, min_sample_size=1000):
    categories = categories.withColumn(rate, when(categories[attr] < min_sample_size, ((categories[rate] * categories[attr]) + (mean * (min_sample_size - categories[attr]))) / min_sample_size).otherwise(categories[rate]))
    return categories

def product_smoothener(products, categories, attr, rate, min_sample_size=1000):
    category_rate = rate + '_cat'
    categories = categories.withColumnRenamed(rate, category_rate)
    products = products.join(categories['category_id', category_rate], on='category_id')
    products = products.withColumn(rate, when(products[attr] < min_sample_size, ((products[rate] * products[attr]) + (products[category_rate] * (min_sample_size - products[attr]))) / min_sample_size).otherwise(products[rate]))
    products = products.drop(category_rate)
    return products

def calculate_relative_price(products):
    categories = products.groupby('category_id').agg(percentile_approx('price', 0.25, 1000).alias('Q1'),
                                                     percentile_approx('price', 0.5, 1000).alias('median'),
                                                     percentile_approx('price', 0.75, 1000).alias('Q3'))
    categories = categories.withColumn('IQR', col('Q3') - col('Q1'))
    categories = categories.withColumn('IQR', when(col('IQR') < 1, 1).otherwise(col('IQR')))
    products = products.join(categories, on='category_id')
    products = products.withColumn('relative_price', (col('price') - col('median')) / col('IQR'))
    products = products.withColumn('relative_price', when(col('relative_price') > 5, 5).otherwise(col('relative_price')))
    products = products.withColumn('relative_price', when(col('relative_price') < -5, -5).otherwise(col('relative_price')))
    products = products.select('product_id', 'relative_price')
    return products

def calculate_interaction_matrix(df, view_weight=0.1, cart_weight=0.3, purchase_weight=1.0):
    # Get the timestamp of the most recent event
    last_date = df.agg(max('event_time')).collect()[0][0]
    df = df.withColumn('last_date', lit(last_date))

    # Calculate the recency of each event in terms of days
    df = df.withColumn('recency', (col('last_date').cast('double') - col('event_time').cast('double')) / 86400)
    df = df.drop('last_date')

    # Half-life decay function
    df = df.withColumn('recency_coef', expr('exp(ln(0.5)*recency/20)'))

    # Find the number of views, carts and purchases for each user-product pair
    interactions = df.groupby(['user_id', 'product_id']).agg(
        sum(when(df['event_type'] == 'view', 1) * df['recency_coef']).alias('views'),
        sum(when(df['event_type'] == 'cart', 1) * df['recency_coef']).alias('carts'),
        sum(when(df['event_type'] == 'purchase', 1) * df['recency_coef']).alias('purchases')
    )
    interactions = interactions.na.fill(0)

    # Create a new column with the weighted interaction value
    df = interactions.withColumn('interaction', view_weight * col('views') + cart_weight * col('carts') + purchase_weight * col('purchases'))

    # Use log10 value for views, carts and purchases
    df = df.withColumn('interaction', log10(col('interaction') + 1))

    # Set the max possible value to 100 (log100 = 2)
    df = df.withColumn('interaction', when(col('interaction') > 2, 2).otherwise(col('interaction')))

    return df


# Load and process dataframes
def load_and_process_data(spark: SparkSession, file_path: str):
    try:
        # Load the raw data
        df = spark.read.option('header', True).csv(file_path)
        print("Raw data loaded successfully.")

        # Preprocess the data
        df = preprocess(df)
        print("Data preprocessed.")

        # Calculate product and category features
        products = product_features(df)
        categories = category_features(df)
        print("Product and category features calculated.")

        # Calculate relative prices
        relative_prices = calculate_relative_price(products)
        df = df.join(relative_prices, on='product_id')
        products = products.join(relative_prices, on='product_id')
        print("Relative prices calculated.")

        # Calculate average interaction rates for smoothing
        # Need to calculate events counts first
        events = df.groupBy('event_type').count()
        events_counts = events.collect()
        event_counts_dict = {row['event_type']: row['count'] for row in events_counts}

        avg_purchase_per_view = event_counts_dict.get('purchase', 0) / event_counts_dict.get('view', 1) if event_counts_dict.get('view', 1) > 0 else 0
        avg_cart_per_view = event_counts_dict.get('cart', 0) / event_counts_dict.get('view', 1) if event_counts_dict.get('view', 1) > 0 else 0
        avg_purchase_per_cart = event_counts_dict.get('purchase', 0) / event_counts_dict.get('cart', 1) if event_counts_dict.get('cart', 1) > 0 else 0

        # Smooth category features
        categories = category_smoothener(categories, avg_purchase_per_view, 'views', 'purchase_per_view', 2000)
        categories = category_smoothener(categories, avg_cart_per_view, 'views', 'cart_per_view', 2000)
        categories = category_smoothener(categories, avg_purchase_per_cart, 'carts', 'purchase_per_cart', 200)
        print("Category features smoothed.")


        # Smooth product features
        products = product_smoothener(products, categories, 'views', 'purchase_per_view', 1000)
        products = product_smoothener(products, categories, 'views', 'cart_per_view', 1000)
        products = product_smoothener(products, categories, 'carts', 'purchase_per_cart', 100)
        print("Product features smoothed.")

        # Calculate user features
        users = user_features(df)
        print("User features calculated.")

        # Calculate interaction matrix
        interaction_matrix = calculate_interaction_matrix(df)
        print("Interaction matrix calculated.")


        return products, users, interaction_matrix

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None, None, None

# Load dataframes
# **IMPORTANT:** Adapt the file_path to your CSV location
spark = get_spark_session() # Initialize spark here
file_path = 'data/2019-Oct.csv'
products_df, users_df, interaction_matrix_df = load_and_process_data(spark, file_path)