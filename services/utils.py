from pyspark.sql.functions import to_timestamp, col, split, when, count, first, mean, min, max
from pyspark.ml.recommendation import ALS

def preprocess(df):
    df = df.withColumn('event_time', to_timestamp('event_time'))
    df = df.withColumn('user_id', col('user_id').cast('integer'))
    df = df.withColumn('product_id', col('product_id').cast('integer'))
    df = df.withColumn('category_id', col('category_id').cast('long'))

    cart_df = df.filter(col('event_type') == 'cart')
    df = df.filter(col('event_type') != 'cart')
    cart_df = cart_df.dropDuplicates(['product_id', 'user_id', 'user_session'])
    df = df.union(cart_df)

    df = df.withColumn('category', split(df['category_code'], '\\.').getItem(0)) \
           .withColumn('sub_category', split(df['category_code'], '\\.').getItem(1)) \
           .withColumn('sub_sub_category', split(df['category_code'], '\\.').getItem(2))
    return df

def product_features(df):
    df = df.groupBy('product_id').agg(
        first('category_id').alias('category_id'),
        first('category_code').alias('category_code'),
        count(when(col('event_type') == 'view', True)).alias('views'),
        count(when(col('event_type') == 'cart', True)).alias('carts'),
        count(when(col('event_type') == 'purchase', True)).alias('purchases'),
        mean('price').alias('price'),
        min('event_time').alias('first_date'),
        max('event_time').alias('last_date')
    )

    df = df.withColumn('purchase_per_view', col('purchases') / col('views'))
    df = df.withColumn('cart_per_view', col('carts') / col('views'))
    df = df.withColumn('purchase_per_cart', when(col('carts') == 0, col('purchases')).otherwise(col('purchases') / col('carts')))
    return df

def user_features(df):
    return df.groupBy('user_id').agg(
        count(when(col('event_type') == 'view', True)).alias('views'),
        count(when(col('event_type') == 'cart', True)).alias('carts'),
        count(when(col('event_type') == 'purchase', True)).alias('purchases')
    )

def calculate_interaction_matrix(df):
    return df.filter(col('event_type') == 'purchase') \
             .groupBy('user_id', 'product_id') \
             .count() \
             .withColumnRenamed('count', 'rating')

def simple_als(interactions):
    als = ALS(
        maxIter=10,
        regParam=0.01,
        userCol="user_id",
        itemCol="product_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=False
    )
    model = als.fit(interactions)
    return model
