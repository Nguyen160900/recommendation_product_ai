import math

from fastapi import APIRouter, HTTPException
from pyspark.sql.functions import col, abs

# Import dependencies directly
from . import dependencies
from .models import UserRecommendationsResponse, ProductRecommendationsResponse

router = APIRouter()

# Access loaded dataframes and model directly from dependencies
spark = dependencies.get_spark_session()
als_model = dependencies.get_als_model(spark)
products_df = dependencies.products_df  # Access the variable directly
users_df = dependencies.users_df  # Access the variable directly
interaction_matrix_df = dependencies.interaction_matrix_df  # Access the variable directly


@router.get("/user/{user_id}", response_model=UserRecommendationsResponse)
async def recommend_products_for_user(
        user_id: int,
        num_recommendations: int = 10,
        filter_interacted: bool = True
):
    if als_model is None or products_df is None or users_df is None or interaction_matrix_df is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded")

    user_df_subset = spark.createDataFrame([(user_id,)], ['user_id'])

    # Ensure the user exists
    if users_df.filter(col('user_id') == user_id).count() == 0:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

    recommendations = als_model.recommendForUserSubset(user_df_subset,
                                                       num_recommendations * 5)  # Generate more to filter later

    if recommendations.isEmpty():
        return {"user_id": user_id, "recommendations": []}

    recs_list = recommendations.collect()[0][1]
    recs_df = spark.createDataFrame(recs_list)

    # Calculate recommendation scores
    recs_df = recs_df.join(products_df.select('product_id', 'purchase_per_view', 'relative_price'), on='product_id')

    user_avg_relative_price_row = users_df.filter(col('user_id') == user_id).select(
        'avg_relative_price_purchased').collect()
    if not user_avg_relative_price_row or user_avg_relative_price_row[0][0] is None:
        user_avg_relative_price = 0  # Or a default value
    else:
        user_avg_relative_price = user_avg_relative_price_row[0][0]

    coef_als_score = 0.8
    coef_conversion_rate = 0.1
    coef_spending_habit = 0.1
    coef_spending_booster = 0.05

    # Scale features
    recs_df = recs_df.withColumn('rating_scaled', col('rating') / 2)
    recs_df = recs_df.withColumn('purchase_per_view_scaled', col('purchase_per_view') / 0.075)
    recs_df = recs_df.withColumn('relative_price_scaled', (col('relative_price') + 5) / 10)
    user_avg_relative_price_scaled = (user_avg_relative_price + 1) / 2

    # Calculate the recommendation scores
    recs_df = recs_df.withColumn('recommendation_score',
                                 ((col('rating_scaled') * coef_als_score) +
                                  (col('purchase_per_view_scaled') * coef_conversion_rate) -
                                  abs(user_avg_relative_price_scaled + coef_spending_booster - col(
                                      'relative_price_scaled')) * coef_spending_habit) /
                                 (coef_als_score + coef_conversion_rate + coef_spending_habit))

    if filter_interacted:
        # Filter non-interacted products
        user_interactions = interaction_matrix_df.filter(col('user_id') == user_id).select('product_id')
        recs_df = recs_df.join(user_interactions, on='product_id', how='leftanti')

    # Sort and get top N
    final_recommendations = recs_df.sort('recommendation_score', ascending=False).limit(num_recommendations)

    # Select relevant columns and convert to Pandas for response model
    result_df = final_recommendations.select('product_id', 'rating', 'recommendation_score', 'purchase_per_view',
                                             'relative_price')

    return {"user_id": user_id, "recommendations": result_df.toPandas().to_dict(orient='records')}


@router.get("/product/{product_id}", response_model=ProductRecommendationsResponse)
async def recommend_similar_products(
        product_id: int,
        num_recommendations: int = 10
):
    if als_model is None or products_df is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded")

    # Get the product factors from the ALS model
    try:
        product_vectors = als_model.itemFactors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing item factors: {e}")

    if product_vectors is None:
        raise HTTPException(status_code=500, detail="Product item factors not available")

    # Ensure the product exists in the products_df and has factors
    if products_df.filter(col('product_id') == product_id).count() == 0:
        raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")

    # Check if the product has item factors
    if product_vectors.filter(col('id') == product_id).count() == 0:
        raise HTTPException(status_code=404, detail=f"Item factors not found for product ID {product_id}")

    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler, Normalizer, BucketedRandomProjectionLSH

    # Prepare product vectors for LSH
    product_vectors_lsh = product_vectors.rdd.map(lambda row: (row[0], Vectors.dense(row[1]))).toDF(
        ['product_id', 'features'])

    assembler = VectorAssembler(inputCols=['features'], outputCol='vector')
    product_vectors_lsh = assembler.transform(product_vectors_lsh)

    normalizer = Normalizer(inputCol='vector', outputCol='norm_vector')
    product_vectors_lsh = normalizer.transform(product_vectors_lsh)

    # Fit LSH model (ideally, fit this once and save/load the LSH model as well)
    brp = BucketedRandomProjectionLSH(inputCol="norm_vector", outputCol="neighbors", numHashTables=5, bucketLength=0.1)
    try:
        brp_model = brp.fit(product_vectors_lsh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fitting LSH model: {e}")

    # Find the nearest neighbors of the specific product
    query_vector_row = product_vectors_lsh.filter(col('product_id') == product_id).select('norm_vector').first()
    if query_vector_row is None:
        raise HTTPException(status_code=404, detail=f"Vector not found for product ID {product_id}")
    query = query_vector_row[0]

    neighbors = brp_model.approxNearestNeighbors(product_vectors_lsh, query,
                                                 numNearestNeighbors=num_recommendations + 1)  # +1 to exclude the product itself

    # Calculate recommendation scores for products
    recs_product_df = neighbors.select('product_id', 'distCol')

    recs_product_df = recs_product_df.join(products_df.select('product_id', 'purchase_per_view', 'relative_price'),
                                           on='product_id')

    coef_distance_score = 0.8
    coef_conversion_rate = 0.1
    coef_relative_price = 0.1
    coef_spending_booster = 0.05

    # Scale features
    recs_product_df = recs_product_df.withColumn('distCol_scaled', (math.sqrt(2) - col('distCol')) / math.sqrt(2))
    recs_product_df = recs_product_df.withColumn('purchase_per_view_scaled', col('purchase_per_view') / 0.075)
    recs_product_df = recs_product_df.withColumn('relative_price_scaled', (col('relative_price') + 5) / 10)

    product_relative_price_row = recs_product_df.filter(col('distCol') == 0).select(
        'relative_price_scaled').first()  # Distance 0 means it's the product itself
    if product_relative_price_row is None:
        product_relative_price_scaled = 0.5  # Default if relative price is not available
    else:
        product_relative_price_scaled = product_relative_price_row[0]

    # Calculate the recommendation scores
    recs_product_df = recs_product_df.withColumn('recommendation_score',
                                                 ((col('distCol_scaled') * coef_distance_score) +
                                                  (col('purchase_per_view_scaled') * coef_conversion_rate) -
                                                  abs(product_relative_price_scaled + coef_spending_booster - col(
                                                      'relative_price_scaled')) * coef_relative_price) /
                                                 (coef_distance_score + coef_conversion_rate + coef_relative_price))

    # Remove the searched product from the recommendations
    recs_product_df = recs_product_df.filter(col('distCol') != 0)

    # Sort and get top N
    final_recommendations = recs_product_df.sort('recommendation_score', ascending=False).limit(num_recommendations)

    # Select relevant columns and convert to Pandas for response model
    result_df = final_recommendations.select('product_id', col('distCol').alias('rating'), 'recommendation_score',
                                             'purchase_per_view',
                                             'relative_price')  # Using distCol as rating placeholder

    return {"product_id": product_id, "recommendations": result_df.toPandas().to_dict(orient='records')}
