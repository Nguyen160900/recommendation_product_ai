from pyspark.sql import SparkSession
from services.utils import preprocess, product_features, user_features, calculate_interaction_matrix, simple_als

def load_and_process_data(file_path):
    spark = SparkSession.builder.appName("RecommendationAPI").getOrCreate()
    df = spark.read.option("header", True).csv(file_path, inferSchema=True)
    df = preprocess(df)

    products = product_features(df)
    users = user_features(df)
    interactions = calculate_interaction_matrix(df)
    als_model = simple_als(interactions)

    return {
        "spark": spark,
        "products": products,
        "users": users,
        "interactions": interactions,
        "als_model": als_model
    }
