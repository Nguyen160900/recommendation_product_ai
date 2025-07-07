import pandas as pd
from services.utils import preprocess, product_features, user_features, calculate_interaction_matrix, simple_als

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess(df)

    products = product_features(df)
    users = user_features(df)
    interactions = calculate_interaction_matrix(df)
    als_model = simple_als(interactions)

    return {
        "products": products,
        "users": users,
        "interactions": interactions,
        "als_model": als_model
    }
