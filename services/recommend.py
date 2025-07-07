from pyspark.sql.functions import col

def get_user_recommendations(user_id, global_data, num_items=10):
    model = global_data["als_model"]
    user_mapping = global_data["user_mapping"]
    product_mapping = global_data["product_mapping"]
    products_df = global_data["products"]

    if user_id not in user_mapping:
        return {"user_id": user_id, "recommendations": []}

    user_index = user_mapping[user_id]
    recommendations = model.recommend(user_index, global_data["interactions"], N=num_items)

    reverse_product_mapping = {v: k for k, v in product_mapping.items()}
    recs = []
    for prod_index, score in recommendations:
        prod_id = reverse_product_mapping[prod_index]
        product_info = products_df[products_df["product_id"] == prod_id].to_dict(orient="records")
        if product_info:
            product_info[0]["rating"] = score
            recs.append(product_info[0])

    return {
        "user_id": user_id,
        "recommendations": recs
    }

def get_similar_products(product_id, global_data, num_items=10):
    products = global_data["products"]
    target_product = products[products["product_id"] == product_id]

    if target_product.empty:
        return {"product_id": product_id, "recommendations": []}

    target_category = target_product.iloc[0]["category"]

    related = products[
        (products["category"] == target_category) &
        (products["product_id"] != product_id)
        ].sort_values("purchase_per_view", ascending=False)

    recs = related.head(num_items).to_dict(orient="records")

    return {
        "product_id": product_id,
        "recommendations": recs
    }

def get_hybrid_recommendations(user_id, product_id, global_data, num_items=10):
    user_recs = get_user_recommendations(user_id, global_data, num_items * 2)["recommendations"]
    product_recs = get_similar_products(product_id, global_data, num_items * 2)["recommendations"]

    product_scores = {}

    for rec in user_recs:
        pid = rec["product_id"]
        product_scores[pid] = product_scores.get(pid, 0) + rec.get("rating", 0)

    for rec in product_recs:
        pid = rec["product_id"]
        product_scores[pid] = product_scores.get(pid, 0) + 0.5  # trọng số nhẹ hơn

    sorted_items = sorted(product_scores.items(), key=lambda x: -x[1])[:num_items]
    product_ids = [pid for pid, _ in sorted_items]

    products = global_data["products"]
    final = products[products["product_id"].isin(product_ids)]

    return {
        "user_id": user_id,
        "product_id": product_id,
        "recommendations": final.to_dict(orient="records")
    }