from pyspark.sql.functions import col

def get_user_recommendations(user_id, global_data, num_items=10):
    spark = global_data["spark"]
    model = global_data["als_model"]
    products = global_data["products"]
    users = global_data["users"]

    user_df = spark.createDataFrame([(user_id,)], ["user_id"])
    recs = model.recommendForUserSubset(user_df, num_items)

    if recs.count() == 0:
        return {"user_id": user_id, "recommendations": []}

    rec_items = recs.selectExpr("explode(recommendations) as rec") \
        .select(col("rec.product_id"), col("rec.rating"))

    final = rec_items.join(products, on="product_id", how="left")
    return {
        "user_id": user_id,
        "recommendations": final.limit(num_items).toPandas().to_dict(orient="records")
    }

def get_similar_products(product_id, global_data, num_items=10):
    products = global_data["products"]
    target_product = products.filter(col("product_id") == product_id).limit(1)

    if target_product.count() == 0:
        return {"product_id": product_id, "recommendations": []}

    target = target_product.collect()[0]
    target_category = target['category']

    related = products.filter(col("category") == target_category) \
                     .filter(col("product_id") != product_id) \
                     .orderBy(col("purchase_per_view").desc())

    return {
        "product_id": product_id,
        "recommendations": related.limit(num_items).toPandas().to_dict(orient="records")
    }

def get_hybrid_recommendations(user_id, product_id, global_data, num_items=10):
    user_recs = get_user_recommendations(user_id, global_data, num_items * 2)["recommendations"]
    product_recs = get_similar_products(product_id, global_data, num_items * 2)["recommendations"]

    # Dùng product_id làm key để merge gợi ý
    product_scores = {}

    for rec in user_recs:
        pid = rec["product_id"]
        product_scores[pid] = product_scores.get(pid, 0) + rec.get("rating", 0)

    for rec in product_recs:
        pid = rec["product_id"]
        product_scores[pid] = product_scores.get(pid, 0) + 0.5  # trọng số đơn giản

    sorted_items = sorted(product_scores.items(), key=lambda x: -x[1])[:num_items]
    product_ids = [pid for pid, _ in sorted_items]

    products = global_data["products"]
    result = products.filter(col("product_id").isin(product_ids))

    return {
        "user_id": user_id,
        "product_id": product_id,
        "recommendations": result.toPandas().to_dict(orient="records")
    }