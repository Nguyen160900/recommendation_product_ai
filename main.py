from fastapi import FastAPI
import os
from services.recommend import get_user_recommendations, get_similar_products, get_hybrid_recommendations
from contextlib import asynccontextmanager
from services.pipeline import load_and_process_data

# Global state
global_data = {
    "products": None,
    "users": None,
    "interactions": None,
    "als_model": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ chạy khi app khởi động
    default_file = "data/2019-Oct.csv"
    if os.path.exists(default_file):
        global global_data
        global_data.update(load_and_process_data(default_file))
        print(f"✅ Dữ liệu được load từ: {default_file}")
    else:
        print("⚠️ File mặc định không tồn tại. Cần upload sau.")

    yield

    # ✅ cleanup nếu cần khi shutdown
    print("🛑 App đang tắt... cleanup tài nguyên nếu cần.")


# Tạo app với lifecycle handler
app = FastAPI(lifespan=lifespan)

@app.get("/recommend/users/{user_id}")
async def recommend_user(user_id: int):
    if not global_data["als_model"]:
        return {"error": "No model found. Upload data first."}
    return get_user_recommendations(user_id, global_data)


@app.get("/recommend/products/{product_id}")
async def recommend_product(product_id: int):
    if global_data["products"] is None:
        return {"error": "No product data found."}
    return get_similar_products(product_id, global_data)


@app.get("/recommend/hybrid/{user_id}/{product_id}")
async def recommend_hybrid(user_id: int, product_id: int):
    if not global_data["als_model"]:
        return {"error": "No model found. Upload data first."}
    return get_hybrid_recommendations(user_id, product_id, global_data)
