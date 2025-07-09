from fastapi import FastAPI

from services.recommendations import router as recommendation_router

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Product Recommendation API"}


app.include_router(recommendation_router, prefix="/recommend")
