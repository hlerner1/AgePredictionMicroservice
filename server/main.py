import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
from model import predict

app = FastAPI()

# Must have CORSMiddleware to enable localhost client and server
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5057",
    "http://localhost:5000",
    "http://localhost:6379",
]

logger = logging.getLogger("api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"running!"}


@app.get("/predict")
async def get_predict_results():
    # TODO: Check if batch with unique key is complete
    # TODO: Return result if batch is done else return buffering code
    return {"result": "complete"}


@app.post("/predict")
async def create_prediction():
    unique_key = str(uuid.uuid4())
    imageFile = None

    # TODO: Ensure this is done in the background
    result = predict(imageFile)
    return {"key": unique_key}
