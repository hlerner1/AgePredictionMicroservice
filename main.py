import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import logging
import asyncio
from model import predict, init
from pydantic import BaseSettings

import os.path
from os import path

class Settings(BaseSettings):
    ready_to_predict = False

settings = Settings()
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


@app.post("/status")
async def initial_startup():
    # Run startup task async
    init_task = asyncio.create_task(init())
    settings.ready_to_predict = True
    return {"result": str(settings.ready_to_predict)}


@app.get("/status")
async def check_status():
    return {"result": str(settings.ready_to_predict)}


@app.post("/predict")
async def create_prediction(filename: str = ""):
    if not settings.ready_to_predict:
        return HTTPException(status_code=503, detail="Model has not been configured. Please run initial startup before attempting to receive predictions.")

    image_file_path = 'images/'+filename
    try:
        image_file = open(image_file_path, 'r')
    except IOError:
        return HTTPException(status_code=400, detail="Unable to open image file. Provided filename can not be found on server.")

    result = predict(image_file)
    image_file.close()
    return {"result": result}
