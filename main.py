from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import logging
import asyncio
from model import predict, init
from pydantic import BaseSettings


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
    """
    Default endpoint for testing if the server is running
    :return: Positive JSON Message
    """
    return {"MLMicroserviceTemplate is Running!"}


@app.post("/status")
async def initial_startup():
    """
    Calls the init() method in the model and prepares the model to receive predictions. The init
    task may take a long time to complete, so the settings field ready_to_predict will be updated
    asynchronously when init() completes.

    :return: {"result": "starting"}
    """
    # Run startup task async
    init_task = asyncio.create_task(init())
    settings.ready_to_predict = True
    return {"result": "starting"}


@app.get("/status")
async def check_status():
    """
    Checks the current prediction status of the model. Predictions are not able to be made
    until this method returns {"result": "True"}.

    :return: {"result": "True"} if model is ready for predictions, else {"result": "False"}
    """

    return {"result": str(settings.ready_to_predict)}


@app.post("/predict")
async def create_prediction(filename: str = ""):
    """
    Creates a new prediction using the model. This method must be called after the init() method has run
    at least once, otherwise this will fail with a HTTP Error. When given a filename, the server will create a
    file-like object of the image file and pass that to the predict() method.

    :param filename: Image file name for an image stored in shared Docker volume photoanalysisserver_images
    :return: JSON with field "result" containing the results of the model prediction.
    """

    # Ensure model is ready to receive prediction requests
    if not settings.ready_to_predict:
        return HTTPException(status_code=503,
                             detail="Model has not been configured. Please run initial startup before attempting to "
                                    "receive predictions.")

    # Attempt to open image file
    try:
        image_file = open('images/' + filename, 'r')
    except IOError:
        return HTTPException(status_code=400,
                             detail="Unable to open image file. Provided filename can not be found on server.")

    # Create prediction with model
    result = predict(image_file)
    image_file.close()
    return {"result": result}
