import asyncio

# Import model
from sodanet_model import SodaModel
from matplotlib.image import imread
from utilities import compute_accuracy_labelwise
from PIL import Image
import io

model = None

async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    global model
    await asyncio.sleep(2)

    # Loading the sodanet module
    print('Loading SodaNet model')
    model = SodaModel()


def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given this input
    """
    if model == None:
        raise RuntimeError("SodaNet model is not loaded properly")

    model.load_image_from_file(image_file.name)
    predicted_value, im_ret = model.evaluate()
    predicted_value_converted_to_yn = "Yes" if str(predicted_value) == 1 else "No"
    return {
        "Contains Coke (Can)": predicted_value_converted_to_yn,
    }