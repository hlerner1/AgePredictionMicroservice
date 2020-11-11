import asyncio
from PIL import Image
from extract_faces import extract_faces
from classify_faces import classify_faces
import os
import shutil

async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    await asyncio.sleep(2)

def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a file-like object
    with the image as an input.
    """
    os.mkdir('test_images')
    shutil.copy(image_file.name, 'test_images')
    image = Image.open('test_images/'+image_file.name[6:], mode='r')

    test_img_folder = "test_images"
    # print('extracting...')
    # dsfd_weights_path = "face_detector/dsfd_inference/dsfd/weights/WIDERFace_DSFD_RES152.pth"
    # extract_faces(dsfd_weights_path=dsfd_weights_path, input_images_path=test_img_folder, extract_faces=True, save_bounding_boxes=True)
    # print('extracted')
    model_path = 'three_class_trained_age_recognition_model.pth'
    unclassified_faces_path = 'test_images/'
    print('classifying...')
    result = classify_faces(model_path=model_path, unclassified_faces_path=unclassified_faces_path)
    shutil.rmtree('test_images')
    shutil.rmtree('classified_faces')
    print('finished classifying')
    return {
        "age_predict": str(result),
    }
