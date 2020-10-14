import asyncio

from SceneDetection import scene_detec_model

async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    await asyncio.sleep(2)
    print('aaa')


def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given this input
    """
    scene_detect_result = scene_detec_model.get_scene_attributes(image_file)
    return {
        "Category Confidence": scene_detect_result['category_results'],
        "Scene Attributes": scene_detect_result['attributes_result'],
    }