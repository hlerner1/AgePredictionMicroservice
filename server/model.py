import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate 

from pathlib import Path
import shutil
import pandas as pd

from model import get_model
from classify_faces_dataset import CustomFaceAgeClassificationDataset
import time

def file_path_collate(batch):
    # Custom collate function for dataloader that allows
    # for the dataloader to process a batch of Path items
    new_batch = []
    file_paths = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        file_paths.append(_batch[-1])
    return default_collate(new_batch), file_paths


def classify_faces(model_path=None, unclassified_faces_path=None, batch_size=32):
    # Set testing device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # images_root = Path(unclassified_faces_path)
    # test_dataset = CustomFaceAgeClassificationDataset(imgs_path=images_root, transform=transform)

    # # Set up dataloaders
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            # shuffle=False, num_workers=0, collate_fn=file_path_collate)

    # Initialize model
    model = get_model()
    model.load_state_dict(torch.load(Path(model_path)))
    model = model.to(device)
    model.eval()

    # Process images for predictions, then move them to predicted label folders
    # Label translation 
    label_translation = {
        0: "<= 12",
        1: "13 to 17",
        2: ">= 18"
    }

    # Set up output sorted folders
    # sorted_faces_folder = images_root.parent.joinpath('classified_faces')
    # faces_12_or_less = sorted_faces_folder.joinpath('12_or_less')
    # faces_13_to_17 = sorted_faces_folder.joinpath('13_to_17')
    # faces_18_and_above = sorted_faces_folder.joinpath('18_and_above')

    # Path(faces_12_or_less).mkdir(parents=True, exist_ok=True)
    # Path(faces_13_to_17).mkdir(parents=True, exist_ok=True)
    # Path(faces_18_and_above).mkdir(parents=True, exist_ok=True)
    # Get input data and corresponding filepaths
    inputs = inputs[0]
    inputs = inputs.to(device)

    # Forward 
    outputs = model(inputs)
    predicted = torch.argmax(outputs, dim=1)

    return predicted

async def init():
    await asyncio.sleep(2)
    
    print('something')




def predict(image_file):
    print('image file:', image_file.name)
    model_path = 'three_class_trained_age_recognition_model.pth'
    unclassified_faces_path = image_file.name
    # Classify faces
    start = time.time()
    classify_faces(model_path=model_path, unclassified_faces_path=unclassified_faces_path)
    end = time.time()
    print(end - start)

    print('Finished!')
    return { "predicted age": ret_val,}





