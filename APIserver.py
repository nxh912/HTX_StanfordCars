#from typing import Union
#from fastapi import FastAPI, Form, Path, File, UploadFile, HTTPException
import pandas as pd
import tensorflow as tf
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

app = FastAPI()

# Directory to save uploaded files
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# trained model
weights = ResNet50_Weights.DEFAULT
model = resnet50( weights=weights)

print(weights)
print(model)

def load_model():
    # Set up Model / Resnet50_predict.ipynb
    from torchvision.models import resnet50, ResNet50_Weights

    model.eval()


# Make predictions on a sample image
def img_classify(image_path, img_height=128, img_width=128):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # image_path = "images.jpg"
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    # Get the class name corresponding to the predicted index
    class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']
    return predicted.item()


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Validate the uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Save the file to the uploads directory
    print(f"# POST: file.filename:{file.filename}")

    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save file")

    # predict car model
    filename = f"{ UPLOAD_DIR }/{ file.filename }"
    print(f"# pred_model = img_classify( '{filename}' )")

    model_id = int( img_classify( filename))
    #print(f"LINE L88 : pred_model : {model_id}")

    #print(f"LINE89 model_id : {model_id}\n\n" )

    #print(f"L91: model_id \n  {df_train[ df_train['train_num'] == model_id ] }")

    pred_model = ""
    result_table = df_train[ df_train['train_num'] == model_id ]
    for date, row in result_table.T.iteritems():
        pred_model = row['Cars']
        break

    return JSONResponse(
        content={"filename": file.filename,
                 "predicted_model": pred_model}
    )

def load_models():
    df_train_labels = pd.read_csv('training_labels.csv',
                                  dtype={'testing_num': 'Int32', 'Cars': str, 'Cars': str, 'Year': 'Int32'})

    df_test_labels = pd.read_csv('testing_labels.csv',
                                  dtype={'testing_num': int,
                                         'Cars':        str,
                                         'Make':        str,
                                         'Year':        int})

    print( df_train_labels.tail() )
    print( df_test_labels.tail() )
    return (df_train_labels, df_test_labels)

df_train, df_test = load_models()

