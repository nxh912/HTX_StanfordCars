# Car-Classification

1. Language / Environment: ```Python 3.9 (Anaconda)```
1. Library : ```Tensorflow-Keras```
1. Pre-trained classifier : ```ResNet50```
1. Jupyter notebook for training: ``` Resnet50_train.ipynb ```
1. ``` Resnet50_train.ipynb ``` has differernt headers for ```data input```, ``` model building```
1. Final trained model is written to file: ``` Rasnet50.pth ```
1. Model can be used for prediction using an FastAPI webservice (```APIserver.py```)
   
**Goal**: Building a vehicle recognition predictive model using machine learning models (traditional and deep learning), and the goal of that model is to classify a car’s make and model based on an input image.

**Raw data source**: https://www.kaggle.com/jessicali9530/stanford-cars-dataset

**Author: Ngai_Lam Ho**

** Data setup **
Copy the training and testing data into the two directories:
``` cars_train ``` and ```  cars_test ```

![](14LB.gif)


** Training of Model (Resnet50):
1. (in a seprate terminal)
1. ```ipython3 Resnet50_train.ipynb ``` 

** Running of API-service (Resnet50):
1. (in a seprate terminal)
1. Install FastAPI Library:
  ``` pip3 install fastapi
      pip3 install python-multipart
  ```
1. Run the API Car model classification server: ``` fastapi dev  APIserver.py  ```
1. Test:
  ```
   curl -X POST "http://127.0.0.1:8000/upload/" -F "file=@./cars_test/cars_test/00386.jpg"
  ```

** Testing with Swagger.API
1. Go to URL: https://editor.swagger.io/
1. Import the API definition file from ```swagger.yaml```
1. Import image files for testing
