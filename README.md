# CARBS Processing Backend
## Overview
The **CARBS Processing Backend** is the server-side component of the CARBS (Carbohydrate Analysis and Recognition By Scan) system. It processes images captured by the Android app to perform food detection, volume estimation using fiducial markers, and carbohydrate calculation, returning results to the client app.

## Features
- Image upload and food detection via a YOLOv2 model trained on UEC-Food100.
- Volume estimation using ArUco fiducial markers.
- Conversion of food volume to weight and carbohydrate estimation.
- Results bundling and response delivery to client app.
- Deployable locally or via cloud services like Microsoft Azure.

## Technologies
- Python 3.x
- Flask 
- OpenCV
- YOLOv2
- UEC-Food100 dataset
    
## API Endpoints
| Method | Endpoint           | Description                               |
|:------:|:------------------:|:----------------------------------------:|
| GET    | `/test`             | Test server connectivity                |
| POST   | `/upload-data`      | Upload an image for processing          |
| GET    | `/get-food-classes` | Retrieve list of supported food classes |

## Deployment
- Backend has been successfully deployed to Azure for remote access.
- HTTPS configuration and resource scaling may be needed for production use.
