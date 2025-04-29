#flaskapp.py - Handles the main functionality of the Processing Server and CARBS Pipeline
import flask
from flask import request, send_file
import zipfile
import os, shutil
import cv2
import subprocess
import numpy as np
import re
from flask_cors import CORS

from food_image_processing import yolov2
app = flask.Flask(__name__)

#Enable CORS (For Azure primarily)
CORS(app)

#Establish directories
uploadDir = "upload"
detectionDir = "detections"
bundlingDir = "bundling"

#Main server version, currently v1.
serverMainVersion = "1.0"

#Known size of printed markers in CM
markerSizeCM = 5 

#Get the food detection model
yolov2_model = yolov2.YoloV2 

#Set up volume to mass, mass to carb dictionaries
with open("volume_estimation/vol_to_mass.txt", "r") as file:
    vol_to_mass_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}

with open("volume_estimation/mass_to_carbs.txt", "r") as file:
    mass_to_carbs_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}

food_class_file_path = "food_image_processing/food100.names"

#Handle connection test route
@app.route("/test", methods=['GET'])
def handle_call():
    print("Successfully connected to android device.", flush=True)
    return "C.A.R.B.S Processing Backend Successfully Connected"

#Flags if marker is not found
markerFound = True 

#Handle image upload and carbs pipeline route
@app.route("/upload-data", methods=['POST'])
def store_data():
    #Alert if image not found in request
    if 'image' not in request.files:
        print("Error: recieved data upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    
    image = request.files['image']

    imgFilePathToSave = f"./upload/{image.filename}"
    image.save(imgFilePathToSave)
    print("Recieved image from app capture, stored at: " + imgFilePathToSave, flush=True)

    #Clear previous detections
    clearDetectionDir()
    #Run CARBS pipeline to get detections and information
    mainImg, foodImages, foodData = run_carbs_algo(imgFilePathToSave)
    #Return a zip with all information required for app display
    return createResponseZip("Data processed successfully. MarkerFound=" + str(markerFound), mainImg, foodImages, foodData)

#Handle route to list all food classes
@app.route("/get-food-classes", methods=['GET'])
def get_food_classes():
    print("Received request for all food item classes", flush=True)
    with open(food_class_file_path, "r") as file:
        classes = sorted(file.read().splitlines())
    return  " ".join(classes)

def run_carbs_algo(imgFile):
    """Starts the CARBS pipeline through processImage and calculateCarbs, returning a labelled image, detection images and foodData"""
    processedImage = processImage(imgFile)
    labelled_img = processedImage[0]
    detections = [item[0] for item in processedImage[1]]
    boxes = [item[1] for item in processedImage[1]]
    confidences = processedImage[2]
    keys = tuple(detections)
    carbs_data, volume_data, weight_data = calculateCarbs(imgFile, detections, boxes)
    foodData = dict(zip(keys, zip(carbs_data, confidences, volume_data, weight_data)))
    detection_images = getDetectionCrops(detections, boxes, imgFile)
    
    return labelled_img, detection_images, foodData

def calculateCarbs(imgFile, detections, boxes):
    """Calculates carbs for each food item detected
    First calculates volume, then mass, then carbohydrates
    Returns a list of all volumes, weights and carbohydrate counts
    """
    carb_list = []
    volume_list = []
    weight_list = []

    for d in range(0, len(detections)):
        detection = detections[d]
        print("\nCalculating carbs for: " + str(detection))

        volume = calculate_volume(imgFile, boxes[d], markerSizeCM)
        weight = get_mass_from_vol(volume, detection)
        carbs = get_carbs_from_weight(weight, detection)

        volume_list.append(round(volume, 2))
        weight_list.append(round(weight, 2))
        carb_list.append(round(carbs, 2))

    # Convert np.float64 to float
    volume_list = [float(val) for val in volume_list]
    weight_list = [float(val) for val in weight_list]
    carb_list = [float(val) for val in carb_list]
    return carb_list, volume_list, weight_list

def get_mass_from_vol(volume, detection):
    """Converts volume (cm^3) into mass (g) by consulting volume to mass ratios"""
    detection_class = re.sub(r'_\d+$', '', detection)
    mass = vol_to_mass_dict[detection_class] * volume
    print("Estimated weight: " + str(mass) + "g")
    return mass

def get_carbs_from_weight(weight, detection):
    """Converts weight (g) into carbohydrates (g) by consulting mass to carb ratios"""
    detectionclass = re.sub(r'_\d+$', '', detection)
    carbs = mass_to_carbs_dict[detectionclass] * weight
    print("Estimated Carbohydrates: " + str(carbs) + "g")
    return carbs

def detect_aruco_marker(image, marker_length):
    """Detects the ArUco fiducial marker, returning the scaling ratio used to convert between device pixels and real world units"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None and len(corners) > 0:
        marker_width_pixels = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
        print("Marker width in pixels: " + str(marker_width_pixels), flush=True)
        pixel_to_real = marker_length / marker_width_pixels
        print("Pixel to real: " + str(pixel_to_real), flush=True)
        return pixel_to_real
    return None


def calculate_volume(foodPath, box, markerLength):
    """Estimates the volume of a bounding box using the ArUco fiducial marker"""
    foodImg = cv2.imread(foodPath)
    pixel_to_real = detect_aruco_marker(foodImg, markerLength)
    global markerFound
    if pixel_to_real is None:
        pixel_to_real = 0.01
        markerFound = False  
        print("Warning: No fiducial marker detected. Using default scaling.", flush=True)
    else:
        markerFound = True
    
    x_min, y_min, width, height = box
    #Calculate real dimensions of food using pixel_to_real ratio
    real_width = width * pixel_to_real
    real_height = height * pixel_to_real

    #Estimate the third dimension of the bounding box to make it 3D
    real_depth = min(real_width, real_height) * 0.5
    return real_width * real_height * real_depth
    
def getDetectionCrops(detections, boxes, imgFile):
    """Crop the main image around the bounding box of a detection. 
    Returns a list of all detection crops as individual images created, in the detections directory"""
    mainImg = cv2.imread(imgFile) 
    detection_crops = []
    for d in range(0, len(detections)):
        box = boxes[d]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cropped_img = mainImg[y:y+h, x:x+w]
        name = "detection_" + detections[d] + ".png"
        path = os.path.join(detectionDir, name)
        try:
            cv2.imwrite(path, cropped_img)
            detection_crops.append(path)

        except Exception as e: #Give placeholder if error creating crop
            detection_crops.append("assets/no_crop_found.png")

    return detection_crops


def createResponseZip(message, mainImg, foodImages, foodData):
    """Create the response ZIP bundle file to send back to android app"""
    zip_filename = "files_bundle.zip"

    #Create zip file and add images
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(mainImg, os.path.basename(mainImg))

        for img_path in foodImages:
            zipf.write(img_path, os.path.basename(img_path))
        foodDataFilePath = os.path.join(bundlingDir, "foodData.txt")
        with open(foodDataFilePath, "w") as info_file:
            info_file.write(f"Message: {message}\n")
            idx = 0
            for key, value in foodData.items():
                # info_file.write(f"Food data: \n{foodData}")
                info_file.write(key + ":" + str(value)+"\n")

        zipf.write(foodDataFilePath, "foodData.txt")

    return send_file(zip_filename, as_attachment=True)

def clearUploadDir():
    """Clear upload directory"""
    # print("Attempting to clear upload directory...", flush=True)
    for filename in os.listdir(uploadDir):
        file_path = os.path.join(uploadDir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e) + "\n")
    if len(os.listdir(uploadDir)) == 0:
        print("Successfully cleared upload directory!", flush=True)

def clearDetectionDir():
    """Clear detection directory"""
    # print("Attempting to clear detections directory...", flush=True)
    for filename in os.listdir(detectionDir):
        file_path = os.path.join(detectionDir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e) + "\n")
    if len(os.listdir(detectionDir)) == 0:
        print("Successfully cleared detections directory!", flush=True)


def clearBundlingDirAndZip():
    """Clear bundling directory and zip bundle"""
    for filename in os.listdir(bundlingDir):
        file_path = os.path.join(bundlingDir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e) + "\n")
    if len(os.listdir(bundlingDir)) == 0:
        print("Successfully cleared bundling directory!", flush=True)

    #Delete ZIP bundle
    if os.path.exists("files_bundle.zip"):
        os.remove("files_bundle.zip")
        if os.path.exists("files_bundle.zip"):
            print("Unable to delete ZIP bundle")
        else:
            print("Successfully deleted ZIP bundle")
    

def processImage(imgFilePath):
    """run YoloV2 food detection on an image, return with a labelled image, detections, and confidences"""
    analysis = yolov2_model.analyse_image(imgFilePath)
    labelled_img = analysis[0]
    detections = analysis[1]
    if len(detections) > 0:
        confidences = analysis[2]
    else:
        confidences = [0.0]
    
    return labelled_img, detections, confidences

def get_git_commit_count():
    """Fetch the number of git commits to keep track of versioning"""
    try:
        # Run git command to get the number of commits
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return "." + str(int(result.stdout.strip()))
    except Exception:
        return None  # In case git fails


def get_version_number():
    """Gets version number if git information is present, updating local file. 
    If not present (deployment), version number is sourced from local file"""
    commits = get_git_commit_count()
    if commits is not None:
        versionNumber = serverMainVersion + commits
        details = open("server_details.txt", "w")
        details.writelines(["Version: " + versionNumber])
        print("Attached to git, fetching version number.")
        return versionNumber
    else:
        print("Not attached to git, using local versioning.")
        details = open("server_details.txt", "r")
        versionNumber = details.readline().split(":")[1].strip(" ")
        return versionNumber


if __name__ == '__main__':
    #Main code run on startup
    version = get_version_number()
    clearUploadDir()
    clearDetectionDir()
    clearBundlingDirAndZip()
    print("-" * 80)
    print("\nC.A.R.B.S Processing backend v" + str(version) + "\n")
    app.run(host="0.0.0.0", port=8000)#, debug=True)
