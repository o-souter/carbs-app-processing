import flask
from flask import request, Flask, send_file, jsonify
import zipfile
import os, shutil
import cv2
import subprocess
import numpy as np
import scipy
import pandas as pd

# from food_image_processing import food_detection
from food_image_processing import yolov2
app = flask.Flask(__name__)
uploadDir = "upload"

markerSizeCM = 6

yolov2_model = yolov2.YoloV2

#Set up volume to mass, mass to carb dictionaries

with open("volume_estimation\\vol_to_mass.txt", "r") as file:
    vol_to_mass_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}

with open("volume_estimation\\mass_to_carbs.txt", "r") as file:
    mass_to_carbs_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}


print("Volume to Mass dictionary: \n" + str(vol_to_mass_dict), flush=True)

print("Mass to Carbs Dictionary: \n" + str(mass_to_carbs_dict), flush=True)



@app.route("/test", methods=['GET'])
def handle_call():
    print("Successfully connected to android device.", flush=True)
    return "C.A.R.B.S Processing Backend v" + version + " Successfully Connected"


@app.route("/upload-data", methods=['POST'])
def store_data():
    if 'image' not in request.files:
        print("Error: recieved data upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    if 'pointcloud' not in request.files:
        print("Error: recieved data upload request but request contained no pointcloud file", flush=True)
        return {"error": "No pointcloud file found"}
    
    image = request.files['image']
    if len(os.listdir(uploadDir)) >= 2:
        clearUploadDir()

    imgFilePathToSave = f"./upload/{image.filename}"
    if os.path.isfile(imgFilePathToSave):
        newFileName = image.filename.split(".")[0] + "_1"
        print("An existing image already exists, renaming image file")
        imgFilePathToSave = f"./upload/{newFileName}" + ".jpg"
    # image.save(f"./upload/{image.filename}")
    image.save(imgFilePathToSave)
    print("Recieved image from app capture, stored at: " + imgFilePathToSave, flush=True)
    
    pointcloud = request.files['pointcloud']
    pointFilePathToSave = f"./upload/{pointcloud.filename}"
    if os.path.isfile(pointFilePathToSave):
        newFileName = pointcloud.filename.split(".")[0] + "_1"
        print("An existing pointcloud already exists, renaming pointcloud file")
        pointFilePathToSave = f"./upload/{newFileName}" + ".jpg"
    
    pointcloud.save(pointFilePathToSave)
    print("Recieved pointcloud from app capture, stored at: " + pointFilePathToSave, flush=True)
    mainImg, foodImages, foodData = run_carbs_algo(imgFilePathToSave, pointFilePathToSave)
    return createResponseZip("Image and pointcloud recieved and stored successfully", mainImg, foodImages, foodData)


#{"message": "Image and pointcloud recieved and stored successfully", "filenames":[image.filename, pointcloud.filename]}



def run_carbs_algo(imgFile, pointCloudFile):
    # foodData = "Not implemented yet!"
    processedImage = processImage(imgFile)
    labelled_img = processedImage[0]
    detections = [item[0] for item in processedImage[1]]# processedImage[1]
    boxes = [item[1] for item in processedImage[1]]
    # carbs = np.full(shape=len(detections), fill_value=50, dtype=np.double) #np.zeros_like(detections, float)
    keys = tuple(detections)
    carbs_data = calculate_carbs(imgFile, detections, boxes)
    foodData = dict(zip(keys, carbs_data))
    detection_images = get_detection_crops(detections, boxes, imgFile)
    
    return labelled_img, detection_images, foodData

def calculate_carbs(imgFile, detections, boxes):
    carb_list = []
    for d in range(0, len(detections)):
        volume = calculate_volume(imgFile, boxes[d], markerSizeCM)
        carbs = get_carbs_from_volume(volume, detections[d])
        carb_list.append(carbs)
    return carb_list



def detect_aruco_marker(image, marker_length):
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

    foodImg = cv2.imread(foodPath)
    pixel_to_real = detect_aruco_marker(foodImg, markerLength)
    if pixel_to_real is None:
        raise ValueError("ArUco marker not detected.")
    
    x_min, y_min, width, height = box
    real_width = width * pixel_to_real
    real_height = height * pixel_to_real
    print("Real width: " + str(real_width), flush=True)
    print("Real height: " + str(real_height), flush=True)
    real_depth = min(real_width, real_height) * 0.5  # Assumption: food is roughly cuboid
    print("Estimated depth: " + str(real_depth), flush=True)
    # Calculate volume (assuming a cuboid shape)
    volume = real_width * real_height * real_depth

    return volume



    # pointcloud = "./upload/point_cloud.xyz"
    # if not os.path.exists(pointcloud) or os.path.getsize(pointcloud) == 0:
    #     raise FileNotFoundError(f"Point cloud file '{pointcloud}' is empty or does not exist.")
    # points = np.loadtxt(pointcloud, delimiter=' ')
    # # Extract bounding box parameters
    # x_min, y_min, w, h = box
    # x_max, y_max = x_min + w, y_min + h
    
    # # Filter points inside the bounding box
    # filtered_points = points[
    #     (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    #     (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    # ]
    
    # if filtered_points.shape[0] < 4:
    #     raise ValueError("Not enough points in the selected bounding box to compute volume.")
    
    # # Compute convex hull volume
    # hull = scipy.spatial.ConvexHull(filtered_points)
    # volume = hull.volume
    
    # return volume

def get_carbs_from_volume(volume, detection):
    mass = get_mass_from_vol(volume, detection)
    carbs = mass_to_carbs_dict[detection] * mass
    return round(carbs, 2) #Placeholder for now

def get_mass_from_vol(volume, detection):
    return vol_to_mass_dict[detection] * volume
    


def get_detection_crops(detections, boxes, imgFile):
    mainImg = cv2.imread(imgFile) #cv2.imread("mainImg.png")
    detection_crops = []
    # boxes = detections
    for d in range(0, len(detections)):
        box = boxes[d]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cropped_img = mainImg[y:y+h, x:x+w]
        name = "detection_" + str(d) + ".png"
        cv2.imwrite(name, cropped_img)
        detection_crops.append(name)

    return detection_crops


def createResponseZip(message, mainImg, foodImages, foodData):
    zip_filename = "files_bundle.zip"

    #Create zip file and add images
    with zipfile.ZipFile(zip_filename, "w") as  zipf:
        zipf.write(mainImg)
        for img in foodImages:
            zipf.write(img)

        with open("foodData.txt", "w") as info_file:
            info_file.write(f"Message: {message}\n")

            for item, carbs in foodData.items():
                # info_file.write(f"Food data: \n{foodData}")
                info_file.write(item + ":" + str(carbs))

        zipf.write("foodData.txt")

    return send_file(zip_filename, as_attachment=True)

def clearUploadDir():
    print("\nAttempting to clear upload directory...", flush=True)
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
        print("Successfully cleared upload directory!\n", flush=True)

def processImage(imgFilePath):
    print("Running image processing on the following files:")

    analysis = yolov2_model.analyse_image(imgFilePath)
    labelled_img = analysis[0]
    detections = analysis[1]
    
    return labelled_img, detections

def get_git_commit_count():
    try:
        # Run git command to get the number of commits
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        return "?"  # In case git fails

if __name__ == '__main__':
    # clearUploadDir()
    version = "0.3." + str(get_git_commit_count())
    print("-------------------------------------------------")
    print("\nC.A.R.B.S Processing backend v" + version + "\n")
    app.run(host="0.0.0.0")#, debug=True)
    # app.run(host="192.168.1.168", port=5000)#, debug=True)
    # app.run(host="10.180.229.100", port=5000)
    # print(processImages())