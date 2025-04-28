import flask
from flask import request, Flask, send_file, jsonify
import zipfile
import os, shutil
import cv2
import subprocess
import numpy as np
import scipy
import random
import re
import math
# import pandas as pd
from flask_cors import CORS

# from food_image_processing import food_detection
from food_image_processing import yolov2
app = flask.Flask(__name__)
CORS(app)
uploadDir = "upload"
augmentationDir = "augmentations"
detectionDir = "detections"
bundlingDir = "bundling"


serverMainVersion = "1.0"

markerSizeCM = 5

markerFound = True

yolov2_model = yolov2.YoloV2

#Set up volume to mass, mass to carb dictionaries

with open("volume_estimation/vol_to_mass.txt", "r") as file:
    vol_to_mass_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}

with open("volume_estimation/mass_to_carbs.txt", "r") as file:
    mass_to_carbs_dict = {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in file}


# print("Volume to Mass dictionary: \n" + str(vol_to_mass_dict), flush=True)

# print("Mass to Carbs Dictionary: \n" + str(mass_to_carbs_dict), flush=True)

food_class_file_path = "food_image_processing/food100.names"

@app.route("/test", methods=['GET'])
def handle_call():
    print("Successfully connected to android device.", flush=True)
    return "C.A.R.B.S Processing Backend Successfully Connected"


@app.route("/upload-data", methods=['POST'])
def store_data():
    if 'image' not in request.files:
        print("Error: recieved data upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    # if 'pointcloud' not in request.files:
    #     print("Error: recieved data upload request but request contained no pointcloud file", flush=True)
    #     return {"error": "No pointcloud file found"}
    
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
    
    # pointcloud = request.files['pointcloud']
    # pointFilePathToSave = f"./upload/{pointcloud.filename}"
    # if os.path.isfile(pointFilePathToSave):
    #     newFileName = pointcloud.filename.split(".")[0] + "_1"
    #     print("An existing pointcloud already exists, renaming pointcloud file")
    #     pointFilePathToSave = f"./upload/{newFileName}" + ".jpg"
    
    # pointcloud.save(pointFilePathToSave)
    # print("Recieved pointcloud from app capture, stored at: " + pointFilePathToSave, flush=True)
    
    clearDetectionDir()
    
    mainImg, foodImages, foodData = run_carbs_algo(imgFilePathToSave)
    return createResponseZip("Data processed successfully. MarkerFound=" + str(markerFound), mainImg, foodImages, foodData)


@app.route("/get-food-classes", methods=['GET'])
def get_food_classes():
    print("Received request for all food item classes", flush=True)
    with open(food_class_file_path, "r") as file:
        classes = sorted(file.read().splitlines())
    return  " ".join(classes)
    

#{"message": "Image and pointcloud recieved and stored successfully", "filenames":[image.filename, pointcloud.filename]}



def run_carbs_algo(imgFile):
    # foodData = "Not implemented yet!"
    # augment_image(imgFile, "augmentations")
    
    processedImage = processImage(imgFile)
    labelled_img = processedImage[0]
    detections = [item[0] for item in processedImage[1]]# processedImage[1]
    boxes = [item[1] for item in processedImage[1]]
    confidences = processedImage[2]
    # carbs = np.full(shape=len(detections), fill_value=50, dtype=np.double) #np.zeros_like(detections, float)
    keys = tuple(detections)
    carbs_data, volume_data, weight_data = calculate_carbs(imgFile, detections, boxes)
    #Maintain the same DP
    # carbs_data = [ '%.2f' % elem for elem in carbs_data ]
    # volume_data = [ '%.2f' % elem for elem in volume_data ]
    # weight_data = [ '%.2f' % elem for elem in weight_data ]



    foodData = dict(zip(keys, zip(carbs_data, confidences, volume_data, weight_data)))
    detection_images = get_detection_crops(detections, boxes, imgFile)
    
    
    return labelled_img, detection_images, foodData

def calculate_carbs(imgFile, detections, boxes):
    carb_list = []
    volume_list = []
    weight_list = []

    for d in range(0, len(detections)):
        detection = detections[d]
        print("\nCalculating carbs for: " + str(detection))
        #TODO add edge detection to improve 3D shaping of food here

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
    # print(volume_list)
    return carb_list, volume_list, weight_list



# def get_carbs_from_volume(volume, detection):
#     mass = get_mass_from_vol(volume, detection)
#     print("Estimated mass/weight: " + str(mass) + "g")
#     carbs = round(mass_to_carbs_dict[detection] * mass, 2)
#     print("Estimated Carbohydrates: " + str(carbs) + "g")
#     return carbs 


def get_carbs_from_weight(weight, detection):
    detectionclass = re.sub(r'_\d+$', '', detection)
    carbs = mass_to_carbs_dict[detectionclass] * weight
    print("Estimated Carbohydrates: " + str(carbs) + "g")
    return carbs

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


#Image Augmentation
def random_crop(image, crop_size=(200, 200)):
    """ Randomly crops the image to a given size """
    h, w = image.shape[:2]
    ch, cw = crop_size

    if ch > h or cw > w:
        raise ValueError("Crop size should be smaller than the original image size")

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    return image[y:y+ch, x:x+cw]

def grid_crop(image, crop_size, step_x, step_y):
    """ 
    Crops the image systematically from left to right, top to bottom.
    `crop_size`: Tuple (width, height) defining crop dimensions.
    `step_x`: Horizontal step size (move right by this many pixels).
    `step_y`: Vertical step size (move down by this many pixels).
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    cropped_images = []
    
    for y in range(0, h - crop_h + 1, step_y):
        for x in range(0, w - crop_w + 1, step_x):
            cropped = image[y:y+crop_h, x:x+crop_w]
            cropped_images.append((cropped, x, y))  # Store crop with position

    return cropped_images

def adjust_contrast(image, alpha):
    """ Adjusts contrast by multiplying pixels by alpha (1.0 = no change) """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted

def rotate_image(image, angle):
    """ Rotates the image by the given angle """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def augment_image(image_path, output_dir, num_crops=20, crop_size=(1008, 1344), step_x=150, step_y=150, rotations=np.arange(0, 360, 20), contrast_levels=np.arange(0.2, 2.0, 0.2)):
    """ Applies cropping, rotation, and contrast adjustments to create multiple images """
    os.makedirs(output_dir, exist_ok=True)
    clearAugmentationDir()
    print("Augmenting Image...", flush=True)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # # Apply Cropping
    # for i in range(num_crops):
    #     cropped = random_crop(image, crop_size=(1344, 1008))
    #     cv2.imwrite(os.path.join(output_dir, f"{base_name}_crop{i}.jpg"), cropped)
     # Grid-based Cropping
    cropped_images = grid_crop(image, crop_size, step_x, step_y)
    for i, (cropped, x, y) in enumerate(cropped_images):
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_crop_{x}_{y}.jpg"), cropped)

    # Apply Rotations
    for angle in rotations:
        rotated = rotate_image(image, angle)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_rot{angle}.jpg"), rotated)

    # Apply Contrast Adjustments
    for level in contrast_levels:
        contrasted = adjust_contrast(image, level)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_contrast{level}.jpg"), contrasted)

    print("Augmentation complete! Images saved to:", output_dir)


vol_accurate_threshold = 7

def calculate_volume(foodPath, box, markerLength):
    foodImg = cv2.imread(foodPath)
    pixel_to_real = detect_aruco_marker(foodImg, markerLength)
    global markerFound
    if pixel_to_real is None:
        # raise ValueError("ArUco marker not detected.")
        pixel_to_real = 0.01
        markerFound = False  # TODO add feedback here to alert user a marker wasn't found
        print("Warning: No fiducial marker detected. Using default scaling.", flush=True)
    else:
        markerFound = True
        
    x_min, y_min, width, height = box
    real_width = width * pixel_to_real
    real_height = height * pixel_to_real
    real_depth = min(real_width, real_height) * 0.5
    return real_width * real_height * real_depth
    # Extend the bounding box by 50px in each direction
    padding = 50
    x_min_extended = max(0, x_min - padding)  # Ensure we don't go below 0
    y_min_extended = max(0, y_min - padding)  # Ensure we don't go below 0
    width_extended = width + 2 * padding
    height_extended = height + 2 * padding

    # Crop the region with the extended bounding box
    cropped = foodImg[y_min_extended:y_min_extended + height_extended, x_min_extended:x_min_extended + width_extended]

    # Preprocess the crop
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = thresh
    debug_output = cropped.copy()

    # Detect contours/edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found. Falling back to bounding box estimation.")
        real_width = width * pixel_to_real
        real_height = height * pixel_to_real
        real_depth = min(real_width, real_height) * 0.5
        return real_width * real_height * real_depth

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) >= 5:
        # Fit ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes) * pixel_to_real
        minor_axis = min(axes) * pixel_to_real

        real_area = math.pi * (major_axis / 2) * (minor_axis / 2)
        estimated_depth = min(major_axis, minor_axis) * 0.5
        volume = real_area * estimated_depth

        if volume < vol_accurate_threshold:
            print("Below par ellipse detected. Falling back to bounding box estimation.")
            real_width = width * pixel_to_real
            real_height = height * pixel_to_real
            real_depth = min(real_width, real_height) * 0.5
            return real_width * real_height * real_depth

        print(f"Ellipse-based area: {real_area:.2f}cm²")
        print(f"Estimated depth: {estimated_depth:.2f}cm")
        print(f"Estimated volume: {volume:.2f}cm³")

        # Draw ellipse on debug image
        cv2.ellipse(debug_output, ellipse, (0, 255, 0), 2)
    else:
        print("Not enough points for ellipse fitting. Falling back to bounding box.")
        real_width = width * pixel_to_real
        real_height = height * pixel_to_real
        real_depth = min(real_width, real_height) * 0.5
        return real_width * real_height * real_depth

    # Show debug images
    # cv2.imshow("Cropped Image with Ellipse", debug_output)
    # cv2.imshow("Threshold", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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


def get_mass_from_vol(volume, detection):
    detection_class = re.sub(r'_\d+$', '', detection)
    mass = vol_to_mass_dict[detection_class] * volume
    print("Estimated weight: " + str(mass) + "g")
    return mass


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
        name = "detection_" + detections[d] + ".png"
        path = os.path.join(detectionDir, name)
        try:
            cv2.imwrite(path, cropped_img)
            detection_crops.append(path)

        except Exception as e:
            detection_crops.append("assets/no_crop_found.png")

    return detection_crops


def createResponseZip(message, mainImg, foodImages, foodData):
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

def clearAugmentationDir():
    # print("Attempting to clear augmentations directory...", flush=True)
    
    for filename in os.listdir(augmentationDir):
        file_path = os.path.join(augmentationDir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e) + "\n")
    if len(os.listdir(augmentationDir)) == 0:
        print("Successfully cleared augmentation directory!", flush=True)

def clearDetectionDir():
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


def processImage(imgFilePath):
    print("Running image processing on the following files:")

    analysis = yolov2_model.analyse_image(imgFilePath)
    labelled_img = analysis[0]
    detections = analysis[1]
    if len(detections) > 0:
        confidences = analysis[2]
    else:
        confidences = [0.0]
    
    return labelled_img, detections, confidences

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
    version = get_version_number()
    clearAugmentationDir()
    clearUploadDir()
    clearDetectionDir()
    print("-" * 80)
    print("\nC.A.R.B.S Processing backend v" + str(version) + "\n")
    
    app.run(host="0.0.0.0", port=8000)#, debug=True)
    # app.run(host="192.168.1.168", port=5000)#, debug=True)
    # app.run(host="10.180.229.100", port=5000)
    # print(processImages())