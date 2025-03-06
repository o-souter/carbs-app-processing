import flask
from flask import request
import os, shutil
import cv2
import subprocess

# from food_image_processing import food_detection
from food_image_processing import yolov2
app = flask.Flask(__name__)
uploadDir = "upload"#

yolov2_model = yolov2.YoloV2

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

    # if len(os.listdir(uploadDir)) >= 2:
    #     detections = processImages()
    #     print("Detected: ",detections)
    #     return{"message": str(detections)}
    
    
    return {"message": "Image and pointcloud recieved and stored successfully", "filename":image.filename}


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

def processImages():
    imagePaths = os.listdir(uploadDir)
    print("Running image processing on the following files:")
    for file in imagePaths:
        print(file)
    
    return yolov2_model.analyse_image(os.path.join(uploadDir, file))

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
    version = "0.1." + str(get_git_commit_count())
    print("-------------------------------------------------")
    print("\nC.A.R.B.S Processing backend v" + version + "\n")
    app.run(host="0.0.0.0")#, debug=True)
    # app.run(host="192.168.1.168", port=5000)#, debug=True)
    # app.run(host="10.180.229.100", port=5000)
    print(processImages())