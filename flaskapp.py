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

@app.route("/", methods=['GET'])
def handle_call():
    return "C.A.R.B.S Processing Backend v" + version + " Successfully Connected"


@app.route("/upload-image", methods=['POST'])
def store_image():
    if 'image' not in request.files:
        print("Error: recieved image upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    image = request.files['image']
    if len(os.listdir(uploadDir)) >= 2:
        clearUploadDir()

    filePathToSave = f"./upload/{image.filename}"
    if os.path.isfile(filePathToSave):
        newFileName = image.filename.split(".")[0] + "_1"
        print("An existing image already exists, renaming image file")
        filePathToSave = f"./upload/{newFileName}" + ".jpg"
    # image.save(f"./upload/{image.filename}")
    image.save(filePathToSave)
    print("Recieved image from app capture, stored at: " + filePathToSave, flush=True)
    if len(os.listdir(uploadDir)) >= 2:
        detections = processImages()
        print("Detected: ",detections)
        return{"message": str(detections)}
    return {"message": "Image recieved and stored successfully", "filename":image.filename}


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
    print(processImages())