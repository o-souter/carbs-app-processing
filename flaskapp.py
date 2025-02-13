import flask
from flask import request
import os, shutil

app = flask.Flask(__name__)
uploadDir = "upload"


@app.route("/", methods=['GET'])
def handle_call():
    return "Successfully Connected"


@app.route("/upload-image", methods=['POST'])
def store_image():
    if 'image' not in request.files:
        print("Error: recieved image upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    image = request.files['image']
    image.save(f"./upload/{image.filename}")
    print("Recieved and stored image from app capture, at: " + f"./upload/{image.filename}", flush=True)
    return {"message": "Image recieved and stored successfully", "filename":image.filename}
    
    return



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
if __name__ == '__main__':
    clearUploadDir()

    app.run(host="0.0.0.0")