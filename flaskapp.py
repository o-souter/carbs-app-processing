import flask
from flask import request
import requests
import PIL
from io import BytesIO
import base64
from PIL import Image
import os, shutil

app = flask.Flask(__name__)
uploadDir = "upload"


@app.route("/", methods=['GET'])
def handle_call():
    return "Successfully Connected"

# @app.route("/")

@app.route("/upload-image", methods=['POST'])
def store_image():
    if 'image' not in request.files:
        print("Error: recieved image upload request but request contained no image", flush=True)
        return {"error": "No image file found"}
    image = request.files['image']
    image.save(f"./upload/{image.filename}")
    print("Recieved and stored image from app capture, at: " + f"./upload/{image.filename}", flush=True)
    return {"message": "Image recieved and stored successfully", "filename":image.filename}
    # # ----- SECTION 1 -----  
    # #File naming process for nameless base64 data.
    # #We are using the timestamp as a file_name.
    # from datetime import datetime
    # dateTimeObj = datetime.now()
    # file_name_for_base64_data = dateTimeObj.strftime("%d-%b-%Y--(%H-%M-%S)")
    
    # #File naming process for directory form <file_name.jpg> data.
    # #We are taken the last 8 characters from the url string.
    # file_name_for_regular_data = url[-10:-4]
    # data = request.form
    # fileName = data.get("name")
    # print("Recieved the file: " + fileName)
    # ----- SECTION 2 -----
    # try:
    #     # Base64 DATA
    #     if "data:image/jpeg;base64," in url:
    #         base_string = url.replace("data:image/jpeg;base64,", "")
    #         decoded_img = base64.b64decode(base_string)
    #         img = Image.open(BytesIO(decoded_img))

    #         file_name = file_name_for_base64_data + ".jpg"
    #         img.save(file_name, "jpeg")

    #     # Base64 DATA
    #     elif "data:image/png;base64," in url:
    #         base_string = url.replace("data:image/png;base64,", "")
    #         decoded_img = base64.b64decode(base_string)
    #         img = Image.open(BytesIO(decoded_img))

    #         file_name = file_name_for_base64_data + ".png"
    #         img.save(file_name, "png")

    #     # Regular URL Form DATA
    #     else:
    #         response = requests.get(url)
    #         img = Image.open(BytesIO(response.content)).convert("RGB")
    #         file_name = file_name_for_regular_data + ".jpg"
    #         img.save(file_name, "jpeg")
        
    # # ----- SECTION 3 -----    
    #     status = "Image has been succesfully sent to the server."
    # except: #Exception as e:
    #     status = "Error! = " + str(e)


    return# status



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