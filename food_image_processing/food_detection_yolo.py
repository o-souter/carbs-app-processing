import cv2
import numpy as np

cfg_path="""food_image_processing\\yolov2-food100.cfg"""
weights_path = """food_image_processing\\yolov2-food100.weights"""
names_path = """food_image_processing\\food100.names"""

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

with open(names_path, "r") as nameFile:
    classes = [line.strip() for line in nameFile.readlines()]

test_image_path = "upload\fishandchips.jpg"
image = cv2.imread(test_image_path)

height, width = image.shape[:2]

# Convert image to blob format
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform forward pass
outputs = net.forward(output_layers)

# Process detections
conf_threshold = 0.2  # Confidence threshold
nms_threshold = 0.4   # Non-Maximum Suppression threshold

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]  # Class scores start from index 5
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # YOLOv2 outputs center x, center y, width, height as a percentage of image size
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")

            # Convert to top-left corner format
            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Check if indices is not empty and is a NumPy array
if len(indices) > 0 and isinstance(indices, np.ndarray):
    indices = indices.flatten()
else:
    indices = []  # Set to an empty list if there are no valid detections

for i in indices:
    class_name = classes[class_ids[i]]
    confidence = confidences[i]
    print(f"Detected: {class_name} - Confidence: {confidence:.2f}")

    x, y, w, h = boxes[i]
    label=f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# Show image
cv2.namedWindow("YOLOv2 Detection", cv2.WINDOW_NORMAL)  # Allow resizing
cv2.resizeWindow("YOLOv2 Detection", 800, 600)  
cv2.imshow("YOLOv2 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()