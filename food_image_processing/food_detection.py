import numpy as np
import tensorflow as tf
import cv2
import tensorflow.lite as tflite
import os

model_path = "food_image_processing\yolov2-food100.tflite"
if os.path.exists(model_path):
    print("Model file exists.")
else:
    print("Model file NOT found! Check the path.")
class FoodDetectionModel:
    #Load our model
    interpreter = tflite.Interpreter(model_path=model_path)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    
    output_details = interpreter.get_output_details()
    print(f"Model Input Shape: {input_details[0]['shape']}")
    print(f"Model Output Shape: {output_details[0]['shape']}")
    interpreter.allocate_tensors()
    print(f"Input details:" , interpreter.get_input_details())
    def get_input_shape():
        input_shape = FoodDetectionModel.input_details[0]['shape']
        print(f"Model input shape: {input_shape}")
        
        return input_shape

    def preprocess_image(image_path, input_shape=(416, 416)):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (input_shape[1], input_shape[2]))
        image = image.astype(np.float32) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    
    def detect_foods(input_data):
        FoodDetectionModel.interpreter.set_tensor(FoodDetectionModel.input_details[0]['index'], input_data)
        FoodDetectionModel.interpreter.invoke()
        return FoodDetectionModel.interpreter.get_tensor(FoodDetectionModel.output_details[0]['index'])
    

    def decode_yolo_output(results, img_shape, confidence_threshold=0.4):
        boxes = []
        confidences = []
        class_ids = []

        for detection in results:
            boxes = []
            confidences = []
            class_ids = []
            print("Detection:", detection)  # Inspect the detection structure
            
            if len(detection) >= 6:
                x, y, width, height, confidence, class_id = detection
            else:
                print(f"Unexpected detection format: {detection}")
                continue  # Skip this detection if it doesn't have enough values
            
            boxes.append((x, y, width, height))
            confidences.append(confidence)
            class_ids.append(class_id)

        return boxes, confidences, class_ids
    
    def load_class_names(file_path):
        with open(file_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def draw_boxes(image_path, boxes, confidences, class_ids, class_names=load_class_names("food_image_processing\\food100.names")):
        # Load the image
        image = cv2.imread(image_path)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            
            # Draw the bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the image with bounding boxes
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
