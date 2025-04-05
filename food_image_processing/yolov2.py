import cv2
import numpy as np
import math
from tqdm import tqdm
import re
cfg_path="""food_image_processing/yolov2-food100.cfg"""
weights_path = """food_image_processing/yolov2-food100.weights"""
names_path = """food_image_processing/food100.names"""

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

with open(names_path, "r") as nameFile:
    classes = [line.strip() for line in nameFile.readlines()]

def get_unique_detection_id(all_detections, detection, count=1):
        existing_ids = [item[0] for item in all_detections]

        if detection not in existing_ids:
            return detection
        
        new_id = f"{detection}_{count}" if "_" not in detection else f"{detection.rsplit("_", 1)[0]}_{count}"
        # Recurse if the new ID also exists
        return get_unique_detection_id(new_id, all_detections, count + 1)

class YoloV2:
    @staticmethod
    def process_image(image):
        """Runs YoloV2 on a given image and returns detections"""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        conf_threshold = 0.75  # Confidence threshold
        nms_threshold = 0.2   # Non-Maximum Suppression threshold

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detections = []

        if len(indices) > 0 and isinstance(indices, np.ndarray):
            indices = indices.flatten()
            for i in indices:
                class_name = classes[class_ids[i]]
                detections.append([class_name, boxes[i], confidences[i]])

        return detections
    
    @staticmethod
    def adjust_bbox_for_rotation(box, angle, img_shape):
        """Convert bbox coordinates from a rotated image to coordinates on the original image"""
        h, w = img_shape[:2]
        x, y, width, height = box

        if angle == 90:
            new_x, new_y = y, w - (x + width)
            new_width, new_height = height, width
        elif angle == 180:
            new_x, new_y = w - (x + width), h - (y + height)
            new_width, new_height = width, height
        elif angle == 270:
            new_x, new_y = h - (y + height), x
            new_width, new_height = height, width
        else:
            return box  # No rotation

        return [new_x, new_y, new_width, new_height]
    

    @staticmethod
    def analyse_image(path):
        """Analyses the image, returns detections, handles applying augmentations
        THIS IS THE MAIN METHOD TO USE FOR IMAGE ANALYSIS
        """
        print("Analysing image...")
        original_image = cv2.imread(path)
        height, width = original_image.shape[:2]

        all_detections = []
        all_confidences = []
        #First run on the initial image
        detections = YoloV2.process_image(original_image)
        for class_name, box, confidence in detections:
            all_detections.append([class_name, box])
            all_confidences.append(confidence)

        segment_size = math.ceil(width / 3)
        step = math.ceil(width / 15)
        print("Augmenting image...")
        vertical_crops = (height - segment_size) // step + 1
        horizontal_crops = (width - segment_size) // step + 1
        total_crops = vertical_crops * horizontal_crops# * 4
        print("With image resolution, this will apply", total_crops, "augmentations.")
        #Segment and rotate the input and process 

        for y in tqdm(range(0, height - segment_size + 1, step)):
            for x in range(0, width - segment_size + 1, step):
                segment = original_image[y:y+segment_size, x:x+segment_size]

                # #Apply rotations
                # for angle in [0, 90, 180, 270]:
                #     if angle == 0:
                #         rotated_segment = segment
                #     elif angle == 90:
                #         rotated_segment = cv2.rotate(segment, cv2.ROTATE_90_CLOCKWISE)
                #     elif angle == 180:
                #         rotated_segment = cv2.rotate(segment, cv2.ROTATE_180)
                #     elif angle == 270:
                #         rotated_segment = cv2.rotate(segment, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                #Run yolov2 and collect detections
                detections = YoloV2.process_image(segment)

                for class_name, box, confidence in detections:
                    # Adjust bounding box from rotated segment back to the original image
                    angle = 0
                    adjusted_box = YoloV2.adjust_bbox_for_rotation(box, angle, segment.shape)

                    # Offset it to match original image coordinates
                    adjusted_box[0] += x
                    adjusted_box[1] += y
                    # all_detections.append([get_unique_detection_id(all_detections, class_name), adjusted_box])
                    unique_class_name = generate_unique_label(class_name, all_detections)
                    all_detections.append([unique_class_name, adjusted_box])
                    all_confidences.append(confidence)

        #Filter out duplicates/overlaps
        print("Detected: " + str(all_detections))

        filtered_detections = filter_overlapping_boxes(all_detections, all_confidences, width, height)

        #Draw detections on the image
        for class_name, box in filtered_detections:
            x, y, w, h = box
            # detection_names = [sublist[0] for sublist in filtered_detections]
            # label = generate_unique_label(class_name, detection_names)# {confidence:.2f}"
            # unique_detection_names.append(label)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            cv2.putText(original_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)

        cv2.namedWindow("YOLOv2 Detection", cv2.WINDOW_NORMAL)  # Allow resizing
        cv2.resizeWindow("YOLOv2 Detection", 800, 600)  
        cv2.imshow("YOLOv2 Detection", original_image)
        cv2.waitKey(5)
        cv2.destroyAllWindows()
        cv2.imwrite("mainImg.png", original_image)

        # # Save and display final image
        # cv2.imwrite("mainImg.png", original_image)
        # cv2.imshow("YOLOv2 Detection", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return "mainImg.png", filtered_detections, all_confidences


    

def generate_unique_label(base_label, existing_detections):
    existing_labels = [sublist[0] for sublist in existing_detections]
    if base_label not in existing_labels:
        return base_label
    
    counter = 1
    new_label = f"{base_label}_{counter}"
    while new_label in existing_labels:
        counter += 1
        new_label = f"{base_label}_{counter}"
    
    return new_label

def filter_overlapping_boxes(all_detections, all_confidences, image_width, image_height):
    def is_box_inside(inner, outer):
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh

    def do_boxes_overlap(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def normalize_class_name(class_name):
        return re.sub(r"_\d+$", "", class_name)


    # Step 0: Remove boxes too close to the image edges (within 10 pixels)
    detections = [
        (class_name, box, confidence)
        for (class_name, box), confidence in zip(all_detections, all_confidences)
        if box[0] > 10 and box[1] > 10 and box[0] + box[2] < image_width - 10 and box[1] + box[3] < image_height - 10
    ]


    # Step 1: Combine into (original_class_name, box, confidence)
    detections = [
        (class_name, box, confidence)
        for (class_name, box), confidence in zip(all_detections, all_confidences)
    ]

    # Step 2: Remove boxes entirely inside another
    filtered = []
    for i, (c1, b1, conf1) in enumerate(detections):
        keep = True
        for j, (c2, b2, conf2) in enumerate(detections):
            if i != j and is_box_inside(b1, b2):
                keep = False
                break
        if keep:
            filtered.append((c1, b1, conf1))

    # Step 3: Remove smaller box when different normalized class names overlap > 50%
    to_remove = set()
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            c1, b1, conf1 = filtered[i]
            c2, b2, conf2 = filtered[j]
            norm1 = normalize_class_name(c1)
            norm2 = normalize_class_name(c2)
            iou = compute_iou(b1, b2)
            print("IoU of " + c1 + " and " + c2 + " is " + str(iou))
            if norm1 != norm2 and iou > 0.3:
                area1 = b1[2] * b1[3]
                area2 = b2[2] * b2[3]
                
                if area1 < area2:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    filtered = [det for idx, det in enumerate(filtered) if idx not in to_remove]

    # Step 4: Group and merge overlapping boxes by normalized class
    visited = [False] * len(filtered)
    final_merged = []

    for i in range(len(filtered)):
        if visited[i]:
            continue
        class_i, box_i, conf_i = filtered[i]
        norm_class = normalize_class_name(class_i)

        # Start new group
        group = [(class_i, box_i, conf_i)]
        visited[i] = True

        for j in range(i + 1, len(filtered)):
            if visited[j]:
                continue
            class_j, box_j, conf_j = filtered[j]
            if normalize_class_name(class_j) == norm_class and do_boxes_overlap(box_i, box_j):
                group.append((class_j, box_j, conf_j))
                visited[j] = True

        # Expand group for transitive overlaps
        changed = True
        while changed:
            changed = False
            for k in range(len(filtered)):
                if visited[k]:
                    continue
                class_k, box_k, conf_k = filtered[k]
                if normalize_class_name(class_k) != norm_class:
                    continue
                for _, gb, _ in group:
                    if do_boxes_overlap(box_k, gb):
                        group.append((class_k, box_k, conf_k))
                        visited[k] = True
                        changed = True
                        break

        # Merge group into one box
        min_x = min(b[0] for _, b, _ in group)
        min_y = min(b[1] for _, b, _ in group)
        max_x = max(b[0] + b[2] for _, b, _ in group)
        max_y = max(b[1] + b[3] for _, b, _ in group)
        merged_box = [min_x, min_y, max_x - min_x, max_y - min_y]

        # Pick class with highest confidence
        best_idx = max(range(len(group)), key=lambda i: group[i][2])
        best_class_name = group[best_idx][0]
        best_conf = group[best_idx][2]

        final_merged.append((best_class_name, merged_box, best_conf))

    return [(c, b) for c, b, _ in final_merged]


def compute_iou(box1, box2):
    """Computes IoU (Intersection over Union) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    # Avoid division by zero
    return intersection / union if union > 0 else 0


def is_box_inside(inner, outer):
    """Checks if the inner box is completely inside the outer box."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    
    # Check if the inner box is within the outer box bounds
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh

    # def analyse_image(path):
    #     detections = []
    #     image = cv2.imread(path)
    #     height, width = image.shape[:2]
    #     # Convert image to blob format
    #     blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    #     net.setInput(blob)
    #     # Get output layer names
    #     layer_names = net.getLayerNames()
    #     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    #     # Perform forward pass
    #     outputs = net.forward(output_layers)

    #     # Process detections
    #     conf_threshold = 0  # Confidence threshold
    #     nms_threshold = 0.2   # Non-Maximum Suppression threshold

    #     boxes = []
    #     confidences = []
    #     class_ids = []

    #     for output in outputs:
    #         for detection in output:
    #             scores = detection[5:]  # Class scores start from index 5
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]

    #             if confidence > conf_threshold:
    #                 # YOLOv2 outputs center x, center y, width, height as a percentage of image size
    #                 center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")

    #                 # Convert to top-left corner format
    #                 x = int(center_x - (w / 2))
    #                 y = int(center_y - (h / 2))

    #                 boxes.append([x, y, w, h])
    #                 confidences.append(float(round(confidence, 2)))
    #                 class_ids.append(class_id)
    #     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    #     # Check if indices is not empty and is a NumPy array
    #     if len(indices) > 0 and isinstance(indices, np.ndarray):
    #         indices = indices.flatten()
    #     else:
    #         indices = []  # Set to an empty list if there are no valid detections
    #         mainImg = cv2.imwrite("mainImg.png", image)
    #         return "mainImg.png", []
    #     #Establish bounding box around entire image/detections
    #     x_min = width
    #     y_min = height
    #     x_max = 0
    #     y_max = 0

        

    #     for i in indices:
    #         class_name = classes[class_ids[i]]
    #         confidence = confidences[i]
    #         print(f"Detected: {class_name} - Confidence: {confidence:.2f}")
    #         # detections.append(class_name)
    #         detections.append([class_name, boxes[i]])
    #         x, y, w, h = boxes[i]
    #         label=f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 40)
    #         cv2.putText(image, label, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            


    #     for i in indices:
    #         x, y, w, h = boxes[i]
    #         x_min = min(x_min, x)
    #         y_min = min(y_min, y)
    #         x_max = max(x_max, x + w)
    #         y_max = max(y_max, y + h)

    #         # Expand bounding box by 200 pixels on each side
    #         padding = 500
    #         x_min = max(0, x_min - padding)
    #         y_min = max(0, y_min - padding)
    #         x_max = min(width, x_max + padding)
    #         y_max = min(height, y_max + padding)

    #         # Crop the image
    #         cropped_mainImg = image[y_min:y_max, x_min:x_max]
    #     cv2.namedWindow("YOLOv2 Detection", cv2.WINDOW_NORMAL)  # Allow resizing
    #     cv2.resizeWindow("YOLOv2 Detection", 800, 600)  
    #     cv2.imshow("YOLOv2 Detection", cropped_mainImg)
    #     cv2.waitKey(5)
    #     cv2.destroyAllWindows()
    #     cv2.imwrite("mainImg.png", cropped_mainImg)
    #     return "mainImg.png", detections, confidences