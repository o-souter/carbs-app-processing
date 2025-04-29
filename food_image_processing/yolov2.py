#yolov2.py - Handles food detection and segmentation using YoloV2, through its config and weights files

import cv2
import numpy as np
import math
from tqdm import tqdm
import re
import os

cfg_path="""food_image_processing/yolov2-food100.cfg"""
weights_path = """food_image_processing/yolov2-food100.weights"""
names_path = """food_image_processing/food100.names"""
bundlingDir = "bundling"

#Set up YoloV2
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Read class list
with open(names_path, "r") as nameFile:
    classes = [line.strip() for line in nameFile.readlines()]

class YoloV2:
    @staticmethod
    def process_image(image):
        """Runs YoloV2 on a given image and returns detections"""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)

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
        THIS IS THE MAIN METHOD TO CALL FOR IMAGE ANALYSIS
        """
        print("Analysing image...")
        original_image =  cv2.imread(path)
        height, width = original_image.shape[:2]

        all_detections = []
        all_confidences = []
        #First run on the initial image

        detections = YoloV2.process_image(original_image)
        for class_name, box, confidence in detections:
            all_detections.append([class_name, box])
            all_confidences.append(confidence)

        segment_size = math.ceil(width / 3)
        step = math.ceil(width / 10)
        print("Augmenting image...")
        vertical_crops = (height - segment_size) // step + 1
        horizontal_crops = (width - segment_size) // step + 1
        total_crops = vertical_crops * horizontal_crops  * 2
        print("With image resolution, this will apply", total_crops, "augmentations.")
        #Segment the input and process 

        for y in tqdm(range(0, height - segment_size + 1, step)):
            for x in range(0, width - segment_size + 1, step):
                segment = original_image[y:y+segment_size, x:x+segment_size]
                #Run yolov2 and collect detections
                detections = YoloV2.process_image(segment)
                for class_name, box, confidence in detections:
                    adjusted_box = box #YoloV2.adjust_bbox_for_rotation(box, angle, segment.shape)
                    # Offset it to match original image coordinates
                    adjusted_box[0] += x
                    adjusted_box[1] += y

                    unique_class_name = generate_unique_label(class_name, all_detections)
                    all_detections.append([unique_class_name, adjusted_box])
                    all_confidences.append(confidence)

        filtered_detections = filter_overlapping_boxes(all_detections, all_confidences, width, height)

        #Draw detections on the image
        for class_name, box in filtered_detections:
            x, y, w, h = box
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            cv2.putText(original_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)

        # Crop image to around where detections are (with 200px padding)
        if filtered_detections:
            x_coords = []
            y_coords = []
            for _, (x, y, w, h) in filtered_detections:
                x_coords.extend([x, x + w])
                y_coords.extend([y, y + h])

            min_x = max(min(x_coords) - 200, 0)
            max_x = min(max(x_coords) + 200, width)
            min_y = max(min(y_coords) - 200, 0)
            max_y = min(max(y_coords) + 200, height)

            cropped_image = original_image[min_y:max_y, min_x:max_x]
        else:
            cropped_image = original_image  # fallback to full image if no detections

        mainImgPath = os.path.join(bundlingDir, "mainImg.png")
        cv2.imwrite(mainImgPath, cropped_image)

        return mainImgPath, filtered_detections, all_confidences

def generate_unique_label(base_label, existing_detections):
    """Creates a unique label for an area where food is detected"""
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
    """Filter out boxes with the following rules:
    If a box is inside another, it is deleted
    If two boxes of the same class name overlap, they merge
    If a two boxes have IoU of >0.3, the smaller is deleted
    If a box is too close to the image edge, it is deleted
    """
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

    def normalize_class_name(class_name): #Remove any trailing _x from a class name so they can be compared and are not unique
        return re.sub(r"_\d+$", "", class_name)


    # Remove boxes too close to the image edges (within 10 pixels)
    detections = [
        (class_name, box, confidence)
        for (class_name, box), confidence in zip(all_detections, all_confidences)
        if box[0] > 10 and box[1] > 10 and box[0] + box[2] < image_width - 10 and box[1] + box[3] < image_height - 10
    ]


    # Combine into (original_class_name, box, confidence)
    detections = [
        (class_name, box, confidence)
        for (class_name, box), confidence in zip(all_detections, all_confidences)
    ]

    # Remove boxes entirely inside another
    filtered = []
    for i, (c1, b1, conf1) in enumerate(detections):
        keep = True
        for j, (c2, b2, conf2) in enumerate(detections):
            if i != j and is_box_inside(b1, b2):
                keep = False
                break
        if keep:
            filtered.append((c1, b1, conf1))

    # Remove smaller box when different normalized class names overlap
    to_remove = set()
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            c1, b1, conf1 = filtered[i]
            c2, b2, conf2 = filtered[j]
            norm1 = normalize_class_name(c1)
            norm2 = normalize_class_name(c2)
            iou = compute_iou(b1, b2)
            # print("IoU of " + c1 + " and " + c2 + " is " + str(iou))
            if norm1 != norm2 and iou > 0.3:
                area1 = b1[2] * b1[3]
                area2 = b2[2] * b2[3]
                
                if area1 < area2:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    filtered = [det for idx, det in enumerate(filtered) if idx not in to_remove]

    # Group and merge overlapping boxes by normalized class
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

