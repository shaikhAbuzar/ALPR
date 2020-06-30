# import the necessary packages
import numpy as np
import time
import cv2
import os

# importing self made libraries
import detectchar as dc

# Section For Number Plates
labelsPath = os.path.sep.join(["weights", "objLP.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["weights", "yolo-objLP_final.weights"])
configPath = os.path.sep.join(["weights", "yolo-objLP.cfg"])

# load our YOLO object detector trained on dataset
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

def yolo(image, frame_count):

    image_copy = image.copy()
    (H, W) = image.shape[:2]
    
    # determine only the *output* layer names that we need
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward
    # pass of the object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # show timing information
    print("[INFO] Took {:.2f} seconds for detection of plates".format(end - start))
    
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence >= 0.85:
                # scale the bounding box coordinates back relative to the
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    # dictinary storing all the objects
    objects_list = []
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
    
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
            text = 'MH XX xx XXXX'
            text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
            # adding the cropped image to the dictionary
            if True:
                if True:
                    try:
                        print('[INFO] Image saved')
                        image_char, license_plate = dc.detect_chars(image_copy[y:y + h, x:x + w])
                        cv2.imwrite("results\\%s.jpg"%(license_plate), image_char)
                        frame_count += 1
                        objects_list.append({'license_plate':license_plate})
                    except:
                        pass
    print('\n')
    return objects_list, frame_count