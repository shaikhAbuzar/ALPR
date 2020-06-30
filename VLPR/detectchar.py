import numpy as np
import time
import cv2
import os

labelsPathChar = os.path.sep.join(["weights", "characters.names"])
LABELSChar = open(labelsPathChar).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORSChar = np.random.randint(0, 255, size=(len(LABELSChar), 3), dtype="uint8")

# derive the paths to the weights and model configuration
weightsPathChar = os.path.sep.join(["weights", "characters.weights"])
configPathChar = os.path.sep.join(["weights", "characters.cfg"])

# load our YOLO object detector trained on dataset
netChar = cv2.dnn.readNetFromDarknet(configPathChar, weightsPathChar)

# load our input image and grab its spatial dimensions
def detect_chars(image):
    image_copy = image.copy()
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need
    ln = netChar.getLayerNames()
    ln = [ln[i[0] - 1] for i in netChar.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    netChar.setInput(blob)
    start = time.time()
    layerOutputs = netChar.forward(ln)
    end = time.time()

    # show timing information
    print("\t[INFO] Took {:.3f} seconds for detection of characters".format(end - start))

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
            if confidence >= 0.15:
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

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.05)
    string = ''
    for classID in classIDs:
        string += LABELSChar[classID]

    # dictinary storing all the objects
    chars_list = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORSChar[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

            text = "{}".format(LABELSChar[classIDs[i]])
            chars_list.append(LABELSChar[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  

    print(f'\t[INFO] Detected plate with Plate No. {string}')
    # cv2.imshow('Chars', image)
    # cv2.waitKey(0)
    # cv2.imwrite('BoxedResult.jpg', image)
    return image, string
