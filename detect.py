import cv2
import numpy
import time


def detection(loc, net, args, labels):
    """Detection of vehicle and license plates using OpenCV DNN modules of Darknet"""
    numpy.random.seed(41)
    colors = numpy.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    if type(loc) is str:
        image = cv2.imread(loc)
    else:
        image = loc
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()
    mean_time = end-start
    boxes = []
    confidences = []
    class_ids = []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detections in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detections[5:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detections[0:4] * numpy.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                               args["threshold"])
    bboxes = []
    names = []
    conf = []
    if len(indexes) > 0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            bboxes.append((x, y, w, h))
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            names.append(labels[class_ids[i]])
            conf.append(confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    # return bounding box coordinates, class names, mean time and confidence level
    return tuple(bboxes), tuple(names), mean_time, tuple(conf)
