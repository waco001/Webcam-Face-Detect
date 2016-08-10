#!/usr/bin/env python
import cv2
import sys
import time
import numpy as np
import rospy
import Image as PIL
from sensor_msgs.msg import Image
import os
from cv_bridge import CvBridge, CvBridgeError
import json
import time
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
recognizerFile = sys.argv[2]
pub = None
bridge = CvBridge()
configData = json.load(open('config.json'))
recognizer = cv2.createLBPHFaceRecognizer(threshold=configData['threshold'])
croppedData = []
if os.path.exists(recognizerFile):
    recognizer.load(recognizerFile)
else:
    recognizer.save(recognizerFile)
def callback(data):
    frame = None
    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(configData['detection']['x'], configData['detection']['y']),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces: 
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        print("Recognized: " + str(nbr_predicted) + " with conf: " + str(conf))
        if(nbr_predicted == -1):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imwrite("faces/" + str(nbr_predicted) + "," + str(time.time()) + ".jpg",cv2.resize(gray[y:(y+h), x:(x+w)], (200,200)))
        pub.publish(bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
if __name__ == '__main__':
    try:
        rospy.init_node('facialRecognizer', anonymous=True)
        pub = rospy.Publisher('/sar/perception/right_cam/usb_cam_1/image_recognized', Image, queue_size=10)
        rospy.Subscriber("/sar/perception/right_cam/usb_cam_1/image_raw", Image, callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    print(croppedData)