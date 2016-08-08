import cv2
import sys
import time
import numpy as np

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(1)

recognizer = cv2.createLBPHFaceRecognizer(threshold=45.0)


# First we want to train the images
timer = time.time() + 7 #15 seconds later

images = []
labels = []

while time.time() < timer:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        images.append(gray[y: y+h, x: x+w])
        labels.append(0)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

timer = time.time() + 5 #15 seconds later
while (time.time() < timer):
    print("PAUSE")

timer = time.time() + 7 #15 seconds later
while time.time() < timer:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        images.append(gray[y: y+h, x: x+w])
        labels.append(1)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
recognizer.train(images,np.array(labels))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        if(nbr_predicted == 0):
            print "{} is Correctly Recognized with confidence {}".format(0, conf)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif(nbr_predicted == 1):
            print "{} is Correctly Recognized with confidence {}".format(1, conf)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print "Incorrectly Recognized " + str(nbr_predicted)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
