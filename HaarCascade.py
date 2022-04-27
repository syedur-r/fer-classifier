import cv2 as cv
import numpy as np
from keras.preprocessing import image


class FaceDetector:
    # constructor function with global variables
    def __init__(self, predictedEmotion=""):
        self.predictedEmotion = predictedEmotion
        self.faceHaarCascade = cv.CascadeClassifier(
            "haarcascade classifiers/haarcascade_frontalface_default.xml")  # initialise haarcascade
        # all emotions labels with a corresponding index
        self.emotions = {0: "Angry",
                         1: "Disgusted",
                         2: "Scared",
                         3: "Happy",
                         4: "Sad",
                         5: "Surprised",
                         6: "Neutral"
                         }

    # helper function for predicting the emotion and outputting it as text
    def detectEmotion(self, model, frame, draw=True):
        grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # converts the image to grayscale
        facesDetected = self.faceHaarCascade.detectMultiScale(grayImg, 1.32,
                                                              5)  # detects the faces in the input image
        extractedFrame = None

        for (x, y, w, h) in facesDetected:  # looping through the dimensions of the bounding box
            region_of_interest = grayImg[y:y + w, x:x + h]  # cropping face area from image
            region_of_interest = cv.resize(region_of_interest, (48, 48))  # resizes the region of interest to the
            # input shape of the model - (48, 48, 1)
            extractedFrame = region_of_interest

            if np.sum([region_of_interest]) != 0:  # checks if the sum of all elements is not equal to 0
                # converting the region of interest to an image array, expanding the dimensions and normalising it
                inputImage = image.img_to_array(region_of_interest)
                inputImage = np.expand_dims(inputImage, axis=0)
                inputImage /= 255

                # the model is predicting the emotions of the input image, storing it in a variable with its index
                # and outputting it in text
                predictions = model.predict(inputImage)
                self.predictedEmotion = self.emotions[np.argmax(predictions[0])]

                if draw:
                    frame = self.drawBoundingBox(frame)  # creates a bounding box in the face area
                    cv.putText(frame, self.predictedEmotion, (int(x), int(y) - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (255, 255, 0), 2)
            else:  # if the sum of all elements is 0, then a face has not been detected
                print("Unable to detect a face")
                cv.putText(frame, "Unable to detect a face", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        return frame, extractedFrame

    # helper function for drawing the bounding box around the face
    def drawBoundingBox(self, frame, length=30, thickness=5, rectThickness=1):
        grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # converts the image to grayscale
        facesDetected = self.faceHaarCascade.detectMultiScale(grayImg, 1.32,
                                                              5)  # detects the faces in the input image

        for (x, y, w, h) in facesDetected:  # looping through the dimensions of the bounding box
            cv.rectangle(frame, (x, y, w, h), (255, 255, 0), rectThickness)
            # Top Left
            cv.line(frame, (x, y), (x + length, y), (255, 255, 0), thickness)
            cv.line(frame, (x, y), (x, y + length), (255, 255, 0), thickness)

            # Top Right
            cv.line(frame, ((x + w), y), ((x + w) - length, y), (255, 255, 0), thickness)
            cv.line(frame, ((x + w), y), ((x + w), y + length), (255, 255, 0), thickness)

            # Bottom Left
            cv.line(frame, (x, (y + h)), (x + length, (y + h)), (255, 255, 0), thickness)
            cv.line(frame, (x, (y + h)), (x, (y + h) - length), (255, 255, 0), thickness)

            # Bottom Right
            cv.line(frame, ((x + w), (y + h)), ((x + w) - length, (y + h)), (255, 255, 0), thickness)
            cv.line(frame, ((x + w), (y + h)), ((x + w), (y + h) - length), (255, 255, 0), thickness)
        return frame
