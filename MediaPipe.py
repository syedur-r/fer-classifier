import cv2 as cv
import mediapipe as mp
import numpy as np
from keras.preprocessing import image


# helper function for drawing the bounding box around the face
def drawBoundingBox(frame, boundingBox, length=30, thickness=5, rectThickness=1):
    x, y, w, h = boundingBox

    cv.rectangle(frame, boundingBox, (255, 255, 0), rectThickness)
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


class FaceDetector:
    # constructor function with global variables
    def __init__(self, minDetectionConfidence=0.75, predictedEmotion=""):
        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        # @param: you can use this parameter to change the confidence of detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)
        self.predictedEmotion = predictedEmotion
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
    def detectEmotion(self, model, frame, draw=True, padding=0):
        imgColor = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgColor)
        height, width, _ = frame.shape
        extractedFrame = None

        # extracting the face from the input and appending it onto an array
        if results.detections:
            faces = []
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box  # getting the coordinates of the face
                # computing the bounding box
                boundingBox = int(box.xmin * width), int(box.ymin * height), int(
                    box.width * width), int(
                    box.height * height)

                x, y, w, h = boundingBox  # getting the coordinates of the bounding box
                face = frame[y - padding:y + h + padding, x - padding:x + w + padding]  # cropping face area from image
                extractedFrame = face
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # converting the image to grayscale
                faces.append(face)  # appending the image onto a list of images containing multiple faces

                if faces == 0:  # error handling: checks if there are no faces in the faces array
                    break
                else:
                    for face in faces:
                        if np.sum([face]) != 0:  # checks if the sum of all elements is not equal to 0
                            inputFace = cv.resize(face, (48, 48))  # resizes the region of interest to the
                            # input shape of the model - (48, 48, 1)
                            inputImage = image.img_to_array(inputFace)  # converting the region of interest to an
                            # image array
                            inputImage = np.expand_dims(inputImage, axis=0)  # expanding the dimensions
                            inputImage /= 255  # normalising the input image

                            # the model is predicting the emotions of the input image, storing it in a variable with
                            # its index and outputting it in text
                            predictions = model.predict(inputImage)
                            self.predictedEmotion = self.emotions[np.argmax(predictions[0])]

                            # drawing a bounding box around all the faces
                            for i in range(len(faces)):
                                if draw:
                                    frame = drawBoundingBox(frame, boundingBox)
                                    cv.putText(frame, f"{self.predictedEmotion}",
                                               (boundingBox[0], boundingBox[1] - 20),
                                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        else:  # if the sum of all elements is 0, then a face has not been detected
                            print("Unable to detect a face")
        return frame, extractedFrame
