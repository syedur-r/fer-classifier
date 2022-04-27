import cv2 as cv
import MediaPipe
import HaarCascade
import Images
import Videos
from keras.models import model_from_json


def main():
    model = model_from_json(open("models/final-model.json", "r").read())  # load model
    model.load_weights("models/final-model.h5")  # load weights
    imgIndex = 0  # image counter to keep track of the number of frames saved

    # COMMENT/UNCOMMENT to use image/video or webcam
    # ////////////////////////////////////////////////////////////////////////////////////
    # cap = cv.VideoCapture(Images.angry_1)  # Images
    # cap = cv.VideoCapture(Videos.angry_1)  # Videos
    cap = cv.VideoCapture(0)  # Webcam
    # ////////////////////////////////////////////////////////////////////////////////////

    if not cap.isOpened():  # checks if the video capture didn't open
        exit()  # exits from the running code

    while True:
        _, frame = cap.read()  # captures frame and returns boolean value and captured image
        if not _:
            continue

        # COMMENT/UNCOMMENT the face detection classifier you wish to use
        # ////////////////////////////////////////////////////////////////////////////////////
        classifier = MediaPipe.FaceDetector()  # Media Pipe Face Detection
        frame, extractFace = classifier.detectEmotion(model, frame, padding=0)  # Change padding to see different
        # results e.g. 20, 50

        # classifier = HaarCascade.FaceDetector()  # Haarcascade Face Detection
        # frame, extractedFrame = classifier.detectEmotion(model, frame)
        # ////////////////////////////////////////////////////////////////////////////////////

        # creating a window to load the input image/video/live webcam feed
        title = "FER Classifier"
        cv.namedWindow(title)
        cv.setWindowProperty(title, cv.WND_PROP_TOPMOST, 1)
        cv.imshow(title, frame)

        # checks which key has been pressed
        # esc key will quit the program
        # space key will save the frames from the running window
        # q key will also quit the program
        j = cv.waitKey(1)
        if j % 256 == 27:  # esc
            break
        elif j % 256 == 32:  # space
            # saves the entire frame
            cv.imwrite("saved-frames/" + str(classifier.predictedEmotion) + "-" + str(imgIndex) + ".jpg", frame)
            # saves the extracted face from the bounding box
            # cv.imwrite("saved-frames/" + str(classifier.predictedEmotion) + "-" + str(imgIndex) + ".jpg", extractFace)
            imgIndex += 1
        elif j % 256 == 113:  # q
            exit()

    cap.release()  # closes the frame
    cv.destroyAllWindows()  # closes all running windows


if __name__ == "__main__":
    main()

main()  # invokes the main method
