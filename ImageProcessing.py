import cv2
import numpy as np


def WebCam():

    cap = cv2.VideoCapture(2)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def Image():
    path = r'/home/artur/Desktop/TestPics/1.jpg'
    image = cv2.imread(path) # reads image 'opencv-logo.png' as grayscale
    window_name = 'Image'
    ksize = (10,10)

    img = cv2.medianBlur(image,55)
    cv2.imshow('Test', img)
    cv2.waitKey()

    # new_image = np.zeros(image.shape, image.dtype)
    # alpha = 1.0
    # beta = 0

    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)




  




Image()