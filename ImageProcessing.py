import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb
import pylab as p

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
    path = r'/home/artur/Desktop/TestPics/tree.jpg'
    image = cv2.imread(path)  # reads image 'opencv-logo.png' as grayscale
    window_name = 'Image'
    scale_percent = 15  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Original', resized)
    cv2.waitKey(0)

     #Contrast and Brightness
    CandB = np.zeros(image.shape, image.dtype)
    alpha = 1
    beta = 3
    CandB = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)
    cv2.imshow('Constrast and Brightness', CandB)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #Convert the image to HSV
    HSV = cv2.cvtColor(CandB, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', HSV)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #Thresholding the tree trunks
    # pixel_colors = HSV.reshape((np.shape(HSV)[0]*np.shape(HSV)[1], 3))
    # norm = colors.Normalize(vmin=-1.,vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()

    # h, s, v = cv2.split(HSV)
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")

    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # plt.show()

    #light_orange = (25, 0, 0)
    #dark_orange = (75, 70, 50)

    light_orange = (0, 0, 0)
    dark_orange = (90, 90, 60)

    # lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
    # do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

    # plt.subplot(1, 2, 1)
    # plt.imshow(hsv_to_rgb(do_square))
    # plt.subplot(1, 2, 2)
    # plt.imshow(hsv_to_rgb(lo_square))
    # plt.show()

    mask = cv2.inRange(HSV, light_orange, dark_orange)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    horizontal = np.copy(mask)
    vertical = np.copy(mask)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    cv2.imshow("horizontal", horizontal)
    cv2.waitKey(0)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 20
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    cv2.imshow("vertical", vertical)
    cv2.waitKey(0)

    final = vertical - horizontal
    cv2.imshow("final", final)
    cv2.waitKey(0)















    # # Grayscale Image
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # #Contrast and Brightness
    # CandB = np.zeros(image.shape, image.dtype)
    # alpha = 2
    # beta = 2
    # CandB = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    # cv2.imshow('Constrast and Brightness', CandB)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # Median Blur
    # #ksize = (10,10)
    # blur = cv2.GaussianBlur(CandB, (7, 7), 0)
    # cv2.imshow('Gaussian Blur', blur)
    # cv2.waitKey(0)

    # #Sobel
    # sobelx = cv2.Sobel(blur,cv2.CV_8U,1,0,ksize=3)
    # cv2.imshow('Sobel', sobelx)
    # cv2.waitKey(0)

    # # DIlation
    # #kernel = np.ones((3,3),np.uint8)
    # #dilation = cv2.dilate(sobelx,kernel,iterations = 1)
    # #cv2.imshow('Dilation', dilation)
    # # cv2.waitKey(0)

    # # Canny Edge Detection
    # edges = cv2.Canny(sobelx, 125, 150, None, 3)
    # cv2.imshow('Canny', edges)
    # cv2.waitKey(0)

    # canny2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # #Hough Transform
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(canny2, pt1, pt2, (0,0,0), 1, cv2.LINE_AA)
    # cv2.imshow('lines', canny2)
    # cv2.waitKey(0)
    # # Adaptive Thresholding
    # # th = np.zeros(resized.shape, resized.dtype)
    # # ret, th = cv2.threshold(blur,150,200,cv2.THRESH_BINARY)
    # # cv2.imshow('Thresholding', th)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()


Image()
