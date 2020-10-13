import cv2
import numpy as np

cap = cv2.VideoCapture('vid.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    scale_percent = 40  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('original', resized)
    #GaussianBlur
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    #cv2.imshow('Gaussian Blur', blur)

    #HSV
    HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV', HSV)

    #Mask
    light_orange = (0, 0, 0)
    dark_orange = (100, 100, 70)
    mask = cv2.inRange(HSV, light_orange, dark_orange)
    cv2.imshow('mask', mask)

    # Specify size on vertical axis
    vertical = np.copy(mask)
    rows = vertical.shape[0]
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    #cv2.imshow("vertical", vertical)

    #sobelx = cv2.Sobel(vertical,cv2.CV_8U,1,0,ksize=3)
    #cv2.imshow('Sobel', sobelx)

    #Opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening', opening)

    cv2.waitKey(20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()