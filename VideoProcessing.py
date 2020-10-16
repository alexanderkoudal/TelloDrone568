import cv2
import numpy as np

cap = cv2.VideoCapture('vid.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    #Resize image
    scale_percent = 40  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    #Center Point
    center_coordinates = (384,215)
    radius = 5
    color = (0,0,255)
    thickness = -1
    cv2.circle(resized,center_coordinates,radius,color,thickness)
    #cv2.imshow('original', resized)
    
    #GaussianBlur
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    #cv2.imshow('Gaussian Blur', blur)

    #HSV
    HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV', HSV)

    #Mask
    light_orange = (0, 0, 0)
    dark_orange = (110, 110, 100)
    mask = cv2.inRange(HSV, light_orange, dark_orange)
    #cv2.imshow('mask', mask)

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
    kernel = np.ones((15,15),np.uint8)
    opening = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening', opening)

    kernel1 = np.ones((3,19),np.uint8)
    erosion = cv2.morphologyEx(vertical, cv2.MORPH_ERODE, kernel1)
    #cv2.imshow('erosion', erosion)

    #New portion of code for BLOB detection and line drawing
    kernel2 = np.ones((55,35),np.uint8)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel2)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        dilation2 = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

        #Circles and Lines for dilation
        cv2.circle(dilation2,center_coordinates,radius,(0,0,255),thickness)
        cv2.drawContours(dilation2, contours, -1, (0, 255, 0), 3) 
        cv2.circle(dilation2, (cX, cY), 5, (255, 0, 0), -1)
        cv2.arrowedLine(dilation2, center_coordinates, (cX, cY), (0,0,255), 1)

        #Circles and Lines for Original
        cv2.circle(resized,center_coordinates,radius,(0,0,255),thickness)
        cv2.drawContours(resized, contours, -1, (0, 255, 0), 3) 
        cv2.circle(resized, (cX, cY), 5, (255, 0, 0), -1)
        cv2.arrowedLine(resized, center_coordinates, (cX, cY), (0,0,255), 1)

        cv2.imshow('Original', resized)
        cv2.imshow('Final', dilation2)

    cv2.waitKey(20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()