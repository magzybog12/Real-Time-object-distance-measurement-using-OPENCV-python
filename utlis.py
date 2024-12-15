import cv2
import numpy as np

def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Blur
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])  # Detect edges
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)  # Dilation
    imgThre = cv2.erode(imgDial, kernel, iterations=2)  # Erosion

    if showCanny:
        cv2.imshow('Canny', imgThre)

    # Find contours
    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:  # Filter based on area
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            bbox = cv2.boundingRect(approx)

            # Check if the contour has the desired number of corners (rectangles have 4)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append((len(approx), area, approx, bbox, i))
            else:
                finalContours.append((len(approx), area, approx, bbox, i))

    # Sort contours by area (largest first)
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    if draw:
        for contour in finalContours:
            cv2.drawContours(img, [contour[4]], -1, (0, 0, 255), 3)

    return img, finalContours


def reorder(myPoints):
    if myPoints.shape[0] != 4:
        print("Error: reorder expects exactly 4 points, but got:", myPoints.shape[0])
        return None  # Return None if the input is invalid

    myPointsnew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))

    add = myPoints.sum(1)
    myPointsnew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsnew[3] = myPoints[np.argmax(add)]  # Bottom-right

    diff = np.diff(myPoints, axis=1)
    myPointsnew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsnew[2] = myPoints[np.argmax(diff)]  # Bottom-left

    return myPointsnew


def warpImg(img, points, w, h,pad=20):
    reorderedPoints = reorder(points)
    if reorderedPoints is None:
        return None

    pts1 = np.float32(reorderedPoints)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp =imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]

    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5



