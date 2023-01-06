import cv2
import numpy as np
import math

cap = cv2.VideoCapture('./test.mp4')

while True:
    retval, src = cap.read()
    if not retval:
        break


    src = cv2.transpose(src)
    src = cv2.flip(src, 1)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    lower1 = (170, 100, 0)
    upper1 = (180, 170, 255)
    dst = cv2.inRange(hsv, lower1, upper1)

    kernelM = np.ones((33, 33), np.uint8)
    morph = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernelM)
    kernelD = np.ones((11, 11), np.uint8)
    dilate = cv2.dilate(morph, kernelD, iterations=5)
    kernelE = np.ones((11, 11), np.uint8)
    erode = cv2.erode(dilate, kernelE, iterations=4)

    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        size = len(cnt)

        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        size = len(approx)

        if cv2.isContourConvex(approx):
            if size == 3:
                x1 = abs(approx[0][0][0] - approx[2][0][0])
                y1 = abs(approx[0][0][1] - approx[2][0][1])
                x2 = abs(approx[1][0][0] - approx[2][0][0])
                y2 = abs(approx[1][0][1] - approx[2][0][1])
                theta = abs(math.atan(y1 / x1) - math.atan(y2 / x2))
                angle = theta * 180.0 / np.pi
                if angle > 55 and angle < 65:
                    cv2.line(src, tuple(approx[0][0]), tuple(approx[size - 1][0]), (0, 255, 0), 3)
                    for k in range(size - 1):
                        cv2.line(src, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 0), 3)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', src)
        cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
        cv2.imshow('dst', dst)

    key = cv2.waitKey(25)
    if key == 27:  # Esc
        break
    if key == 26:
        cv2.imwrite("./cap.jpg", src)
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()