import cv2 as cv
import numpy as np


def preprocessing(image):
    image = cv.medianBlur(image, 3)
    kernel = np.ones((5, 5), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image, kernel, iterations=3)
    return image


def order_points(corner):
    corner = corner.reshape(-1, 2)
    order = np.zeros(4).astype(np.uint8)

    summ = np.sum(corner, axis=1)
    order[0] = np.argmin(summ)
    order[3] = np.argmax(summ)

    diff = np.diff(corner, axis=1)
    order[1] = np.argmin(diff)
    order[2] = np.argmax(diff)

    return corner[order]
