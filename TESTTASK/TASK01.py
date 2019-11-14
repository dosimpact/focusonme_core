import cv2
import numpy as np


input_img = cv2.imread("input/input01.png", cv2.IMREAD_COLOR)
cv2.imwrite('output/input_img.png',input_img)
cv2.imshow('input_img', input_img)
cv2.waitKey()

input_img = cv2.imread("input/input01.png", cv2.IMREAD_REDUCED_COLOR_8)
cv2.imwrite('output/input_img.png',input_img)
cv2.imshow('input_img', input_img)
cv2.waitKey()

input_img = cv2.imread("input/input01.png", cv2.IMREAD_REDUCED_COLOR_4)
cv2.imwrite('output/input_img.png',input_img)
cv2.imshow('input_img', input_img)
cv2.waitKey()


input_img = cv2.imread("input/input01.png", cv2.IMREAD_REDUCED_COLOR_2)
cv2.imwrite('output/input_img.png',input_img)
cv2.imshow('input_img', input_img)
cv2.waitKey()
