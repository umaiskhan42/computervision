import cv2 as cv
import numpy as np

img=cv.imread('Files/images/cat1.jpg')
cv.imshow('Cat',img)

blank= np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

# convert to greyscale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# canny egdes
canny=cv.Canny(img,125,175)
cv.imshow('Canny Edges', canny)

# find contours method
contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours(s) found ')

# draw contours

cv.drawContours(blank, contours, -1, (0,0,255), 2)
cv.imshow('Contoursdrawn', blank)
# contours- All the points in egdes of image
# heirarchies- to find contours
# Retr-list--Retrieves contours
# chainapprox/none/simple- returns all contours/points in a line
cv.waitKey(0)
