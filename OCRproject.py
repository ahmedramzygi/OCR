import pytesseract as ocr
import cv2 as cv
import imutils
import numpy as np
import re
height = 900
width  = 900

img1 = cv.imread('input3.png') # we read the image
#functions to be used in our algorithm 
def printImages():
    #I resized every variable so that i can print it aside
    imgProcessedFit=cv.resize(imgProcessed,(380,380))
    contorsFit=cv.resize(contors,(380,380))
    ImagePerspectedFit=cv.resize(ImagePerspected,(380,380))
    imgWarpWarpFit=cv.resize(ImagePerspectedGrey,(380,380))
    cv.imshow('Canny Edges before Contouring',imgProcessedFit) 
    cv.imshow('ImageContours',contorsFit)
    cv.imshow('The wrapped perspective',ImagePerspectedFit)
    cv.imshow('The gray image ',imgWarpWarpFit)
   # I rescale our input image to be of width 800
def imageRescaling(img):
    img = imutils.resize(img, width=width-100)
    return img
    # Here i have my input image being preprocessed to be ready for contoring phase by detecting all images using canny detector
def imagePreProcessing(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #My image is changed into grey
    
    blur = cv.GaussianBlur(grey, (5,5), 0) # We smoothed the grey image by a 5*5 gaussian Mask
    b=cv.resize(blur,(380,380))
    cv.imshow('Blureed image',b)
    edged = cv.Canny(blur, 75, 200) # Here we detected the edges using canny by giving 2 threshold values
    
    return edged
    #Contor Manipulation function is used to get all the edged image contours and also the largest one 
def ContourManipulation(imageProcessed):
    imgContours = img.copy() #We get a copy of the original image as find cotour function rewrites on the image
    contours, hierarchy = cv.findContours(imageProcessed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #RETR_TREE Where we compute the relationship between contours and CHAIN_APPROX_SIMPLE.
    cv.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # I drawed all the contours on the imgContours variables
    contours=sorted(contours,key=cv.contourArea,reverse=True)[:5]# Then i sort them
    Largest=np.array([]) 
    for i in contours:
        perimeter = cv.arcLength(i, True)
        approximation = cv.approxPolyDP(i, 0.02 * perimeter, True)
        if len(approximation) == 4:
            Largest = approximation
            break
        else:
            return img1,imgContours
    return Largest,imgContours #return the largest contour and all contours image
    # This function is used to get the biggest contour points and reorderd them by a specific manner Where:-
def reorderAndDraw(biggest):
    biggest = biggest.reshape((4, 2))
    PointsOrdered = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    PointsOrdered[0] = biggest[np.argmin(add)] #Here point 0 has the minimum sum
    PointsOrdered[3] =biggest[np.argmax(add)] #Here point 3 has the minimum sum
    diff = np.diff(biggest, axis=1)
    PointsOrdered[1] =biggest[np.argmin(diff)] #Here point 1 has the minimum diffrence
    PointsOrdered[2] = biggest[np.argmax(diff)] #Here point 2 has the biggest diffrence
    return PointsOrdered
    # This function is used to get the largest contour and wrap into to be in readable transform perspective

def ImagePerspective(largestPoints):
    if largestPoints.size != 0 and largestPoints.size==8 : 
        largestPoints=reorderAndDraw(largestPoints)
        Matrix_1 = np.float32(largestPoints) 
        Matrix_2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
        matrix = cv.getPerspectiveTransform(Matrix_1, Matrix_2)
        ImagePerspected = cv.warpPerspective(img, matrix, (width, height))
        return ImagePerspected   
    else:
        return largestPoints
# It is the last function that takes the processed image and change it to string wriiten in the OCR.txt
def imgToString(img):
    data = ocr.image_to_string(ImagePerspectedGrey)
    with open('Ocr.txt', mode = 'w') as f:
        f.write(data)
        print("The number of charachters in this text file is ")
        print (len(re.sub(r"\W","",data))) # Here we used the regular expression module to count the data string charachters after transforming from the image and print it to the terminal

# Here our algorithm starts

img=imageRescaling(img1) # I Rescaled the input image


imgProcessed=imagePreProcessing(img) #  pre-processed the input image

ProcessedCopy=imgProcessed.copy() # I took a copy from the pre-processed to not affect the original image

biggest,contors=ContourManipulation(ProcessedCopy)#Here we recieve biggest contor and all contours in the image

ImagePerspected=ImagePerspective(biggest) # Here we apply our easy prespective for human for the not well taken photos

ImagePerspectedGrey=  cv.cvtColor(ImagePerspected, cv.COLOR_BGR2GRAY) #We transformed the colored image into gray-level
# These are postprocess functions for enhancement of the image depending on the quality of the image but it is not udes in the given samples
# ret,Thresh= cv.threshold(ImagePerspectedGrey,145,255,cv.THRESH_BINARY)
# ret,greyThresh = cv.threshold(grey, 100, 255, cv.THRESH_BINARY)
# blur=cv.medianBlur(greyThresh,3)

imgToString(ImagePerspectedGrey) #We send our transformed image to get written to the OCR.txt

printImages()# We printed the images of every phase

cv.waitKey(0)

cv.destroyAllWindows()

