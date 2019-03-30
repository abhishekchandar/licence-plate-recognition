# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random
import matplotlib.pyplot as plt
import Preprocess
import DetectChars
from PIL import Image
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

###################################################################################################


def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    if Main.showSteps == True:  # show steps #######################################################
        #cv2.imshow("0", imgOriginalScene)
        Image.fromarray(imgOriginalScene).show()
        input('Press any key to continue...')

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(
        imgOriginalScene)         # preprocess to get grayscale and threshold images

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    # Here we get a list of all the contours in the image that may be characters.
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps == True:  # show steps #######################################################
        #print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        Image.fromarray(imgOriginalScene).show()
        input('Press any key to continue...')
        # This is for the boxing of all the contours
        """
        for possibleChar in listOfPossibleCharsInScene:
            cv2.rectangle(imgContours,(possibleChar.intBoundingRectX,possibleChar.intBoundingRectY),(possibleChar.intBoundingRectX+possibleChar.intBoundingRectWidth,possibleChar.intBoundingRectY+possibleChar.intBoundingRectHeight),(0.0, 255.0, 255.0),1)
            cv2.imshow('PossiblePlate',imgContours)
            cv2.waitKey(0)

        """
        # given a list of all possible chars, find groups of matching chars
        # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(
        listOfPossibleCharsInScene)
    if Main.showSteps == True:  # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " +
              str(len(listOfListsOfMatchingCharsInScene)))    # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            #imgContours2 = np.zeros((height, width, 3), np.uint8)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            #cv2.drawContours(imgContours, contours, -1, (255, 255, 255))
            cv2.drawContours(imgContours, contours, -1,
                             (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

            #imgContours = Image.fromarray(imgContours,'RGB').show()

    # end if # show steps #########################################################################
    # for each group of matching chars
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        # attempt to extract plate
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:                          # if plate was found
            # add to list of possible plates
            listOfPossiblePlates.append(possiblePlate)

    if Main.showSteps == True:
        print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")
    if Main.showSteps == True:  # show steps #######################################################
        print("\n")

        Image.fromarray(imgContours, 'RGB').show()
        input('Press any key to continue...')
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(
                listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(
                p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(
                p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(
                p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(
                p2fRectPoints[0]), Main.SCALAR_RED, 2)

            #cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) +
                  ", click on any image and press a key to continue . . .")
            # Image.fromarray(listOfPossiblePlates[i].imgPlate,'RGB').show()

        # end for
        print("\nplate detection complete, press a key to begin char recognition . . .\n")
        input()
    # end if # show steps #########################################################################

    return listOfPossiblePlates

###################################################################################################


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()
    #print('Now we start to find the contours in the thresholded image that may be characters:')

    contours, hierarchy = cv2.findContours(
        imgThreshCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        if Main.showSteps == True:  # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_YELLOW)
            # Image.fromarray(imgContours,'RGB').show()

        # Here we calculate the x,y,w,h,flatdiagonalsize,aspedctratio,area and (x,y) of the center of the rectangle that is bounding the contour.
        possibleChar = PossibleChar.PossibleChar(contours[i])

        # if contour is a possible char, note this does not compare to other chars (yet) . . .
        if DetectChars.checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + \
                1           # increment count of possible chars
            # and add to list of possible chars
            listOfPossibleChars.append(possibleChar)
            #cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
            #print('This contour may be a character :')
        # else:
            #print('This contour is not a character :')
        # end if
    # end for

    if Main.showSteps == True:  # show steps #######################################################
        print("\nstep 2 - Total number of contours found in the image are = " +
              str(len(contours)))
        print("step 2 - number of contours those may be characters = " +
              str(intCountOfPossibleChars))
        #print("These are the contours those may be characters :")
        Image.fromarray(imgContours, 'RGB').show()
    # end if # show steps #########################################################################

    return listOfPossibleChars

###################################################################################################


def extractPlate(imgOriginal, listOfMatchingChars):
    # this will be the return value
    possiblePlate = PossiblePlate.PossiblePlate()

    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(
        listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(
        listOfMatchingChars) - 1].intCenterY) / 2.0
    # This is the probable centeral point of this plate.
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(
        listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    # Here we calculate the probable width of this plate.
    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    # Here we calculate the probale height of this particular plate.
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    # We include the padding factor.
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(
        listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(
        listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (
        tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    # The first poin tis the point of rotaion or center,theta and scaling factor
    rotationMatrix = cv2.getRotationMatrix2D(
        tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    # unpack original image width and height
    height, width, numChannels = imgOriginal.shape

    # rotate the entire image
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(
        ptPlateCenter))  # We extract the probable plate from the Original image

    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.imgPlate = imgCropped

    return possiblePlate
