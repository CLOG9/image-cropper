# import cv2
# import numpy as np
# import utlis


# ########################################################################
# webCamFeed = True
# pathImage = "./0.jpg"


# heightImg = 640
# widthImg  = 480
# ########################################################################

# utlis.initializeTrackbars()
# count=0

# while True:

#     img = cv2.imread(pathImage)
#     img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
#     imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
#     imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
#     thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
#     imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
#     kernel = np.ones((5, 5))
#     imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
#     imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

#     ## FIND ALL COUNTOURS
#     imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
#     imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
#     contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
#     cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


#     # FIND THE BIGGEST COUNTOUR
#     biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
#     if biggest.size != 0:
#         biggest=utlis.reorder(biggest)
#         cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
#         imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
#         pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
#         pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
#         imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

#         #REMOVE 20 PIXELS FORM EACH SIDE
#         imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
#         imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

#         # APPLY ADAPTIVE THRESHOLD
#         imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
#         imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
#         imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
#         imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

#         # Image Array for Display
#         imageArray = ([img,imgGray,imgThreshold,imgContours],
#                       [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

#     else:
#         imageArray = ([img,imgGray,imgThreshold,imgContours],
#                       [imgBlank, imgBlank, imgBlank, imgBlank])

#     # LABELS FOR DISPLAY
#     lables = [["Original","Gray","Threshold","Contours"],
#               ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

#     stackedImage = utlis.stackImages(imageArray,0.75,lables)
#     cv2.imshow("Result",stackedImage)

#     # SAVE IMAGE WHEN 's' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite(str(count)+".jpg",imgWarpColored)
#         cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
#                       (1100, 350), (0, 255, 0), cv2.FILLED)
#         cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
#                     cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
#         cv2.imshow('Result', stackedImage)
#         cv2.waitKey(300)
#         count += 1





import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

# cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

pathImage = "./0.jpg"

cap =cv2.imread(pathImage)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

count = 0
scale = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX

WIDTH, HEIGHT = 480, 640
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap = cv2.resize(cap, (WIDTH, HEIGHT)) # RESIZE IMAGE


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)


def center_text(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)


while True:

    # _, frame = cap.read()
    frame = cv2.rotate(cap, cv2.ROTATE_180)
    frame_copy = frame.copy()

    scan_detection(frame_copy)

    cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

    processed = image_processing(warped)
    processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
    cv2.imshow("Processed", cv2.resize(processed, (int(scale * processed.shape[1]),
                                                   int(scale * processed.shape[0]))))

    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 27:
        break

    elif pressed_key == ord('s'):
        cv2.imwrite("scanned_" + str(count) + ".jpg", processed)
        count += 1

        center_text(frame, "Scan Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(500)

    elif pressed_key == ord('o'):
        file = open("recognized_" + str(count - 1) + ".txt", "w")
        ocr_text = pytesseract.image_to_string(warped)
        # print(ocr_text)
        file.write(ocr_text)
        file.close()

        center_text(frame, "Text Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(500)

cv2.destroyAllWindows()