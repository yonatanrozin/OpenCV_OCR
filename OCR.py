print("---\nPython Webcam OCR\nYonatan Rozin\n---")
print('Importing modules...')

import subprocess as sp
import cv2
import pytesseract
import numpy as np
from sys import argv
from scipy.ndimage import rotate

cam_index = 0
capture_api = cv2.CAP_DSHOW

# pre-processing for finding long contours that may be text:
#   blurring for anti noise
#   adaptive thresholding for flexible binarization
#   inverting + horizontal dilation - causes letters in lines of text to blur together into single contours
def preprocess_frame(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5, 4) 
    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

# receives an image, probably from cv2.VideoCapture.read()
#   pre-processes image to prepare for contour detection
#   performs contour detection, finding rotated (min-area) rect for each
#   imposes a slight width threshold on results 
#   returns list of cropped rectangles, aligned horizontally
def get_potential_text_areas(img):

    processed = preprocess_frame(img)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aligned_areas = []
    for contour in contours:

        rect = cv2.minAreaRect(contour)
        if rect[1][0] < 50:
            continue

        center, size, theta = rect

        size = (size[0]*1.2, size[1]*1.2) # include some whitespace

        # align (potential) text using angle of rotated bounding box
        center, size = tuple(map(int, center)), tuple(map(int, size))
        M = cv2.getRotationMatrix2D( center, theta, 1)
        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        area = cv2.getRectSubPix(dst, size, center)
        if theta > 45:
            area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)

        aligned_areas.append(area)
    return aligned_areas

def main(args):

    if 'opencv-python-headless' in str(sp.check_output(['pip', 'list'])):
        print("opencv-python-headless detected. Uninstall it and re-install opencv-python (non-headless).\n")
        exit()

    if 'list' in args:
        print('testing camera indexes to find available devices- ignore camera index errors')
        available_cameras = []

        for i in range(10):
            try:
                cam_test = cv2.VideoCapture(i)
                if cam_test is not None and cam_test.isOpened():
                    available_cameras.append(i)
                cam_test.release()
            except cv2.Error as e:
                print("nope")
                pass

        print("\nAvailable camera indexes: ", available_cameras)
        exit()

    # cam = cv2.VideoCapture(cam_index)
    cam = cv2.VideoCapture(cam_index, capture_api)

    pytesseract.pytesseract.tesseract_cmd = r'C:\dev\Tesseract-OCR\tesseract.exe'

    cam_read_attempts = 0

    margin = 10

    while True:

        cam_read_attempts += 1

        print(f"Getting new frame - attempt #{cam_read_attempts}")
        success, frame = cam.read()
        
        #quit after 5 consecutive failed read attempts
        if not success:
            if cam_read_attempts < 5:
                print(f"unsuccessful camera read. Retrying.")
            else:
                print("\nError reading from camera. Try a different camera index or a backend API such as 'cv2.CAP_MSMF' or 'cv2.CAP_DSHOW' in 'cv2.VideoCapture() above")
                print('Run "OCR.py list" in command line to get list of available video input devices')
                print("See OpenCV docs here for info on video capture APIs: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d")
                exit()

        if success:
            cam_read_attempts = 0

            areas = get_potential_text_areas(frame)

            xOff = margin
            yOff = margin
            rowXMax = 0

            #single blank image to store potential text areas for efficient OCR
            totalScan = np.zeros([1000,1000],dtype=np.uint8) 
            totalScan.fill(255) # or img[:] = 255

            #parallel to scan but with colors - for use in displaying results
            scan_colors = np.zeros([1000,1000, 3], dtype=np.uint8)

            hScan, wScan = totalScan.shape
            for area in areas:

                try:
                    grey = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
                    kernel = np.ones((3, 3), np.uint8) 
                    eroded = cv2.erode(thresh, kernel) 
                except:
                    print('error preprocessing text area - skipping.')
                    continue 

                processed = eroded
                h, w = processed.shape
                
                if xOff + w > wScan:
                    print("ran out of space!")
                    break
                if yOff + h < hScan:
                    totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                    scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                    yOff += h + margin
                    if w > rowXMax:
                        rowXMax = w
                else:
                    xOff = rowXMax + margin
                    yOff = margin
                    totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                    scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                    yOff += h + margin
                    rowXMax = w

            resultYOff = margin

            data = pytesseract.image_to_data(totalScan, lang="eng_best", config='--psm 11').split('\n')[:-1] #ignore last item
            data_labels = data[0].split('\t')
            data_entries = [row.split('\t') for row in data[1:]]
            
            for entry in data_entries:
                data_parsed = dict(zip(data_labels, entry))
                if float(data_parsed['conf']) < 70 or len(data_parsed['text']) < 3: 
                    continue
                print(data_parsed)
                x, y, w, h = (
                    int(float(data_parsed['left'])),
                    int(float(data_parsed['top'])),
                    int(float(data_parsed['width'])),
                    int(float(data_parsed['height']))
                )
                frame[resultYOff:resultYOff+h, margin:margin+w] = scan_colors[y:y+h, x:x+w]
                cv2.putText(frame, data_parsed['text'], (margin + w + 30, resultYOff+h), cv2.FONT_HERSHEY_PLAIN, h/10, (0,255,0), h//10)
                resultYOff += margin + h

            cv2.imshow('test', frame)
            cv2.waitKey(500)

            cv2.imshow('view',scan_colors)
            cv2.waitKey(5)

                
main(argv[1:])