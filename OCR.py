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

margin = 10 # margin width (in px) between text areas / image borders

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
        if rect[1][0] < 20:
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

def assemble_text_scan_image(img, areas):

    xOff = margin
    yOff = margin
    rowXMax = 0

    #single blank image to store potential text areas for efficient OCR
    totalScan = np.zeros([1000,1000],dtype=np.uint8) 
    totalScan.fill(255) # or img[:] = 255

    #parallel to scan but with colors - for use in displaying results
    scan_colors = np.ones([1000,1000, 3], dtype=np.uint8) * 255

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

    return totalScan, scan_colors

def text_data_from_image(img):
    data = pytesseract.image_to_data(img, lang="eng_best", config='--psm 12').split('\n')[:-1] #ignore last item
    data_labels = data[0].split('\t')
    data_values = [row.split('\t') for row in data[1:]]
    data_entries = [dict(zip(data_labels, entry)) for entry in data_values]

    return data_entries

def overlay_results(img, data, scan_colors):
    resultYOff = margin
    for row in data:
        if float(row['conf']) < 50 or len(row['text']) < 3: 
            continue
        print(row)
        x, y, w, h = (
            int(float(row['left'])),
            int(float(row['top'])),
            int(float(row['width'])),
            int(float(row['height']))
        )
        if 0 in (w, h):
            print("gotcha!")
            continue
        img[resultYOff:resultYOff+h, margin:margin+w] = scan_colors[y:y+h, x:x+w]
        cv2.putText(img, row['text'], (margin + w + 30, resultYOff+h), cv2.FONT_HERSHEY_PLAIN, h/10, (0,255,0), h//10)
        resultYOff += margin + h

    return img

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

            #detect areas containing long contours that may be text
            areas = get_potential_text_areas(frame)

            #get single image containing text areas to be scanned and parallel colored version for visuals
            scan_image, colors = assemble_text_scan_image(frame, areas)

            #get a list of dict objects containing information per text detected in image
            text_results = text_data_from_image(scan_image)

            #overlay detected text and extracted string side-by-side on top of captured frame
            output_image = overlay_results(frame, text_results, colors)
            
            cv2.imshow('test', output_image)
            cv2.waitKey(50)


                
main(argv[1:])