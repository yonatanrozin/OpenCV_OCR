print("---\nPython Webcam OCR\nYonatan Rozin\n---")
print('Importing modules...')

import subprocess as sp
import cv2
import pytesseract
import numpy as np
from sys import argv

from scipy.ndimage import rotate

def skew_correction(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except: 
        print("caught!")
        return None, None
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    angles = range(-30,30, 2)
    scores = []
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, #thresh should be image
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def preprocess_frame(img):

    output_image = np.ones(img.shape, 'uint8') * 255

    orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5, 4) 
    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    img = cv2.dilate(img, kernel, iterations=1)

    kept_contours = []

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        
        c = contours[i]
        x,y,w,h = cv2.boundingRect(c)
        if w < h or w * h < 500: #find horizontal contours only + ignore small ones
            continue
        
        x-=10
        y-=10
        w+=20
        h+=20

        enclosed = False
        for c_check in kept_contours:
            if x > c_check[0] and x+w < c_check[0]+c_check[3] and y > c_check[1] and y+h < c_check[1] + c_check[3]:
                enclosed = True
                break
        if not enclosed:
            kept_contours.append((x, y, w, h))

            cropped = orig[y:y+h, x:x+w]
            output_image[y:y+h, x:x+w] = cropped

            angle, deskewed = skew_correction(cropped)

            if deskewed is not None:
                cv2.imshow('test', deskewed)
                cv2.waitKey(50)



    
    return output_image

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

    cam_index = 0
    # cam = cv2.VideoCapture(cam_index)

    capture_api = cv2.CAP_DSHOW
    cam = cv2.VideoCapture(cam_index, capture_api)

    pytesseract.pytesseract.tesseract_cmd = r'C:\dev\Tesseract-OCR\tesseract.exe'

    cam_read_attempts = 0

    while True:

        cam_read_attempts += 1

        print(f"Getting new frame - attempt #{cam_read_attempts}")
        success, frame = cam.read()
        
        if not success:
            if cam_read_attempts < 5:
                print(f"unsuccessful camera read. Retrying.")
            else:
                print("\nError reading from camera. Try a different camera index or a backend API such as 'cv2.CAP_MSMF' or 'cv2.CAP_DSHOW'")
                print('Run "OCR.py list" to get list of available devices')
                print("See OpenCV docs here for info on video capture APIs: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d")
                exit()

        if success:
            cam_read_attempts = 0

            processed = preprocess_frame(frame)

            continue

            output_frame = processed.copy()

            data = pytesseract.image_to_data(processed, config='--psm 11').split('\n')[:-1] #ignore last item

            data_labels = data[0].split('\t')
            data_entries = [row.split('\t') for row in data[1:]]

            cv2.imshow('test', output_frame)
            cv2.waitKey(50)

            continue

            if not data:
                continue
            for entry in data_entries:
                data_parsed = dict(zip(data_labels, entry))
                if float(data_parsed['conf']) < 50:
                    continue

                (x, y, w, h) = (
                    int(float(data_parsed['left'])), int(float(data_parsed['top'])),
                    int(float(data_parsed['width'])), int(float(data_parsed['height'])))
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255,0,0), 1)
                cv2.putText(output_frame, data_parsed['text'], (x, y-10), cv2.FONT_HERSHEY_PLAIN, .5, (0,0,255))

                
main(argv[1:])