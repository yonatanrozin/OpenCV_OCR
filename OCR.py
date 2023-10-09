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

def skew_correction(image, non_processed):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    corrected = cv2.warpAffine(non_processed, M, (w, h), flags=cv2.INTER_CUBIC, #thresh should be image
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def preprocess_frame(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5, 4) 
    img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    img = cv2.dilate(img, kernel, iterations=1)

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

            processed = preprocess_frame(frame)

            contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            highlights = frame.copy() # to draw rectangles without interfering with CV

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)

                if w < 50:
                    continue

                x -= 10
                y -= 10
                w += 20
                h += 20

                angleRect = cv2.minAreaRect(c)

                # cv2.rectangle(highlights, (x, y), (x+w, y+h), (255, 0, 0), 2)
                try:
                    cropped = frame[y:y+h,x:x+w]
                    highlights[y:y+h,x:x+w] = cv2.cvtColor(processed[y:y+h,x:x+w], cv2.COLOR_GRAY2BGR)
                    cv2.putText(highlights, str(int(angleRect[2])), (x-30, y-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

                except:
                    print('cant')

            cv2.imshow('view', highlights)
            cv2.waitKey(10)


            # data = pytesseract.image_to_data(processed, lang="eng_best", config='--psm 11').split('\n')[:-1] #ignore last item

            # data_labels = data[0].split('\t')
            # data_entries = [row.split('\t') for row in data[1:]]

            # for entry in data_entries:
            #     data_parsed = dict(zip(data_labels, entry))
            #     if float(data_parsed['conf']) < 50:
            #         continue
            #     print(entry, float(data_parsed['conf']))




            continue
            textAreas = getPotentialTextAreas(frame, processed)

            for area in textAreas:
                grey = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

                # grey = cv2.bitwise_not(grey)

                # area_thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5, 4) 
                ret, area_thresh = cv2.threshold(grey,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                area_text = pytesseract.image_to_string(processed, config='--psm 3')
                print(area_text)

                cv2.imshow('test', area_thresh)
                if cv2.waitKey(500) == ord('q'):
                    cv2.destroyAllWindows()
                    exit()

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