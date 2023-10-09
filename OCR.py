print("---\nPython Webcam OCR\nYonatan Rozin\n---")
print('Importing modules...')

import subprocess as sp
import cv2
import pytesseract
import numpy as np
from sys import argv

cam_index = 0
capture_api = cv2.CAP_DSHOW

margin = 10 # margin width (in px) between text areas / image borders

# receives an image, performs some pre-processing to prepare for finding long contours that may be text:
#   convert to grayscale
#   blur for anti noise
#   adaptive thresholding for flexible binarization
#   [mostly] horizontal dilation - causes letters in lines of text to blur together into single contours
def preprocess_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 5, 4) #binary-inv?
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 7))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

# receives an image, probably from cv2.VideoCapture.read()
#   pre-processes image to prepare for contour detection
#   performs contour detection, finding rotated (min-area) rect for each
#   imposes a slight width threshold on results 
#   returns list of cropped rectangles, aligned horizontally and
#   a parallel list containing central coordinates for each corresponding rectangle
def get_potential_text_areas(img, processed):

    processed_show = processed.copy()

    contours = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    aligned_areas = []
    centroids = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < h:
            continue

        rect = cv2.minAreaRect(contour)

        center, size, theta = rect

        if size[0]*size[1] < 100:
            continue

        size = (size[0]+10, size[1]+10) # include some whitespace

        # align (potential) text using angle of rotated bounding box
        print('rotating contour to align text.')
        center, size = tuple(map(int, center)), tuple(map(int, size))
        M = cv2.getRotationMatrix2D( center, theta, 1)
        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        area = cv2.getRectSubPix(dst, size, center)
        if theta > 45:
            area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)

        if area is not None:
            if area.shape[0] < 100:
                print('enlarging area')
                WH_ratio = area.shape[1]/area.shape[0]
                new_size = (100, int(100 / WH_ratio))
                area = cv2.resize(area, new_size)
            aligned_areas.append(area)
            centroids.append(center)

            box = cv2.boxPoints(rect) 
            box = np.intp(box)
            cv2.drawContours(processed_show,[box],0,(255,255,255),1)

    return aligned_areas, centroids

def assemble_text_scan_image(img, areas, centroids):

    xOff = margin
    yOff = margin

    #single blank image to store potential text areas for efficient OCR
    totalScan = np.zeros([2000,2000],dtype=np.uint8) 
    totalScan.fill(255) # or img[:] = 255

    #parallel to scan but with colors - for use in displaying results
    scan_colors = np.ones([2000,2000, 3], dtype=np.uint8) * 255

    centroid_location_buffer = np.zeros([2000,2000,3], dtype="float32")

    hScan, wScan = totalScan.shape
    for i in range(len(areas)):

        area = areas[i]
        cent = centroids[i]

        processed = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        norm_matrix = np.zeros((img.shape[0], img.shape[1]))
        processed = cv2.normalize(processed, norm_matrix, 0, 255, cv2.NORM_MINMAX)
        
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]

        h, w = processed.shape

        if xOff + w < wScan:
            totalScan[yOff:yOff+h, xOff: xOff+w] = processed
            scan_colors[yOff:yOff+h, xOff: xOff+w] = area
            cv2.rectangle(centroid_location_buffer, (xOff, yOff), (xOff + w, yOff + h), (cent[0]/1000,cent[1]/1000,0), -1)

            cv2.putText(scan_colors, str(cent), (xOff, yOff), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
            xOff += (w + margin)
        else:
            xOff = margin
            yOff += 200 + margin
            totalScan[yOff:yOff+h, xOff: xOff+w] = processed
            scan_colors[yOff:yOff+h, xOff: xOff+w] = area
            cv2.rectangle(centroid_location_buffer, (xOff, yOff), (xOff + w, yOff + h), (cent[0],cent[1],0), -1)
            
            cv2.putText(scan_colors, str(cent), (xOff, yOff), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
            xOff += w + margin

    return totalScan, scan_colors, centroid_location_buffer

def text_data_from_image(img, buffer):
    data = pytesseract.image_to_data(img, lang="eng_best", config='--psm 12').split('\n')[:-1] #ignore last item
    data_labels = data[0].split('\t')
    data_values = [row.split('\t') for row in data[1:]]
    data_entries = [dict(zip(data_labels, entry)) for entry in data_values]
    for row in data_entries:
        row['centerX'] = int(float(row['left']) + float(row['width'])/2)
        row['centerY'] = int(float(row['top']) + float(row['height'])/2)
        row['orig_point'] = buffer[row['centerY'], row['centerX']][:2]*1000
    return data_entries

def overlay_results(img, data, scan, show_processed=False):
    if show_processed:
        scan = cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)
    resultYOff = margin
    for row in data:
        x, y, w, h = (
            int(float(row['left'])),
            int(float(row['top'])),
            int(float(row['width'])),
            int(float(row['height']))
        )

        if 0 in (w, h):
            continue
        try:
            img[resultYOff:resultYOff+h, margin:margin+w] = scan[y:y+h, x:x+w]
        except:
            pass
        cv2.putText(img, row['text'], (margin + w + 30, resultYOff+h), cv2.FONT_HERSHEY_PLAIN, h/10, (0,255,0), h//10)
        cv2.line(img, (margin + w + 30, resultYOff + int((resultYOff+h)/2)), (int(row['orig_point'][0]), int(row['orig_point'][1])), (0,0,0), 2)
        resultYOff += margin + h

    return img

def complete_text_from_image(input):
    print('pre-processing image/frame for text approximation...')
    processed = preprocess_frame(input)

    #detect areas containing long contours that may be text
    print('pre-processing complete. assembling list of potential text locations...')
    areas, centroids = get_potential_text_areas(input, processed)

    #get single image containing text areas to be scanned and parallel colored version for visuals
    print('potential text locations collected. Generating composite image for OCR scanning...')
    scan_image, colors, buffer = assemble_text_scan_image(input, areas, centroids)

    #scan single image for text
    print('composite image created. Scanning for text...')
    text_results = text_data_from_image(scan_image, buffer)
    results_filtered = [row for row in text_results if float(row['conf']) > 80 and 
                        len(row['text']) > 0]

    #overlay detected text and extracted string side-by-side on top of captured frame
    print('results received and sorted. Overlaying results on input image...')
    output_image = overlay_results(input, results_filtered, colors)

    return output_image

def main(args):

    if 'opencv-python-headless' in str(sp.check_output(['pip', 'list'])):
        print("opencv-python-headless detected. Uninstall it and re-install opencv-python (non-headless).\n")
        exit()

    #user has supplied an argument?
    if len(args) > 0:

        if args[0] == 'list':
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

        #argument is not list, assume it's a path to an image
        else:
            inputImage = cv2.imread(args[0])
            if inputImage is None:
                print("error opening input file.")
                exit()

            out_frame = complete_text_from_image(inputImage)

            cv2.imshow('test', out_frame)
            cv2.waitKey()

        cv2.destroyAllWindows()
        exit()

    # no argument supplied, use webcam
    else:

        pytesseract.pytesseract.tesseract_cmd = r'C:\dev\Tesseract-OCR\tesseract.exe'

        # cam = cv2.VideoCapture(cam_index)
        cam = cv2.VideoCapture(cam_index, capture_api)

        cam_read_attempts = 0

        while True:

            cam_read_attempts += 1

            print(f"Getting new frame - attempt #{cam_read_attempts}")
            success, latest_frame = cam.read()
            
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

                out_frame = complete_text_from_image(latest_frame)

                cv2.imshow('test', out_frame)
                cv2.waitKey()
                



main(argv[1:])