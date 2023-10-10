print("---\nPython Webcam OCR\nYonatan Rozin\nPress Q at any time to quit\n---")
print('Importing modules...')
import time 

launch_time = time.time()
import subprocess as sp
import cv2
import pytesseract
import numpy as np
from sys import argv

#give reader time to read message
while time.time() - launch_time < 2:
    pass

#camera index and capture API (optional)
cam_index = 0
capture_api = cv2.CAP_DSHOW
# capture_api = cv2.CAP_MSMF
# capture_api = None

margin = 10 # margin width (in px) between text areas / image borders

show_intermediate = 'debug' in argv

def print_debug(msg):
    if show_intermediate:
        print(msg)

# receives an image, performs some pre-processing to prepare for finding long contours that may be text:
#   convert to grayscale
#   blur for anti noise
#   adaptive thresholding for flexible binarization
#   [mostly] horizontal dilation - causes letters in lines of text to blur together into single contours
def preprocess_frame(img):
    print_debug('applying greyscale +  blur')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    # print_debug('applying adaptive threshold + dilation')
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
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

    print_debug('retrieving contours')
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    aligned_areas = []
    centroids = []

    for i in range(len(contours)):

        print_debug(f'contour {i}/{len(contours)}:')

        contour = contours[i]

        print_debug('fetching rotated bounding box')
        rect = cv2.minAreaRect(contour)
        center, size, theta = rect

        print_debug(f'Box shape: {size[1]}x{size[0]}, angle={theta}')

        if size[0]*size[1] < 3000:
            print_debug('contour size below threshold, skipping.')
            continue

        size = (size[0]+10, size[1]+10) # include some whitespace

        # align (potential) text using angle of rotated bounding box
        print_debug('rotating contour to align text.')
        center, size = tuple(map(int, center)), tuple(map(int, size))
        M = cv2.getRotationMatrix2D( center, theta, 1)

        # dst_processed = cv2.warpAffine(processed, M, (img.shape[1], img.shape[0]))
        # area_processed = cv2.getRectSubPix(dst_processed, size, center)

        # white_pixel_count = np.sum(area_processed == 255)      # extracting only white pixels 
        # black_pixel_count = np.sum(area_processed == 0)        # extracting only black pixels 
        # total_pixel_count = white_pixel_count + black_pixel_count

        # if white_pixel_count/total_pixel_count < .4:
        #     print_debug('contour is mostly whitespace, skipping.')
        #     continue

        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        area = cv2.bitwise_not(cv2.getRectSubPix(dst, size, center))

        if theta > 45:
            area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)

        if area is not None:
            print_debug('adding potential text area to list')
            aligned_areas.append(area)
            centroids.append(center)

            box = cv2.boxPoints(rect) 
            box = np.intp(box)
            cv2.drawContours(processed_show,[box],0,(255,255,255),1)
        else:
            print_debug("invalid area extracted. Skipping.")

    print_debug('sorting potential text areas by size')
    aligned_areas = sorted(aligned_areas, key=lambda item: item.shape[0]*item.shape[1])

    return aligned_areas, centroids, processed_show

def assemble_text_scan_image(img, areas, centroids):

    #clear composite scan and parallel colors
    totalScan.fill(255)
    scan_colors.fill(255)

    #parallel to scan but contains (x,y) positions of centroids of contours
    #used to lookup original location of a text result using its position in the composite scan image
    centroid_location_buffer = np.zeros([2000,1000,3], dtype="float32")

    hScan, wScan = totalScan.shape

    xOff = margin
    yOff = margin
    row_height = 0
    for i in range(len(areas)):

        print_debug(f'Text area #{i}/{len(areas)}:')

        area = areas[i]
        cent = centroids[i]

        print_debug('pre-processing area for text detection')
        processed = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        norm_matrix = np.zeros((img.shape[0], img.shape[1]))
        processed = cv2.normalize(processed, norm_matrix, 0, 255, cv2.NORM_MINMAX)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)[1]

        h, w = processed.shape

        try:
            
            if xOff + w < wScan:
                totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                cv2.rectangle(centroid_location_buffer, (xOff, yOff), (xOff + w, yOff + h), (cent[0]/1000,cent[1]/1000,0), -1)
                xOff += (w + margin)
                row_height = max(row_height, h)
            else:

                xOff = margin
                yOff += row_height + margin
                totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                cv2.rectangle(centroid_location_buffer, (xOff, yOff), (xOff + w, yOff + h), (cent[0],cent[1],0), -1)
                xOff += w + margin
                row_height = 0
            print_debug('adding area to composite scan image')
        except:
            print_debug('Error adding area to composite scan image.')
            pass

    return totalScan, scan_colors, centroid_location_buffer

def text_data_from_image(img, buffer):
    print_debug("Retrieving text from scan image with Tesseract")
    data = pytesseract.image_to_data(img, lang="eng", config='--psm 4').split('\n')[:-1] #ignore last item
    data_labels = data[0].split('\t')
    data_values = [row.split('\t') for row in data[1:]]
    data_entries = [dict(zip(data_labels, entry)) for entry in data_values]
    # for row in data_entries:
    #     row['centerX'] = int(float(row['left']) + float(row['width'])/2)
    #     row['centerY'] = int(float(row['top']) + float(row['height'])/2)
    #     row['orig_point'] = buffer[row['centerY'], row['centerX']][:2]*1000
    return data_entries

def overlay_results(img, data, scan):

    resultYOff = margin
    for i in range(len(data)):
        row = data[i]
        print_debug(f"text result #{i}/{len(data)}:")
        x, y, w, h = (
            int(float(row['left'])),
            int(float(row['top'])),
            int(float(row['width'])),
            int(float(row['height']))
        )
        
        if 0 in (w, h):
            print_debug("invalid text dimensions. Skipping.")
            continue

        #ignore text with no letters or numbers
        if len([char for char in row['text'] if char.isalpha() or char.isdigit()]) == 0:
            print_debug('Text result contains no letters or numbers. Skipping.')
            continue

        try:
            img[resultYOff:resultYOff+h, margin:margin+w] = scan[y:y+h, x:x+w]
            cv2.putText(img, row['text'], (margin + w + 30, resultYOff+h), cv2.FONT_HERSHEY_PLAIN, h/10, (0,255,0), h//10)
            print_debug("overlaying detected text and UTF-encoded result.")
        except:
            print_debug("Error overlaying detected text and UTF-encoded result.")

        resultYOff += margin + h

    return img

def complete_text_from_image(input):

    shape_orig = input.shape

    print_debug(f"\nInput image size: {str(shape_orig)} - enlarging image, maintaining proportions...")
    ratio = input.shape[1]/input.shape[0]
    input = cv2.resize(input, (2000, int(2000/ratio)))

    print_debug("\nPre-processing frame...")
    processed = preprocess_frame(input)

    print_debug("\nFinding potential text areas from contour data...")
    #detect areas containing long contours that may be text
    areas, centroids, contour_highlights = get_potential_text_areas(input, processed)

    if show_intermediate:
        cv2.imshow('intermediate', cv2.resize(contour_highlights, (1000,1000)))

    #get single image containing text areas to be scanned and parallel colored version for visuals
    print_debug("\nAssembling text areas into single image for text detection...")
    scan_image, colors, buffer = assemble_text_scan_image(input, areas, centroids)

    if show_intermediate:
        cv2.imshow('composite scan image', scan_image)
        cv2.waitKey(5)

    #scan single image for text
    print_debug("\nGathering formatted text data from image...")
    text_results = text_data_from_image(scan_image, buffer)
    results_filtered = [row for row in text_results if float(row['conf']) > 80 and 
                        len(row['text']) > 0]

    #overlay detected text and extracted string side-by-side on top of captured frame
    print_debug("\nDisplaying results...")
    output_image = overlay_results(input, results_filtered, colors)

    output_image = cv2.resize(output_image, (1000, int(1000/ratio)))

    return output_image

def main(args):

    if 'opencv-python-headless' in str(sp.check_output(['pip', 'list'])):
        print("opencv-python-headless detected. Uninstall it and re-install opencv-python (non-headless), even if it's already installed.")
        exit()

    if len(args) == 0:
        print('run "OCR.py <\'image\'/\'webcam\'> <file_path> <\'debug\' (optional)>')
        print('or run "OCR>py list" to list available camera input devices.')

    elif args[0] == 'list':
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
        print("Replace 'cam_index' at top of document with one of these")
        print("Your camera may also require a video capture API. Uncomment a 'capture_api' option at top of document.")
        exit()


    elif args[0] == 'webcam' or args[0] == 'video':

        if args[0] == 'webcam':
            if capture_api is None:
                print("Opening webcam without capture API")
                cam = cv2.VideoCapture(cam_index)
            else:
                print('Opening webcam with capture API')
                cam = cv2.VideoCapture(cam_index, capture_api)

        elif args[0] == 'video':
            try:
                print("Getting video file")
                video_file = args[1]
            except:
                print("no video file path supplied.")
                exit()

            cam = cv2.VideoCapture(video_file)

        read_attempts = 0

        while True:

            read_attempts += 1

            print_debug(f"Getting new frame - attempt #{read_attempts} for this frame")
            success, latest_frame = cam.read()
            
            #quit after 5 consecutive failed read attempts
            if not success:
                if read_attempts > 5:
                    print("Error reading from camera. Try a different camera index or a backend API such as 'cv2.CAP_MSMF' or 'cv2.CAP_DSHOW' in 'cv2.VideoCapture() above")
                    print('Run "OCR.py list" in command line to get list of available video input devices')
                    print("See OpenCV docs here for info on video capture APIs: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d")
                    exit()
                continue

            if success:
                read_attempts = 0

                out_frame = complete_text_from_image(latest_frame)

                cv2.imshow('OCR Webcam', out_frame)

                if cv2.waitKey() == ord('q'):
                    cv2.destroyAllWindows()
                    exit()

    elif args[0] == 'image':

        try:
            image_path = args[1]
        except:
            print('No image supplied. Supply an image path using "OCR.py image <file_path>"')
            exit()

        print_debug('Getting input image')
        inputImage = cv2.imread(image_path)
        if inputImage is None:
            print(f"error opening input file '{args[0]}'")
            exit()

        out_frame = complete_text_from_image(inputImage)

        cv2.imshow('Image OCR', out_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        exit()

    else:
        print('run "OCR.py <\'image\'/\'webcam\'> <file_path?> <\'debug\' (optional)>')
        print('or run "OCR.py list" to list available camera input devices.')         

#single blank image to store all potential text areas together for efficient OCR
totalScan = np.zeros([3000,3000],dtype=np.uint8) 
scan_colors = np.ones([3000,3000, 3], dtype=np.uint8) * 255

main(argv[1:])
