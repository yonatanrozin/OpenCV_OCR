print("---\nPython Webcam OCR\nYonatan Rozin\nPress Q at any time to quit\n---")
print('Importing modules...')
import time 

launch_time = time.time()
import subprocess as sp
import cv2
import pytesseract
import numpy as np
import traceback
from sys import argv



#give reader time to read message
while time.time() - launch_time < 2:
    pass

#camera index and capture API (optional)
cam_index = 0
# capture_api = cv2.CAP_DSHOW
# capture_api = cv2.CAP_MSMF
capture_api = None

# uncomment this line to manually provide path to tesseract executable file
# pytesseract.pytesseract.tesseract_cmd = '' # add path to executable here

result_margin = 10 # margin width (in px) between text areas / image borders

debug_mode = 'debug' in argv

if debug_mode:
    logFile = open('logs/log.txt', 'w+')

def print_debug(msg):
    if debug_mode:
        # print(msg)
        logFile.write(msg + '\n')

# receives an image, performs some pre-processing to prepare for finding long contours that may be text:
#   convert to grayscale
#   blur for anti noise
#   adaptive thresholding for flexible binarization
#   [mostly] horizontal dilation - causes letters in lines of text to blur together into single contours
def preprocess_frame(img):
    print_debug('applying greyscale +  blur')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(9,9),0)
    # print_debug('applying adaptive threshold + dilation')
    img = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    img = cv2.dilate(img, kernel, iterations=1)

    if debug_mode:
        cv2.imwrite('logs/processed_frame.jpg', img)
    return img

# receives an image, probably from cv2.VideoCapture.read()
#   pre-processes image to prepare for contour detection
#   performs contour detection, finding rotated (min-area) rect for each
#   imposes a slight width threshold on results 
#   returns list of cropped rectangles, aligned horizontally and
#   a parallel list containing central coordinates for each corresponding rectangle
def get_potential_text_areas(img, processed):

    processed_show = img.copy()

    print_debug('retrieving contours')
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    aligned_areas = []

    for i in range(len(contours)):

        print_debug(f'contour {i}/{len(contours)}:')

        contour = contours[i]

        print_debug('fetching rotated bounding box')
        rect = cv2.minAreaRect(contour)
        center, size, theta = rect

        print_debug(f'Box shape: {size[1]}x{size[0]}, angle={theta}')

        if size[0]*size[1] < 4000:
            print_debug('contour size below threshold, skipping.')
            continue

        size = (size[0]+10, size[1]+10) # include some whitespace

        # align (potential) text using angle of rotated bounding box
        print_debug('rotating contour to align text.')
        center, size = tuple(map(int, center)), tuple(map(int, size))
        M = cv2.getRotationMatrix2D( center, theta, 1)

        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        area = cv2.getRectSubPix(dst, size, center)

        if theta > 45:
            area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)

        if area is not None:

            print_debug('adding potential text area to list')
            aligned_areas.append(area)

            box = cv2.boxPoints(rect) 
            box = np.intp(box)
            cv2.drawContours(processed_show,[box],0,(0,0,0),2)
        else:
            print_debug("invalid area extracted. Skipping.")

    print_debug('sorting potential text areas by size')
    aligned_areas = sorted(aligned_areas, key=lambda item: item.shape[0]*item.shape[1])
    if debug_mode:
        cv2.imwrite('logs/processed_highlighted_text_areas.jpg', processed_show)
        cv2.imshow('intermediate', cv2.resize(processed_show, (1000,1000)))
    return aligned_areas

def assemble_text_scan_image(areas):

    margin = 30

    #clear composite scan and parallel colors
    totalScan.fill(255)
    scan_colors.fill(255)

    hScan, wScan = totalScan.shape

    xOff = margin
    yOff = margin
    row_height = 0
    for i in range(len(areas)):

        print_debug(f'Text area #{i}/{len(areas)}:')

        area = areas[i]

        print_debug('pre-processing area for text detection')
        processed = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        # norm_matrix = np.zeros((img.shape[0], img.shape[1]))
        # processed = cv2.normalize(processed, norm_matrix, 0, 255, cv2.NORM_MINMAX)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]

        h, w = processed.shape

        try:
            
            if xOff + w < wScan:
                totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                xOff += (w + margin)
                row_height = max(row_height, h)
            else:

                xOff = margin
                yOff += row_height + margin
                totalScan[yOff:yOff+h, xOff: xOff+w] = processed
                scan_colors[yOff:yOff+h, xOff: xOff+w] = area
                xOff += w + margin
                row_height = 0
            print_debug('adding area to composite scan image')
        except:
            print_debug('Error adding area to composite scan image.')
            pass

    if debug_mode:
        cv2.imwrite('logs/scan_image.jpg', totalScan)
        cv2.imwrite('logs/scan_image_color.jpg', scan_colors)

    return totalScan, scan_colors

def text_data_from_image(img):
    print_debug("Retrieving text from scan image with Tesseract")
    data = pytesseract.image_to_data(img, lang="eng", config='--psm 12').split('\n')[:-1] #ignore last item
    data_labels = data[0].split('\t')
    data_values = [row.split('\t') for row in data[1:]]
    data_entries = [dict(zip(data_labels, entry)) for entry in data_values]

    return data_entries

def overlay_results(img, data, scan):

    resultYOff = result_margin
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
            img[resultYOff:resultYOff+h, result_margin:result_margin+w] = scan[y:y+h, x:x+w]
            cv2.putText(img, row['text'], (result_margin + w + 30, resultYOff+h), cv2.FONT_HERSHEY_PLAIN, h/10, (255,100,100), h//7)
            print_debug("overlaying detected text and UTF-encoded result.")
        except:
            print_debug("Error overlaying detected text and UTF-encoded result.")

        resultYOff += result_margin + h

    return img

def complete_text_from_image(input):

    shape_orig = input.shape

    cv2.imwrite('logs/input_frame.jpg', input)

    print_debug(f"\nInput image size: {str(shape_orig)} - enlarging image, maintaining proportions...")
    ratio = input.shape[1]/input.shape[0]
    input = cv2.resize(input, (2000, int(2000/ratio)))

    print_debug("\nPre-processing frame...")
    processed = preprocess_frame(input)

    print_debug("\nFinding potential text areas from contour data...")
    #detect areas containing long contours that may be text
    areas = get_potential_text_areas(input, processed)

    #get single image containing text areas to be scanned and parallel colored version for visuals
    print_debug("\nAssembling text areas into single image for text detection...")
    scan_image, colors = assemble_text_scan_image(areas)

    if debug_mode:
        cv2.imshow('composite scan image', scan_image)
        cv2.waitKey(5)

    #scan single image for text
    print_debug("\nGathering formatted text data from image...")
    text_results = text_data_from_image(scan_image)
    results_filtered = [row for row in text_results if float(row['conf']) > 70 and 
                        len(row['text']) > 1] 

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

            video_duration = -1

        elif args[0] == 'video':

            try:
                print("Getting video file")
                video_file = args[1]
            except:
                print("no video file path supplied.")
                exit()

            cam = cv2.VideoCapture(video_file)

            video_fps = cam.get(cv2.CAP_PROP_FPS)
            frameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = frameCount // video_fps

            print_debug(f'video file loaded: {video_duration}s at {video_fps}fps.')

        read_attempts = 0
        vid_start_time = time.time()
        while True:
            elapsed_time = time.time() - vid_start_time

            if video_duration != -1:
                video_frame_number = int(elapsed_time * video_fps)
                if video_frame_number > frameCount:
                    print("end of video file reached.")
                    exit()
                cam.set(cv2.CAP_PROP_POS_FRAMES, video_frame_number-1)

            read_attempts += 1
            print_debug(f"\n\nGetting new frame - attempt #{read_attempts} for this frame")
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

                frame_process_start_time = time.time()
                out_frame = complete_text_from_image(latest_frame)
                frame_process_elapsed_time = time.time() - frame_process_start_time
                print(f'Frame processing complete in {frame_process_elapsed_time:.2f}s.')

                cv2.imshow('OCR Webcam', out_frame)

                if cv2.waitKey(5) == ord('q'):
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
totalScan = np.zeros([10000,2000],dtype=np.uint8) 
scan_colors = np.ones([10000,2000, 3], dtype=np.uint8) * 255
centroid_location_buffer = np.zeros([2000,1000,3], dtype="float32")

try:
    main(argv[1:])
except Exception as e:
    if debug_mode:
        logFile.write(traceback.format_exc())
    raise e
