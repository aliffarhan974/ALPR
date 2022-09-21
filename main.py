#! /usr/bin/env python3

import argparse
import queue
import textwrap
import threading

from detect import detection
from ocrimage import *
from utils import *


class CustomArgumentFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Formats argument help which maintains line length restrictions as well as appends default value if present."""

    def _split_lines(self, text, width):
        # noinspection PyProtectedMember
        text = super()._split_lines(text, width)
        new_text = []

        # loop through all the lines to create the correct wrapping for each line segment.
        for line in text:
            if not line:
                # this would be a new line.
                new_text.append(line)
                continue

            # wrap the line's help segment which preserves new lines but ensures line lengths are
            # honored
            new_text.extend(textwrap.wrap(line, width))

        return new_text


def argument():
    """ Command Line Interface (CLI) side"""
    config_paths = "yolov4-obj.cfg"
    weights_paths = "yolov4-obj_1000.weights"
    labels_paths = "obj.names"
    backends = (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_VKCOM, cv2.dnn.DNN_BACKEND_CUDA)
    targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16,
               cv2.dnn.DNN_TARGET_MYRIAD, cv2.dnn.DNN_TARGET_VULKAN, cv2.dnn.DNN_TARGET_CUDA,
               cv2.dnn.DNN_TARGET_CUDA_FP16, cv2.dnn.DNN_TARGET_HDDL)
    ocr = {'Tesseract': 1, 'EasyOCR': 2}
    parse = argparse.ArgumentParser(description="ALPR detection system. For arguments given, "
                                                "path or filename can be given either way "
                                                "as the %(prog)s will determine "
                                                "the existence of path or filename given and"
                                                "find the file if nonexistent.",
                                    formatter_class=CustomArgumentFormatter)
    parse.add_argument('input', metavars='INPUT',  
                       help="Root or relative path to input image/video file/folder/text or "
                            "simply give the name of the image or video file")
    parse.add_argument('-d', "--dont_show", action="store_true",
                       help="Display for detection")
    parse.add_argument('--confidence', type=float, default=0.5,
                       help="minimum probability to filter detections")
    parse.add_argument('-t', '--threshold', type=float, default=0.3,
                       help="threshold when applying non-maxima suppression")
    parse.add_argument('-c', '--config', default=config_paths,
                       help="Root or relative path to config file")
    parse.add_argument('-w', '--weights', default=weights_paths,
                       help="Root or relative path to the weights file")
    parse.add_argument('-l', '--labels', default=labels_paths,
                       help="Root or relative path to the labels/'.names' file")
    parse.add_argument('--count', type=int,
                       help="Count the approximation of the total number of "
                            "vehicle passing through the lane for ? minutes")
    parse.add_argument('--backend', choices=backends, default=cv2.dnn.DNN_BACKEND_DEFAULT, type=int,
                       help="Choose one of computation backends: \n"
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/),\n"
                            "%d: Intel's Deep Learning Inference Engine"
                            "(https://software.intel.com/openvino-toolkit),\n"
                            "%d: OpenCV implementation, "
                            "%d: VKCOM, "
                            "%d: CUDA" % backends)
    parse.add_argument('--target', choices=targets, default=cv2.dnn.DNN_TARGET_CPU, type=int,
                       help="Choose one of target computation devices:\n"
                            "%d: CPU target (by default), "
                            "%d: OpenCL, "
                            "%d: OpenCL fp16 (half-float precision),"
                            "%d: NCS2 VPU, "
                            "%d: Vulkan,\n"
                            "%d: CUDA,"
                            "%d: CUDA fp16 (half-float preprocess), " 
                            "%d: HDDL VPU" % targets)
    parse.add_argument('--ocr', choices=ocr.values(), default=2, type=int,
                       help="Choose one of OCR engine to be used (1, 2):\n"
                            "1: Tesseract OCR "
                            "2: Easy OCR")
    return parse.parse_args()


def check_argument(args):
    """Check if any arguments is either invalid (threshold) or missing file"""
    assert 0 < args['threshold'] < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args['config']):
        args['config'] = search_file(args['config'])
    if not os.path.exists(args['weights']):
        args['weights'] = search_file(args['weights'])
    if not os.path.exists(args['labels']):
        args['labels'] = search_file(args['labels'])
    if args['input'] and not os.path.exists(args['input']):
        if args['input'].endswith(('.mp4', '.mkv', '.avi')):
            args['input'] = search_file(args['input'])
    return args


def video_display(image_queue, net, args, time_taken):
    """Display video feed of the detection with its cropped license plate window"""
    caps = cv2.VideoCapture(args['input'])
    x = y = 0
    labels_path = args['labels']
    with open(labels_path, 'r') as f:
        labels = f.read().strip().split('\n')
    while caps.isOpened():
        ret, frame = caps.read()
        prev_frame_time = time.time()
        if not ret:
            break
        box, name, mean_time, conf = detection(frame, net, args, labels)
        time_taken.put(mean_time)
        crop_image = None
        put_fps(frame, prev_frame_time)
        if "license-plate" in name:
            index = name.index("license-plate")
            duplicate = [i for i, n in enumerate(name) if n == 'license-plate']
            if len(duplicate) > 1:
                index = get_highest(conf, duplicate)
            x, y, w, h = box[index]
            crop_image = frame[y:y + h, x:x + w]
        # if using dont_show flags, no video window is displayed
        if not args['dont_show']:
            cv2.namedWindow('cap', cv2.WINDOW_NORMAL)
            cv2.namedWindow('license plate', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('cap', 960, 1080)
            cv2.moveWindow('cap', 960, 0)
            cv2.imshow('cap', frame)
            if crop_image is not None:
                if 540 <= y <= 1080 and 0 <= x <= 1920:
                    cv2.imshow("license plate", crop_image)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == 32:
                cv2.waitKey(-1)

        image_queue.put(crop_image)

    caps.release()
    cv2.destroyAllWindows()
    # Stopping the thread
    image_queue.put('stop')
    time_taken.put(-1)

    print("exiting ...")


def results_operation(image_queue, time_taken):
    """Saving the image of license plates and writing the results into spreadsheets"""
    count = 0
    results, copy_results = ({'Timestamp': [], "Detected Plate": [], "Image Name": []} for _ in range(2))
    mean_time_list = []
    loc = create_image_folder()
    reader = load_easyocr()
    change_time = check_time(False, 3)
    while True:
        plate = image_queue.get()
        mean_time = time_taken.get()
        if str(plate) == 'stop' or mean_time == -1:
            break
        if datetime.datetime.now() >= change_time:
            # need to modify so only one spreadsheet is saved (currently spreadsheet is saved in new folder every ??
            # minutes)
            change_time = check_time(False, 3)
            loc = create_image_folder()
            final, not_found = get_highest_accuracy(results)
            pandas_to_excel(final, not_found=False, sheet_name=str(datetime.datetime.now()
                                                                   .strftime("%d-%m-%Y, %H%M,%S")))
            results = {'Timestamp': [], "Detected Plate": [], "Image Name": []}

        if plate is not None:
            try:
                new = time.time()
                # need to add choices for Tesseract OCR
                timestamp, text = easy_ocr(plate, reader)
                mean_time = mean_time + (time.time() - new)
                mean_time_list.append(mean_time)
                name = loc + '/' + str(count) + ".jpg"
                cv2.imwrite(name, plate)
                # printing information on the terminal
                print(f"{text:12} \t\t\t\t {mean_time:10.3f} \t\t {timestamp}")
                count += 1
                results['Timestamp'].append(timestamp)
                results["Detected Plate"].append(text)
                results["Image Name"].append(name)
                copy_results['Timestamp'].append(timestamp)
                copy_results["Detected Plate"].append(text)
                copy_results["Image Name"].append(name)
            except cv2.error:
                pass
    final, not_found = get_highest_accuracy(results)
    pandas_to_excel(final, not_found=False, sheet_name=str(datetime.datetime.now().strftime("%d-%m-%Y, %H%M,%S")))
    final, not_found = get_highest_accuracy(copy_results)
    pandas_to_excel(final, not_found, sheet_name='final')
    average = sum(mean_time_list) / len(mean_time_list)
    print(f"Average time taken to process one image is {average:.2f}")


def main():

    args = vars(argument())
    args = check_argument(args)

    cuda = cv2.getBuildInformation()
    if 'cuda' in cuda:
        args['backend'] = cv2.dnn.DNN_BACKEND_CUDA
        args['target'] = cv2.dnn.DNN_TARGET_CUDA
    else:
        args['backend'] = cv2.dnn.DNN_BACKEND_DEFAULT
        args['target'] = cv2.dnn.DNN_TARGET_CPU
    print("[INFO] loading YOLO into disk\n")
    net = cv2.dnn.readNetFromDarknet(args['config'], args['weights'])
    net.setPreferableBackend(args['backend'])
    net.setPreferableTarget(args['target'])

    plate_queue = queue.Queue()
    time_taken = queue.Queue()
    print("Plate number \t\t\t\t Time taken \t\t Timestamp")
    print("*-*-*-*-*-*- \t\t\t\t *-*-*-*-*- \t\t *-*-*-*-*")

    start = datetime.datetime.now()
    p1 = threading.Thread(target=video_display, args=(plate_queue, net, args, time_taken))
    p2 = threading.Thread(target=results_operation, args=(plate_queue, time_taken))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print(f"Program took {datetime.datetime.now() - start} to finish")


if __name__ == '__main__':
    key_queue = queue.Queue()
    main()
