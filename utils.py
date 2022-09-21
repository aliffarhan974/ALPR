import os
import time
import cv2
import sys
import datetime


def search_file(file_name=''):
    """
        Function to find specific file. Example; configuration and weights file,
        input image/video or any file name given
    """
    file_path = []
    if os.name == 'posix':
        search_dir = "/home"
    elif os.name == 'nt':
        search_dir = r"C:\\"
    else:
        print("Unsupported OS")
        search_dir = ''
    if file_name:
        file_list = []
        not_found = ''
        for root, dirs, files in os.walk(search_dir):
            file_list.append(files)
            if file_name in files:
                rel_path = os.path.relpath(os.path.join(root, file_name))
                file_path.append(rel_path)

        for file in file_list:
            if file_name not in file:
                not_found = True
            else:
                not_found = False
                break

        if not_found:
            sys.exit(f"no file name '{file_name}' found\n")

    print(f"{file_name} found at {min(file_path, key=len)}")
    return min(file_path, key=len)


def search_folder(folder_name=''):
    """
    Function to find specific folder. Example; Darknet folder containing configuration and weights file,
    folder of input image or video or any folder name given
    """
    if os.name == 'posix':
        search_dir = "/home"
    elif os.name == 'nt':
        search_dir = r"C:\\"
    else:
        print("Unsupported OS")
        search_dir = ''

    if folder_name:
        folder_list = []
        found_folder_list = []

        for root, dirs, files in os.walk(search_dir):
            folder_list.append(dirs)

            if folder_name in dirs:
                found_folder_list.append(str(os.path.join(root, folder_name)))

        if len(found_folder_list) > 1:
            if not os.path.isdir(os.path.join(os.path.commonpath(found_folder_list), folder_name)):
                return min(found_folder_list, key=len)
            else:
                return os.path.join(os.path.commonpath(found_folder_list), folder_name)

        not_found = True

        if not_found:
            sys.exit(f"no folder name '{folder_name}' found\n")


def create_image_folder(image_folder_name='Image'):
    """Mainly used for creating image folder but still can be used to create any new folder"""
    image_folder_name = os.path.join(image_folder_name, str(datetime.date.today()))
    current_folder_name = os.getcwd()
    image_folder = os.path.join(current_folder_name, str(image_folder_name))
    minutes = str(datetime.datetime.now().strftime("%I:%M %p"))
    image_location = os.path.join(image_folder, minutes)
    try:
        try:
            os.path.exists(os.path.join(current_folder_name, image_folder_name))
            os.makedirs(image_folder)
        except FileExistsError:
            pass
        os.path.exists(image_location)
        os.makedirs(image_location)
    except FileExistsError:
        pass
    finally:
        return image_location


def put_fps(frame, prev_frame_time):
    """Putting the information of frames per second (FPS) into the video display"""
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    fps = float(f'{fps:.2f}')
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (100, 255, 0), 3, cv2.LINE_AA)


def check_time(elapsed_hour, elapsed_minutes):
    """Keep track of time, so results can be saved every ?? minutes or once an hour"""
    current_minutes = int(datetime.datetime.now().strftime("%M"))
    current_time = datetime.datetime.now()
    change_hour = 60 - current_minutes
    change_time = current_time + datetime.timedelta(minutes=elapsed_minutes)
    if elapsed_hour:
        return change_hour
    else:
        return change_time


def get_highest(conf_list, duplicate_index):
    """Get the bounding box coordinate of license plate with the highest confidences if there are many"""
    res = [conf_list[n] for n in duplicate_index]
    res = max(res, key=float)
    res = conf_list.index(res)
    return res
