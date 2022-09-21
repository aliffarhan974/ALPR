import os
import re
import cv2
import easyocr
import datetime
import warnings
import openpyxl
import pytesseract
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)


def process_image(plate_image):
    """
    This function will process the image for Tesseract OCR only

    param:
        plate_image: cropped plate image

    return:
        new_img: processed plate image

    """
    ret, thresh1 = cv2.threshold(plate_image, 120, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh1, 170, 200)
    contours, new = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    img2 = plate_image.copy()
    cv2.drawContours(img2, contours, -1, (0, 255, 0), 1)

    x, y, w, h = cv2.boundingRect(contours[0])
    new_img = img2[y:y + h, x:x + w]
    return new_img


def tesseract_ocr(img):
    new_img = process_image(img)
    text = pytesseract.image_to_string(new_img, lang='eng',
                                       config='--oem 3 --psm 6 -c '
                                              'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = re.sub("\\n", "", text)
    return datetime.datetime.now(), text


def easy_ocr(images, reader):
    text = reader.readtext(images, detail=0)
    text = ''.join(text)
    return datetime.datetime.now(), text


def load_easyocr():
    reader = easyocr.Reader(['en'])
    return reader


def filter_result(prefilter):
    """
    This function will filter the plate number according to preexisting rules from JPJ

    param:
        pre_filter: dict{key=object name ('Timestamp, Plate, Confidence'): value=[detected plate, image name]}

    return:
        post_f: dict{key=timestamp: value=[detected plate, image name]}

    """
    post_f = {"Timestamp": [], "Detected Plate": [], "Image Name": []}
    for index, plate_num in enumerate(prefilter['Detected Plate']):
        plate_num = plate_num.replace("\n", '')
        plate_num = plate_num.replace(" ", '')
        plate_num = plate_num.upper()
        if plate_num and 5 <= len(plate_num) <= 11:
            if plate_num.isalnum():
                if plate_num.isnumeric():
                    pass
                elif plate_num.isalpha():
                    pass
                elif plate_num[:1].isdigit():
                    pass
                # elif re.search('I', plate_num) or re.search('O', plate_num):
                #     pass
                else:
                    digit = 0
                    for n in plate_num:
                        if n.isnumeric():
                            digit += 1
                    if digit > 4:
                        continue
                    plate_num = re.split(r'(\d+)', plate_num)
                    count = 0
                    for li in plate_num:
                        if li.isnumeric():
                            count += 1
                    if count == 1:
                        pass
                    else:
                        continue
                    plate_num = ''.join(plate_num)
                    if plate_num in post_f["Detected Plate"]:
                        continue
                    post_f["Timestamp"].append(prefilter["Timestamp"][index])
                    post_f["Detected Plate"].append(plate_num)
                    post_f["Image Name"].append(prefilter["Image Name"][index])
    return post_f


def pandas_to_excel(output, not_found, sheet_name):
    """
        This function will use pandas to save result into the spreadsheet

        param:
            output: a dictionary containing information about the license plate
            not_found: a list of undetected license plates
            sheet_name: name of sheet in workbook, default is current timestamp
    """
    output = pd.DataFrame(output)
    not_found = pd.Series(not_found)
    from utils import create_image_folder
    path = create_image_folder(image_folder_name='Results') + '/' + \
        str(datetime.datetime.now().strftime("%I:00 %p")) + ".xlsx"
    if os.path.isfile(path):
        writer = pd.ExcelWriter(path, engine='openpyxl', mode='a')
        output.to_excel(writer, sheet_name=sheet_name, index=False)
        row = 2
        writer.sheets[sheet_name].insert_cols(4)
        writer.sheets[sheet_name]['D1'] = "Detected Image"
        for i in output.loc[:, "Image Name"]:
            img = openpyxl.drawing.image.Image(i)
            img.width = 51.84
            img.height = 18.24
            img.anchor = "D" + str(row)
            writer.sheets[sheet_name].add_image(img)
            row += 1
        writer.close()
    else:
        writer = pd.ExcelWriter(path, engine='openpyxl')
        output.to_excel(writer, sheet_name=sheet_name, index=False)
        row = 2
        writer.sheets[sheet_name].insert_cols(4)
        writer.sheets[sheet_name]['D1'] = "Detected Image"
        for i in output.loc[:, "Image Name"]:
            img = openpyxl.drawing.image.Image(i)
            img.width = 51.84
            img.height = 18.24
            img.anchor = "D" + str(row)
            writer.sheets[sheet_name].add_image(img)
            row += 1
        writer.close()
    if not_found.any():
        writer = pd.ExcelWriter(path, engine='openpyxl', mode='a')
        not_found.to_excel(writer, sheet_name="not found", index=False)
        writer.close()


def accuracy_res(post_f):
    """
    This function is used to calculate accuracy of OCR result and only used by get_highest_accuracy function

    param:
        post_f: a dictionary of filtered license plates with its other info

    return:
        post_f: the same dictionary with addition info on accuracy/confidences of the license plates
        correct_list: a list of correct license plates taken from Assignment 2
    """
    correct_list = anchor_list()
    key, test = zip(*post_f.items())
    post_f['Confidence'] = []
    post_f["Correct Plate"] = []
    announce = []
    pop = []
    for index, testing in enumerate(test[1]):
        count_list, order_list = ([] for _ in range(2))
        # case1 = in order and complete > 100%
        if testing in correct_list:
            post_f['Confidence'].append(100)
            post_f["Correct Plate"].append(testing)
            announce.append(testing)
            continue
        for correct in correct_list:
            count = len([x for x in testing if x in correct])
            order = len([1 for x in range(min(len(testing), len(correct))) if testing[x] == correct[x]])
            count_list.append(count)
            order_list.append(order)
        count = max(count_list)
        count_idx = count_list.index(count)
        order = max(order_list)
        if correct_list[count_idx] in announce:
            pop.append(index)
            continue
        # case2 = in order but incomplete > count correct character
        if count == order and count < len(correct_list[count_idx]):
            pass
        # case3 = unordered but complete > 50%
        elif count >= len(correct_list[count_idx]):
            count = len(correct_list[count_idx]) / 2
        # case4 = unordered and incomplete > same as case2 but if result < 50%, final = 0
        else:
            count = count if count >= (len(correct_list[count_idx]) / 2) else 0
        confidence = count / len(correct_list[count_idx]) * 100
        post_f['Confidence'].append(confidence)
        post_f["Correct Plate"].append(correct_list[count_idx])
    for x in key:
        for y in sorted(pop, reverse=True):
            del post_f[x][y]
    return post_f, correct_list


def get_highest_accuracy(pre_f):
    """
        This function will get the license plate with the highest accuracy for each plates in correct_list by first
        calling other functions to filter the results and calculate the confidence

        param:
            pre_f: a dictionary of unfiltered license plates result

        return:
            final: a dictionary of the final results that will be saved into the spreadsheet
            not_found: a list of undetected license plates
        """
    post_f = filter_result(pre_f)
    res_list, correct_list = accuracy_res(post_f)
    key_list, value_list = zip(*res_list.items())
    final = {k: [] for k in key_list}
    not_found, existing = ([] for _ in range(2))

    for idx, value in enumerate(value_list[3]):
        if value_list[4][idx] in existing:
            continue
        if value == 100:
            [final[key_list[x]].append(value_list[x][idx]) for x in range(5)]
            existing.append(value_list[4][idx])
            continue
        temp_list = [[value_list[3][index], val, index] for index, val in enumerate(value_list[1])
                     if value_list[4][index] in value_list[4][idx]]
        if len(temp_list) == 0:
            not_found.append(value_list[4][idx])
        elif len(temp_list) == 1:
            [final[key_list[x]].append(value_list[x][temp_list[0][2]]) for x in range(5)]
            existing.append(value_list[4][idx])
        else:
            temp_list = max(temp_list, key=lambda x: x[0])
            [final[key_list[x]].append(value_list[x][temp_list[2]]) for x in range(5)]
            existing.append(value_list[4][idx])

    [not_found.append(correct) for correct in correct_list if correct not in existing]
    final["Timestamp"] = sorted(final["Timestamp"])
    indexes = [value_list[0].index(idxs) for idxs in final["Timestamp"]]
    for idxs, key in enumerate(key_list):
        if idxs == 0:
            continue
        final[key] = [value_list[idxs][y] for y in indexes]
    final["Timestamp"] = [str(x) for x in final["Timestamp"]]
    return final, not_found


def anchor_list():
    correct_list = ['WDQ8239', 'MDK8115', 'TCB1475', 'MCV9619', 'VHG8193', 'JSL8009', 'JUA2459', 'WCW2531', 'HBB5644',
                    'BNR2398', 'PNU7796', 'KFF901', 'JNA2835', 'MCF8493', 'WTM5546', 'VDM3719', 'PPL116', 'BKS2145',
                    'WA4217B', 'BNJ253', 'DDK6828', 'AED2656', 'WQS4229', 'BLG1684', 'HBB5184', 'BHX4032', 'W5689U',
                    'BHE6706', 'VU9196', 'VBU5075', 'JJT9122', 'AKR9233', 'BMU1457', 'BPS7061', 'BNU259', 'WFJ8056',
                    'PNK6300', 'VFA8025', 'KFB981', 'VAH3109', 'BHS8972', 'VAH1133', 'PLM1147', 'VFK8339', 'BHV7966',
                    'VCN7021', 'PNG7232', 'WXR9131', 'BPH3877', 'WRC8633', 'NDE9392', 'BMY1828', 'WYY2563', 'BKF6514',
                    'JSV682', 'BGE4649', 'VGQ7328', 'BHK2020', 'VCL5902', 'KEG6115', 'WQT2154', 'VBS5316', 'WB2234W',
                    'QBD5868', 'VFD7990', 'W5898L', 'WKP7787', 'WXT7198', 'BNS9931', 'BLV9871', 'JLF2510', 'HWE7809',
                    'NBL2917', 'WA3881M', 'JTS7883', 'WD9872F', 'VAU2586', 'VBK6989']
    return correct_list
