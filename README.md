# ALPR
ALPR systems using YOLOv4. By default, the system will first check if OpenCV has CUDA and if does, GPU will be used.

    The calculation for accuracy of the detected plates are calculated manually by comparing the detected plates with license plates already in the databasa/list. Firstly, the detected plates number are checked if they are present in the database (e.g., 'MDK8115' in correct_list) and if they are the plate number is promptly chosen as the final result before putting it into a list, for example, final_list (this list is only for 100% accuracy). Then, the digits and characters of detected plates number will be checked against all of the license plates in the database and it will also check if they are in correct order. The detected plates with the highest similarity to the one in database is chosen but if it was similar to the one already in final_list, the program skip the plates and continue the loop.

    ### Example

| Detected plate               | Confidence | Correct Plate |     final_list       |
|:----------------------------:|:----------:|--------------:|---------------------:|
| WDO8239                      | 85.71      | WDQ8239       |          []          |
| MDO8239                      | 71.43      | WDQ8239       |          []          | 
| WDQ8239                      | 100        | WDQ8239       |     ['WDQ8239']      |
| MDQ8349                      | 71.43      | WDQ8239       |     ['WDQ8239']      |
| MDK8115                      | 100        | MDK8115       |['WDQ8239', 'MDK8115']|

    In the above example, the first two plates are checked against data in database and their accuracy were calculated. Then, when the third plate (WDQ8239) number undergoes testing, it was found to be present in the database. It was then add into final_list and any subsequent detected plate that are similar or equal to 'WDQ8239', they are skipped as it was already present in final list. Lastly, the same thing happens for the final plate number. 

    For other cases such as no detected license plates that are equal to plates in database were found, the detected plates will be checked agains every plates in database. For example;
    - detected plates = 'IC2531'
    - database = ['WDQ8239', 'MDK8115', 'TCB1475', 'MCV9619', 'VHG8193', 'JSL8009', 'JUA2459', 'WCW2531', 'HBB5644', 'BNR2398', 'PNU7796', 'KFF901','JNA2835', 'MCF8493', 'WTM5546',]


## Usage

Install the requirement first 
```bash
# Install the dependencies.
$ pip install -r requirements.txt
```

Linux systems:
```bash
# Move into folder directory
$ cd ALPR

# Give execute permission to main.py
$ chmod +x ./main.py

# Run main.py
$ ./main.py input_video_directory

# Using custom weights and configuration files
$ ./main.py input_video_directory -w weights_file_location -c config_file_location
```

Windows systems (untested):
```bash
# Move into folder directory
$ cd ALPR

# Run main.py
$ python .\main.py -i input_video_directory

# Using custom weights and configuration files
$ .\main.py -i input_video_directory -w weights_file_location -c config_file_location
```
