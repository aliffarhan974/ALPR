# ALPR
ALPR systems using YOLOv4. By default, the system will first check if OpenCV has CUDA and if does, GPU will be used.

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
$ ./main.py -i input_video_directory

# Using custom weights and configuration files
$ ./main.py -i input_video_directory -w weights_file_location -c config_file_location
```

Windows systems (untested):
```bash
# Move into folder directory
$ cd ALPR

# Run main.py
$ python .\main.py -i input_video_directory

# Using custom weights and configuration files
$ ./main.py -i input_video_directory -w weights_file_location -c config_file_location
```
