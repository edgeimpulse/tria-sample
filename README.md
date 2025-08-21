# tria-sample

Best if tested on a RPi5 running RaspberryPiOS.

## Installation steps:

Follow the instructions https://github.com/edgeimpulse/linux-sdk-python in the README to install the SDK first (clone, then read install instructions in the README.md file)

You also need to install our SDK package for Python: https://pypi.org/project/edge-impulse-linux

main.py is based on this code example https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-image.py 

## Running steps:

Run "v4l2-ctl --list-devices" on the RPi5 to determine which /dev/videoX file is your camera (.. i.e. /dev/video1 for example)

Download a eim file containing your model/impulse you want to run

Invoke "python3 ./main.py $HOME/myModelFile.eim 3"

where: "myModelFile.eim" is replaced with the name of your model file and "3" is the number representing the camera you discovered above...
