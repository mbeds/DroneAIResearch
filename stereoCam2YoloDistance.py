#!/usr/bin/python3

import os

import numpy as np

import cv2

import gi

gi.require_version('Gst', '1.0')

from gi.repository import Gst

from pymavlink import mavutil


MODEL_DIRECTORY = 'path/to/yolo/models'

FOCAL_LENGTH = # focal length of the camera

KNOWN_WIDTH = # width of the object in real life

TARGET_DISTANCE = 10 # distance to fly to in meters


def select_model():

   models = os.listdir(MODEL_DIRECTORY)

   for i, model in enumerate(models):

       print(f'{i+1}. {model}')

   choice = int(input('Select a model by number: '))

   return os.path.join(MODEL_DIRECTORY, models[choice-1])


model_path = select_model()

# Read the model

net = cv2.dnn.readNet(model_path, "cfg/coco.data")


# Initialize Gstreamer

Gst.init(None)


# Create pipeline for stereo camera

pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! videoconvert ! appsink name=sink0 "

                           "v4l2src device=/dev/video1 ! videoconvert ! appsink name=sink1")


# Start pipeline

pipeline.set_state(Gst.State.PLAYING)


# Get camera frames

sink0 = pipeline.get_by_name("sink0")

sink1 = pipeline.get_by_name("sink1")


# Connect to the vehicle

master = mavutil.mavlink_connection('udp:127.0.0.1:14551')


while True:

   # Retrieve left and right frames

   sample0 = sink0.emit("pull-sample")

   sample1 = sink1.emit("pull-sample")

   buffer0 = sample0.get_buffer()

   buffer1 = sample1.get_buffer()

   data0 = buffer0.extract_dup(0, buffer0.get_size())

   data1 = buffer1.extract_dup(0, buffer1.get_size())


   # Convert data to numpy array

   frame0 = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=data0)

   frame1 = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=data1)


   # Perform object detection

   blob = cv2.dnn.blobFromImage(frame0, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

   net.setInput(blob)

   outs = net.forward(getOutputsNames(net))

   #...

   # Estimate distance of detected object

   distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width_of_detected_object

   if distance > TARGET_DISTANCE:

       # Send MAVLink command to fly to target distance

       master.mav.command_long_send(

           master.target_system,

           master.target_component,

           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,

           0, 0, 0, 0, 0, 0,

           TARGET_DISTANCE, 0, 0

       )

   else:

       # Send MAVLink command to follow the object

       master.mav.command_long_send(

           master.target_system,

           master.target_component,

           mavutil.mavlink.MAV_CMD_DO_FOLLOW,

           0, 0, 0, 0, 0, 0,

           width_of_detected_object, 0, 0

       )


   # Show frames

   cv2.imshow("Left Frame", frame0)

   cv2.imshow("Right Frame", frame1)


   if cv2.waitKey(1) & 0xFF == ord('q'):

       break


# Stop pipeline

pipeline.set_state(Gst.State.NULL)


# Close windows

cv2.destroyAllWindows()






