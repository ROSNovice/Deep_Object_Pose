#!/usr/bin/env python

from __future__ import print_function
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Image_msg
import cv2
import argparse
import glob
import os
import rospkg

def image_read(image_list, freq = 1):

    rospy.init_node('image_read', anonymous=True)
    image_out = rospy.Publisher("image_read", Image_msg, queue_size=1)
    rate = rospy.Rate(freq)

    if image_list is None:
        print("Failed to read the image!")
        exit(1)
    else:
        print("Succeed to read the image!")
        print("Ctrl-C to stop")
        #cv2.imshow("image",image)
        #cv2.waitKey(0)

    while not rospy.is_shutdown():
        for image_name in image_list:  
            image = cv2.imread(image_name,1)
            image = CvBridge().cv2_to_imgmsg(image,"bgr8")
            image_out.publish(image)
            rate.sleep()

if __name__ == "__main__":
    #default folder path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('dope')
    image_path = package_path + "/images/" 
    #add command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path",
        default=image_path,
        help = "add the folder path to imamges")
    opt, remaining_args = parser.parse_known_args()
    # load images
    image_list = sorted(glob.glob(os.path.join(opt.path, '*.jpg'))) + \
                 sorted(glob.glob(os.path.join(opt.path, '*.png')))

    try:
        image_read(image_list)
    except rospy.ROSInterruptException as e:
        print("Error:",e)
         
