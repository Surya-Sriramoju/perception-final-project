#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
bridge=CvBridge()

def image_callback(ros_image):
    print("Got an image")
    global bridge
    try:
        cv_image=bridge.imgmsg_to_cv2(ros_image,"bgr8")
    except CvBridgeError as e:
        print(e)
    if cv_image is not None:
        cv2.imshow("Yeh Dekho",cv_image)
        cv2.waitKey(10)

def main():
    rospy.init_node('cam_sub_node',anonymous=True)
    image_sub=rospy.Subscriber('/camera/image_raw',Image, image_callback)
    try:
        rospy.spin()
    except:
        print("Nai hoga humse")
    cv2.destroyAllWindows()
if __name__=='__main__':
    main()
    