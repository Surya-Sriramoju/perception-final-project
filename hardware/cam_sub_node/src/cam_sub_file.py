#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as rosimg
from cv_bridge import CvBridge, CvBridgeError
import cv2
from utils.infer_pi import *
import copy
import time

bridge=CvBridge()
alpha = 0.5
video_output = cv2.VideoWriter('/home/mys/catkin_ws/src/cam_sub_node/src/final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (128*2, 64*2*2))

def image_callback(ros_image):
    global bridge
    global alpha
    # global video_output
    try:
        cv_image=bridge.imgmsg_to_cv2(ros_image,"bgr8")
        

    except CvBridgeError as e:
        print(e)
    image, original_frame = get_tensor(cv_image)
    a = time.time()
    segmentations = predict_img(image)
    b = time.time()
    print("fps: ",1/(b-a))
    segmentations = cv2.normalize(segmentations, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # original_frame = cv2.resize(original_frame, None, fx = 0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    original_frame = cv2.resize(original_frame, (256,128), interpolation=cv2.INTER_AREA)
    real_img = original_frame.copy()
    # print('segmentation shape: ', segmentations.shape)
    # print('frame shape: ', original_frame.shape)
    final = cv2.addWeighted(segmentations, alpha,  original_frame, abs(1-alpha), 0, original_frame)
    final = np.vstack([real_img, final])
    video_output.write(final)
    cv2.imshow("inference",final)
    cv2.waitKey(10)
    

def main():
    rospy.init_node('cam_sub_node',anonymous=True)
    image_sub=rospy.Subscriber('/camera/image_raw',rosimg, image_callback)
    try:
        rospy.spin()
    except:
        print("error")
    video_output.release()
    # cv2.destroyAllWindows()
if __name__=='__main__':
    main()
    