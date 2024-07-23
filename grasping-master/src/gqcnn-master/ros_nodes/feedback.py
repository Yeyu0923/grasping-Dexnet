import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

def feedback(yolo):
    color_image = rospy.wait_for_message("/zedxm/zed_node/left/image_rect_color",Image)
    depth_image = rospy.wait_for_message("/zedxm/zed_node/depth/depth_registered",Image)
    cv_bridge = CvBridge()
    color_image = cv_bridge.imgmsg_to_cv2(color_image,"bgr8")
    depth_image = cv_bridge.imgmsg_to_cv2(depth_image)
    Xmin = 0
    Xmax = 1280
    Ymin = 0
    Ymax = 720
    count = 0
    number = 0 
    outputs, img_info = yolo.predictor.inference(color_image)
    try:
        result_img_rgb, bboxes, scores, cls, cls_names = yolo.predictor.visual(outputs[0], img_info)
        cv2.imwrite("/home/dlmux/Perception/Dex-Net/grasping-master/src/gqcnn-master/feed_back_data/feedback.png",result_img_rgb)
    except Exception as e:
        print(e)
        print("no object")
        return False
    for bbox in bboxes:
        if Xmin < bbox[0]:
            number = number + 1
        if Ymin < bbox[1]:
           number = number + 1
        if Xmax > bbox[2]:
           number = number + 1
        if Ymax > bbox[3]:
           number = number + 1
        if number > 1:
           count = count + 1
        number = 0
    if count > 0:
            return True
    return False    

if __name__ == "__main__":
    feedback()
