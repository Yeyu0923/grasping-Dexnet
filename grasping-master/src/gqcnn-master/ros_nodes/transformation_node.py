#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import yaml
import types
# sys.path.append("/home/dmu/WorkSpace/ocean_rotor_x/src/planning")
sys.path.append("/home/dlmux/Perception/Dex-Net/grasping-master/src/YOLOX-ROS/yolox_ros_py")
from feedback import feedback
from scripts import yolox_ros
# from planning.srv import SetMode, SetModeRequest
import torch
import torch.backends.cudnn as cudnn
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger, vis
import threading
import json
import math
import os
import time
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image as imgss
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from autolab_core import (Point, YamlConfig, Logger,CameraIntrinsics, ColorImage,
                          DepthImage, BinaryImage, RgbdImage)
from visualization import Visualizer2D as vis2
from gqcnn.grasping import (Grasp2D, SuctionPoint2D, RgbdImageState,
                            RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction,
                            GraspAction)
from gqcnn.utils import GripperMode, NoValidGraspsException
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header, Int16
from gqcnn.srv import (GQCNNGraspPlanner, GQCNNGraspPlannerBoundingBox,GQCNNGraspPlannerRequest,GQCNNGraspPlannerBoundingBoxResponse,GQCNNGraspPlannerBoundingBoxRequest,
                       GQCNNGraspPlannerSegmask)
from gqcnn.msg import GQCNNGrasp
from bboxes_ex_msgs.msg import BoundingBoxes, GRconvnet
import cv2
# import pyrealsense2 as rs
from sensor_msgs.msg import Image,CameraInfo
from rospy.numpy_msg import numpy_msg
import message_filters
import io
import argparse
import logging
import math
import numpy as np
import torch.utils.data
import time
# from roboticgrasping.hardware.camera import RealSenseCamera
# from roboticgrasping.hardware.device import get_device
# from roboticgrasping.inference.post_process import post_process_output
# from roboticgrasping.utils.data.camera_data import CameraData
# from roboticgrasping.utils.visualisation.plot import save_results, plot_results
sys.path.append("/home/dlmux/Perception/Dex-Net/grasping-master/src/gqcnn-master/roboticgrasping")

class yolox_ros():
    def __init__(self) -> None:

        # ROS1 init
        # self.setting_yolox_exp()

        # self.bridge = CvBridge()

        # self.client_state = rospy.ServiceProxy("set_mode",SetMode)

        self.pub = rospy.Publisher('yolox/bounding_boxes', BoundingBoxes,queue_size=1)
        self.pub_image = rospy.Publisher("yolox/image_raw",Image,queue_size=1)
        # rospy.Subscriber("/d405/color/image_raw",Image,self.imageflow_callback,queue_size=1)
        # rospy.Subscriber("/d405/color/camera_info",CameraInfo,self.wide,queue_size=1)

        # rospy.spin()

    def setting_yolox_exp(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        WEIGHTS_PATH = '../../weights/yolox_s.pth'

        # =============================================================
        self.imshow_isshow = rospy.get_param('imshow_isshow', True)

        yolo_type = rospy.get_param('~yolo_type', 'yolox-s')
        fuse = rospy.get_param('~fuse', False)
        trt = rospy.get_param('~trt', False)
        rank = rospy.get_param('~rank', 0)
        ckpt_file = rospy.get_param('~ckpt_file', WEIGHTS_PATH)
        conf = rospy.get_param('~conf', 0.3)
        nmsthre = rospy.get_param('~nmsthre', 0.65)
        img_size = rospy.get_param('~img_size', 640)
        self.input_width = rospy.get_param('~image_size/width', 360)
        self.input_height = rospy.get_param('~image_size/height', 240)

        # ==============================================================

        cudnn.benchmark = True

        exp = get_exp(None, yolo_type)

        BASE_PATH = "/home/dlmux/Perception/Dex-Net/grasping-master/src/YOLOX-ROS/yolox_ros_py/scripts"
        file_name = os.path.join(BASE_PATH, "YOLOX_PATH/")
        # os.makedirs(file_name, exist_ok=True)

        exp.test_conf = conf # test conf
        exp.nmsthre = nmsthre # nms threshold
        exp.test_size = (img_size, img_size) # Resize size

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()

        if not trt:
            logger.info("loading checkpoint")
            loc = "cuda:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc)
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        # TensorRT
        if trt:
            assert (not fuse),\
                "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(trt_file), (
                "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            )
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        self.predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder)

    def wide(self,msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.K = msg.K
        
    def pixel_to_camera(self,pixel_coord1, camera_matrix):
        pixel_homogeneous = np.array([pixel_coord1[0],pixel_coord1[1],1.0])
        
        camera_coord_homogeneous = np.linalg.inv(camera_matrix) @ pixel_homogeneous.T
        camera_coord = camera_coord_homogeneous[:2]/camera_coord_homogeneous[2]
        return camera_coord

        
    
    def distance(self,h):
        real_wide = 0.11
        dis_inch = (self.fx * real_wide) / h
        return dis_inch

    def yolox2bboxes_msgs(self, bboxes, scores, cls, cls_names):
        bboxes_msg = BoundingBoxes()
        i = 0
        for bbox in bboxes:
                one_box = BoundingBox()
                one_box.xmin = int(bbox[0])
                one_box.ymin = int(bbox[1])
                one_box.xmax = int(bbox[2])
                one_box.ymax = int(bbox[3])
                one_box.probability = float(scores[i])
                one_box.class_id = str(cls_names[int(cls[i])])
                wide = bbox[2]-bbox[0]
                high = bbox[3]-bbox[1]
                distance_real = self.distance(wide)
                one_box.z = distance_real
                x_px = bbox[0]+(wide/2)
                y_px = bbox[1]+(high/2)
                pixel_coord1 = np.array([x_px,y_px])
                camera_matrix = np.array([[self.K[0],self.K[1],self.K[2]],[self.K[3],self.K[4],self.K[5]],[self.K[6],self.K[7],self.K[8]]])
                camera_coord = self.pixel_to_camera(pixel_coord1,camera_matrix)
                one_box.x = camera_coord[0]*distance_real
                one_box.y = camera_coord[1]*distance_real
                bboxes_msg.bounding_boxes.append(one_box)
                i = i+1
        
        return bboxes_msg

    def imageflow_callback(self,msg:Image) -> None:
        try:
            img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            img_rgb = cv2.resize(img_rgb,(self.input_width,self.input_height))

            outputs, img_info = self.predictor.inference(img_rgb)

            try:
                result_img_rgb, bboxes, scores, cls, cls_names = self.predictor.visual(outputs[0], img_info)
                bboxes = self.yolox2bboxes_msgs(bboxes, scores, cls, cls_names, msg.header)

                self.pub.publish(bboxes)
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_rgb,"bgr8"))
                # req_state = SetModeRequest()
                # pose = PoseStamped()
                # req_state.mode = "task"
                # pose.pose.position.x = bboxes.bounding_boxes.x
                # pose.pose.position.y = bboxes.bounding_boxes.y
                # pose.pose.position.z = bboxes.bounding_boxes.z
                # req_state.goal = pose
                # self.client_state.call(req_state)

                if (self.imshow_isshow):
                    cv2.imshow("YOLOX", result_img_rgb)
                    cv2.waitKey(10)
            except:
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",img_rgb)
                    cv2.waitKey(10)

        except:
            pass
    # def imageflow_callback(self,msg:Image) -> None:
    #     img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
    #     img_rgb = cv2.resize(img_rgb,(self.input_width,self.input_height))

    #     outputs, img_info = self.predictor.inference(img_rgb)

            
    #     result_img_rgb, bboxes, scores, cls, cls_names = self.predictor.visual(outputs[0], img_info)
    #     bboxes = self.yolox2bboxes_msgs(bboxes, scores, cls, cls_names, msg.header)

    #     self.pub.publish(bboxes)
    #     self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_rgb,"bgr8"))

    #     if (self.imshow_isshow):
    #         cv2.imshow("YOLOX", result_img_rgb)
    #         cv2.waitKey(10)





class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = img

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img, ratio = preproc(img, self.test_size)
        img_info['ratio'] = ratio
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info['ratio']
        img = img_info['raw_img']
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, bboxes, scores, cls, self.cls_names

def callback1(camerainfo1):
        global camerainfo
        camerainfo = camerainfo1      


#输入的Rx, Ry, Rz为角度
def Rxyz_to_Quaternion(Rx, Ry, Rz):
    X, Y, Z = Rx/180 * math.pi,Ry/180 * math.pi,Rz/180 * math.pi
    Qx = math.cos(Y/2)*math.cos(Z/2)*math.sin(X/2)-math.sin(Y/2)*math.sin(Z/2)*math.cos(X/2)                  
    Qy = math.sin(Y/2)*math.cos(Z/2)*math.cos(X/2)+math.cos(Y/2)*math.sin(Z/2)*math.sin(X/2)
    Qz = math.cos(Y/2)*math.sin(Z/2)*math.cos(X/2)-math.sin(Y/2)*math.cos(Z/2)*math.sin(X/2)
    QW = math.cos(Y/2)*math.cos(Z/2)*math.cos(X/2)+math.sin(Y/2)*math.sin(Z/2)*math.sin(X/2)
    print(Qx,Qy,Qz,QW)
    return Qx,Qy,Qz,QW
  

# def GR_ConvNet_callback(colorimage, depthimage, camerainfo, boundingbox):

#         start_time = time.time()
#         model_path = '/home/xfy/Dex-Net/grasping-master/src/gqcnn-master/roboticgrasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98' 

#         cam_data = CameraData(include_depth=1, include_rgb=1)
       
       
#         net = torch.load(model_path)

       
#         device = get_device(False)
       
#         # try:
#         fig = plt.figure(figsize=(10, 10))
#         cv_bridge = CvBridge()
#         depthimage = cv_bridge.imgmsg_to_cv2(depthimage, desired_encoding="passthrough").astype(np.float32)/1000
#         colorimage = cv_bridge.imgmsg_to_cv2(colorimage,desired_encoding="passthrough")
#         depthimage = np.expand_dims(depthimage,axis=2)
#         rgb = colorimage
#         depth = depthimage
#         camerainfo = camerainfo
#         # try
#         # print(len(boundingbox.bounding_boxes))
#         if boundingbox:
#                 l = len(boundingbox.bounding_boxes)
#                 GR_pose = GRconvnet()
#                 for i in range(l):
#                         x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth, boundingbox=boundingbox.bounding_boxes[i])
#                         rospy.loginfo("ok")
#                         with torch.no_grad():
#                                 xc = x.to(device)
#                                 pred = net.predict(xc)
                                
#                                 q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
#                                 grasps = save_results(
#                                         rgb_img=cam_data.get_rgb(rgb, boundingbox=boundingbox.bounding_boxes[i],norm=False),
#                                         depth_img=np.squeeze(cam_data.get_depth(depth,boundingbox=boundingbox.bounding_boxes[i])),
#                                         grasp_q_img=q_img,
#                                         grasp_angle_img=ang_img,
#                                         no_grasps=1,
#                                         grasp_width_img=width_img
#                                         )
#                                 if grasps:
#                                         pos_z = depth[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]]
#                                         pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - camerainfo.K[2],
#                                                         pos_z / camerainfo.K[0])
#                                         pos_y = np.multiply(grasps[0].center[0] + cam_data.top_left[0] - camerainfo.K[5],
#                                                         pos_z / camerainfo.K[4])
#                                         # 单位为米
#                                         pose = PoseStamped()
#                                         pose.pose.position.x = pos_x
#                                         pose.pose.position.y = pos_y
#                                         pose.pose.position.z = pos_z
#                                         Qx,Qy,Qz,Qw = Rxyz_to_Quaternion(0,0,grasps[0].angle)
#                                         pose.pose.orientation.x = Qx
#                                         pose.pose.orientation.y = Qy
#                                         pose.pose.orientation.z = Qz
#                                         pose.pose.orientation.w = Qw
#                                         GR_pose.poses.append(pose)
#                                         width = grasps[0].length
#                                         GR_pose.width.append(width)
#                                         quality = grasps[0].quality
#                                         GR_pose.q_value.append(quality)
#                 pub = rospy.Publisher("/GR_convNet/grasp_pose",GRconvnet,queue_size=1)
#                 pub.publish(GR_pose)
#                 end_time = time.time()
#                 print(end_time-start_time)
                

def gqcnn_callback(colorimage,depthimage,camerainfo, boundingbox=None):
# def callback(colorimage,depthimage):
        global yolo
        global get_it
        rospy.loginfo("ok")
        cv_bridge = CvBridge()
                # global timeout_timer
                # timeout_timer.shutdown()
        # cam_data = CameraData(include_depth=1, include_rgb=1)
        yolo.wide(camerainfo)
        colorimage = cv_bridge.imgmsg_to_cv2(colorimage,"bgr8")
        depthimage = cv_bridge.imgmsg_to_cv2(depthimage)
        # colorimage = colorimage[0:320, 313:651,:]
        # depthimage = depthimage[0:320, 313:651]
        img_rgb = cv2.resize(colorimage,(yolo.input_width,yolo.input_height))

        outputs, img_info = yolo.predictor.inference(img_rgb)
        bboxes = None
        try:
            result_img_rgb, bboxes, scores, cls, cls_names = yolo.predictor.visual(outputs[0], img_info)
            bboxes = yolo.yolox2bboxes_msgs(bboxes, scores, cls, cls_names)
            l = len(bboxes.bounding_boxes)
                # if l > 2:
                    # req_state = SetModeRequest()
                    # pose = PoseStamped()
                    # req_state.mode = "task"
                    # pose.pose.position.x = bboxes.bounding_boxes[0].x
                    # pose.pose.position.y = bboxes.bounding_boxes[0].y
                    # pose.pose.position.z = bboxes.bounding_boxes[0].z
                    # pose.pose.orientation.w = 1
                    # pose.pose.orientation.x = 0
                    # pose.pose.orientation.y = 0
                    # pose.pose.orientation.z = 0
                    # req_state.goal = pose
                    # yolo.client_state.call(req_state)
                # if (yolo.imshow_isshow):
                #         cv2.imshow("YOLOX", result_img_rgb)
                #         cv2.waitKey(10)
                # yolo.pub_image.publish(cv_bridge.cv2_to_imgmsg(result_img_rgb))
            cv2.imwrite("/home/dlmux/yolo_result.png",result_img_rgb)
        except:
                # if (yolo.imshow_isshow):
                #     cv2.imshow("YOLOX",img_rgb)
                #     cv2.waitKey(10)    
                # yolo.pub_image.publish(cv_bridge.cv2_to_imgmsg(img_rgb))
                pass
        # except:

        if bboxes:
                # if get_it > 0:
                #     if if_boundingbox(bboxes):
                #         get_it = get_it + 1
                #         if get_it > 4:
                #             req_state.mode = "succesd"
                #             yolo.client_stste.call(req_state)
                #             get_it = 0 
                #         return
                #     else :
                #         req_state.mode = "false"
                #         yolo.client_stste.call(req_state)
                #         get_it = 0
                l = len(bboxes.bounding_boxes)
                thresh = 0.2
                # for i in range(l):
                #         depthimage1 = depthimage.astype(np.float32)/1000
                #         depthimage1 = np.expand_dims(depthimage1,axis=2)
                #         mean_value = cam_data.average_depth(depth=depthimage1, boundingbox=bboxes.bounding_boxes[i])
                #         if mean_value < thresh:
                #                 thresh = mean_value

                if thresh < 0.4:
                        # client_stste = rospy.ServiceProxy("set_mode",SetMode)
                        # req_state = SetModeRequest()
                        client = rospy.ServiceProxy("gqcnn/grasp_planner_bounding_box",GQCNNGraspPlannerBoundingBox)
                        rospy.loginfo("开始等待")
                        client.wait_for_service()
                        req = GQCNNGraspPlannerBoundingBoxRequest()
                        ros_cim = cv_bridge.cv2_to_imgmsg(colorimage,"bgr8")
                        ros_dim = cv_bridge.cv2_to_imgmsg(depthimage)
                        req.color_image = ros_cim
                        req.depth_image = ros_dim
                        req.camera_info = camerainfo
                        req.bounding_box.minX = bboxes.bounding_boxes[0].xmin
                        req.bounding_box.minY = bboxes.bounding_boxes[0].ymin
                        req.bounding_box.maxX = bboxes.bounding_boxes[0].xmax
                        req.bounding_box.maxY = bboxes.bounding_boxes[0].ymax
                        rospy.loginfo("筹备数据")
                        raw_camera_info = camerainfo
                        camera_intr = CameraIntrinsics(
                        raw_camera_info.header.frame_id, raw_camera_info.K[0],
                        raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5],
                        raw_camera_info.K[1], raw_camera_info.height,
                        raw_camera_info.width)
                        depth_im = DepthImage(depthimage.astype(np.float32),
                                                frame=camera_intr.frame)
                        depth_im = depth_im.inpaint(rescale_factor=0.5)
                        color_im = ColorImage(colorimage,
                                        frame=camera_intr.frame)
                        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
                        # 获取抓取grasp
                        try:
                                resp = client.call(req)
                                rospy.loginfo(resp)
                                # rospy.loginfo("ooooooooooooooooooooooooo")
                                grasp = resp.grasp
                                # Convert to a grasp action.
                                grasp_type = grasp.grasp_type
                                if grasp_type == GQCNNGrasp.PARALLEL_JAW:
                                        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                                                frame=camera_intr.frame)
                                        grasp_2d = Grasp2D(center,
                                                        grasp.angle,
                                                        grasp.depth,
                                                        width=0.11,
                                                        # grasp.width,
                                                        camera_intr=camera_intr)
                                elif grasp_type == GQCNNGrasp.SUCTION:
                                        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                                                frame=camera_intr.frame)
                                        grasp_2d = SuctionPoint2D(center,
                                                                np.array([0, 0, 1]),
                                                                grasp.depth,
                                                                camera_intr=camera_intr)
                                else:
                                        raise ValueError("Grasp type %d not recognized!" % (grasp_type))
                                try:
                                        thumbnail = DepthImage(cv_bridge.imgmsg_to_cv2(
                                        grasp.thumbnail, desired_encoding="passthrough"),
                                                        frame=camera_intr.frame)
                                except CvBridgeError as e:
                                        rospy.loginfo(e)
                                action = GraspAction(grasp_2d, grasp.q_value, thumbnail)
                                # req_state.mode = "grasp"
                                pose_res = PoseStamped()
                                pose_res.pose.position.x = action.grasp.center.x
                                pose_res.pose.position.y = action.grasp.center.y
                                pose_res.pose.position.z = action.grasp.depth
                                pose_res.pose.orientation.w = 1
                                pose_res.pose.orientation.x = 0
                                pose_res.pose.orientation.y = 0
                                pose_res.pose.orientation.z = 0
                                # req_state.goal = pose_res
                                # client_stste.call(req_state)
                                grasp_res = rospy.Publisher("grasp",PoseStamped,queue_size=1)
                                grasp_res.publish(pose_res)

                                # Vis final grasp.
                                # vis.figure(size=(10, 10))
                                # vis.imshow(depth_im, vmin=0.6, vmax=0.9)
                                # vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
                                # vis.title("Planned grasp on depth (Q=%.3f)" % (action.q_value))
                                # vis.show()
                                # vis.savefig("/home/dmu/result.png")
                                
                                res_image = rospy.Publisher("res_image",Image,queue_size=1)
                                res_image.publish(grasp.thumbnail)

                                # 等待抓取完成信号
                                # data = rospy.wait_for_message("",,timeout=None)

                                # 设置抓取判断为1，下次循环进行
                                get_it = 1

                        except rospy.ServiceException as exception:
                                rospy.loginfo(exception)

def if_boundingbox(if_data):
       Xmin = 0
       Xmax = 1280
       Ymin = 0
       Ymax = 720
       count = 0
       number = 0 
       l = len(if_data.bounding_boxes)
       for i in range(l):
              if Xmin < if_data.bounding_boxes[i].xmin:
                     number = number + 1
              if Ymin < if_data.bounding_boxes[i].ymin:
                     number = number + 1
              if Xmax > if_data.bounding_boxes[i].xmax:
                     number = number + 1
              if Ymax > if_data.bounding_boxes[i].ymax:
                     number = number + 1
              if number > 1:
                     count = count + 1
              number = 0
       if count > 0:
              return True
       return False    

if __name__ == "__main__":
    
    # Set up logger.
    logger = Logger.get_logger("ros_nodes/transformation_node.py")
    # Initialize the ROS node.

    rospy.init_node("Transformation")
    yolo = yolox_ros()
        # try:
    yolo.setting_yolox_exp()
    absolute = rospy.get_param("~absolute", False)
    bufsize = rospy.get_param("~bufsize", 100)
    show_framerate = rospy.get_param("~show_framerate", True)
    grasp_mothed = rospy.get_param("~grasp_mothed","gqcnn")

    colorimage = message_filters.Subscriber("/zedxm/zed_node/left/image_rect_color", Image)
    depthimage = message_filters.Subscriber("/zedxm/zed_node/depth/depth_registered",Image)
    camerainfo = message_filters.Subscriber("/zedxm/zed_node/depth/camera_info",CameraInfo)

    rospy.loginfo("收到数据")
    ts = message_filters.ApproximateTimeSynchronizer([colorimage, depthimage, camerainfo],queue_size=1,slop=0.1, allow_headerless=True)

    # a = feedback(yolo)
    # print("result:"+str(a))

    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_device(str(218622274523))
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # # Start streaming
    # cfg = pipeline.start(config)
    # #深度图像向彩色对齐
    # align_to_color=rs.align(rs.stream.color)
 
    # try:
    #     while True:
    #         # Wait for a coherent pair of frames: depth and color
    #         frames = pipeline.wait_for_frames()
            
    #         frames = align_to_color.process(frames)
 
    #         depth_frame = frames.get_depth_frame()
    #         color_frame = frames.get_color_frame()
    #         if not depth_frame or not color_frame:
    #             continue
    #         # Convert images to numpy arrays
    #         depth_image = np.asanyarray(depth_frame.get_data())
    #         color_image = np.asanyarray(color_frame.get_data())
    #         with open('/home/dlmux/Perception/Dex-Net/grasping-master/src/gqcnn-master/data/calib/realsense/intr.yaml', 'r') as f:
    #             data = yaml.safe_load(f)
    #         camerainfo_intr = CameraInfo()
    #         camerainfo_intr.width = data["width"]
    #         camerainfo_intr.height = data["height"]
    #         camerainfo_intr.K = data["K"]
    #         camerainfo_intr.header.frame_id = data["header"]["frame_id"]
    #         camerainfo_intr.header.seq = 1
    #         camerainfo_intr.header.stamp.nsecs = 762873650
    #         camerainfo_intr.header.stamp.secs = 1691131843
    #         # config = YamlConfig("/home/dlmux/Perception/ros1_ws/src/YOLOX-ROS/yolox_ros_py/scripts/intr.yaml")
    #         # intr = config['config']
    #         # intrs = intr['K']

    #         get_it = 0
    #         print("得到")
    #         gqcnn_callback(color_image,depth_image,camerainfo_intr)
    # finally:
    #     # Stop streaming
    #     pipeline.stop()

    print(grasp_mothed)
    # if grasp_mothed == "gqcnn":
    #     ts.registerCallback(gqcnn_callback)
    # elif grasp_mothed == "GR-ConvNet":
    #     ts.registerCallback(GR_ConvNet_callback)
    ts.registerCallback(gqcnn_callback)
    rospy.spin()