import argparse
import json
import os
import time
import sys
sys.path.append("/home/xfy/gqcnn_ws/src/gqcnn-master")
import numpy as np
import matplotlib.pyplot as plt
from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode

# Set up logger.
logger = Logger.get_logger("examples/policy.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Run a grasping policy on an example image")
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of a trained model to run")
    parser.add_argument(
        "--depth_image",
        type=str,
        default=None,
        help="path to a test depth image stored as a .npy file")
    parser.add_argument("--segmask",
                        type=str,
                        default=None,
                        help="path to an optional segmask to use")
    parser.add_argument("--camera_intr",
                        type=str,
                        default=None,
                        help="path to the camera intrinsics")
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to the folder in which the model is stored")
    parser.add_argument("--config_filename",
                        type=str,
                        default=None,
                        help="path to configuration file to use")
    parser.add_argument(
        "--fully_conv",
        action="store_true",
        help=("run Fully-Convolutional GQ-CNN policy instead of standard"
              " GQ-CNN policy"))
    args = parser.parse_args()
    model_name = args.model_name
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intr
    model_dir = args.model_dir
    config_filename = args.config_filename
    fully_conv = args.fully_conv

    assert not (fully_conv and depth_im_filename is not None
                and segmask_filename is None
                ), "Fully-Convolutional policy expects a segmask."

    if depth_im_filename is None:
        if fully_conv:
            depth_im_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "data/examples/clutter/primesense/depth_0.npy")
        else:
            depth_im_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "data/examples/single_object/primesense/depth_0.npy")
    if fully_conv and segmask_filename is None:
        segmask_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/examples/clutter/primesense/segmask_0.png")
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/calib/primesense/primesense.intr")

    # Set model if provided.
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_path = os.path.join(model_dir, model_name)

    # Get configs.1
    model_config = json.load(open(os.path.join(model_path, "config.json"),
                                  "r"))
    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))

    # Set config.
    if config_filename is None:
        if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
                or gripper_mode == GripperMode.PARALLEL_JAW):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_pj.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_pj.yaml")
        elif (gripper_mode == GripperMode.LEGACY_SUCTION
              or gripper_mode == GripperMode.SUCTION):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_suction.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_suction.yaml")

    # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    depth_data = np.load(depth_im_filename)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Optionally read a segmask.
    # segmask = None
    # valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    # if segmask is None:
    #     segmask = valid_px_mask
    # else:
    #     segmask = segmask.mask_binary(valid_px_mask)
    
    segmask = BinaryImage(255 *
                                  np.ones(depth_im.shape).astype(np.uint8),
                                  frame=color_im.frame)

    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)

    rgbd_state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    vis.figure(size=(10, 10))
    num_plot = 1
    if segmask is not None:
        num_plot = 2
    vis.subplot(1, num_plot, 1)
    vis.imshow(depth_im)
    if segmask is not None:
        vis.subplot(1, num_plot, 2)
        vis.imshow(segmask)
        
    vis.show()
    vis.imshow(rgbd_im)
    vis.show()

    # plt.imshow(image)
    # plt.savefig('mask.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
    # plt.show()     
