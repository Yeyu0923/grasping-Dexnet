import argparse
import logging
import os


import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
import time
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--data_path', type=str, default='cornell/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--save_path', type=str, default='/media/dmu/Elements/飞宇实验数据备份/实验数据/data_科学会馆_3.0_第二次水下_已标定/1280x720-30cm-有强光-180度/data1/GR-conv_results_pre-process',
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    save_path_input = args.save_path
    # depth_path = args.depth_path
    color_list = os.listdir(data_path+"/color")
    depth_list = os.listdir(data_path+"/depth_tiff")
    color_list.sort(key=lambda x: int(x.split('.')[0]))
    depth_list.sort(key=lambda x: int(x.split('.')[0]))

    for i in range(len(color_list)):
    # Load image
        save_path = save_path_input+"/"+str(i)
        logging.info('Loading image...')
        pic = Image.open(data_path+"/color/"+color_list[i], 'r')
        rgb = np.array(pic)
        pic = Image.open(data_path+"/depth_tiff/"+depth_list[i], 'r')
        # pic = np.load(args.depth_path)

        depth = np.expand_dims(np.array(pic), axis=2)

        start = time.time()
        # Load Network
        logging.info('Loading model...')
        net = torch.load(args.network)
        logging.info('Done')

        # Get the compute device
        start1 = time.time()
        device = get_device(args.force_cpu)
        end1 = time.time()
        print(end1-start1)

        img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

        x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)
            # pred = net.predict(x)


            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            end = time.time()
            print(end-start)
            save_results(
                    rgb_img=img_data.get_rgb(rgb, False),
                    # depth_img=img_data.get_depth(depth),
                    depth_img=np.squeeze(img_data.get_depth(depth)),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=args.n_grasps,
                    grasp_width_img=width_img,
                    save_path=save_path
            )
        save_path = save_path_input
            # else:
            #     fig = plt.figure(figsize=(10, 10))
            #     plot_results(fig=fig,
            #                 rgb_img=img_data.get_rgb(rgb, False),
            #                 grasp_q_img=q_img,
            #                 grasp_angle_img=ang_img,
            #                 no_grasps=args.n_grasps,
            #                 grasp_width_img=width_img)
            #     fig.savefig('img_result.pdf')
