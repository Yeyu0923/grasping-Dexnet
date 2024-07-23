import numpy as np
import torch
import sys
import cv2
import skimage.transform as skt
sys.path.append('/home/dlmux/Perception/Dex-Net/grasping-master/src/gqcnn-master/roboticgrasping/utils' )
from dataset_processing import image
# from utils.dataset_processing import image


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self,
                 width=848,
                 height=480,
                 output_size_height = 480,
                 output_size_width = 848,
                 include_depth=True,
                 include_rgb=True
                 ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """
        self.output_size_height = output_size_height
        self.output_size_width = output_size_width
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size_width) // 2
        top = (height - output_size_height) // 2
        right = (width + output_size_width) // 2
        bottom = (height + output_size_height) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    def get_depth_number(self, img, boundingbox=None):
        depth_img = image.Image(img)
        if boundingbox:
            self.get_boundingbox(boundingbox)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        exist = (depth_img.img != 0)
        mean_value = depth_img.img.sum()/exist.sum()
        return mean_value

    def imresize(image, size, interp="nearest"):

        skt_interp_map = {
            "nearest": 0,
            "bilinear": 1,
            "biquadratic": 2,
            "bicubic": 3,
            "biquartic": 4,
            "biquintic": 5,
        }
        if interp in ("lanczos", "cubic"):
            raise ValueError(
                '"lanczos" and "cubic"' " interpolation are no longer supported."
            )
        assert (
            interp in skt_interp_map
        ), 'Interpolation "{}" not' " supported.".format(interp)

        if isinstance(size, (tuple, list)):
            output_shape = size
        elif isinstance(size, (float)):
            np_shape = np.asarray(image.shape).astype(np.float32)
            np_shape[0:2] *= size
            output_shape = tuple(np_shape.astype(int))
        elif isinstance(size, (int)):
            np_shape = np.asarray(image.shape).astype(np.float32)
            np_shape[0:2] *= size / 100.0
            output_shape = tuple(np_shape.astype(int))
        else:
            raise ValueError('Invalid type for size "{}".'.format(type(size)))

        return skt.resize(
            image.astype(np.float),
            output_shape,
            order=skt_interp_map[interp],
            anti_aliasing=False,
            mode="constant",
        )

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def get_boundingbox(self, boundingbox):
        ymax = boundingbox.ymax+5
        xmax = boundingbox.xmax+5
        ymin = boundingbox.ymin-5
        xmin = boundingbox.xmin-5
        self.bottom_right = (ymax, xmax)
        self.top_left = (ymin, xmin)

    def get_depth(self, img, boundingbox=None):
        depth_img = image.Image(img)
        np.set_printoptions(threshold=np.inf)
        
        # depth_img.make_boundingbox(boundingbox=boundingbox,depth=True)
        if boundingbox:
            self.get_boundingbox(boundingbox)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        # depth_img.img = depth_img.img.astype(np.float32)/1000

        # resized_data = self.imresize(depth_img.img, size, interp="nearest").astype(
        #     np.uint8
        # )

        # mask = 1 * (np.sum(resized_data, axis=2) == 0)
        # impaint_im = cv2.inpaint(
        #     resized_data, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA
        # )
        # new_img = depth_img.img
        # new_img[depth_img.img == 0] = impaint_im[depth_img.img == 0]
        # depth_img.img = new_img

        depth_img.img[depth_img.img==0] = 500
        depth_img.img[depth_img.img>500] = 500
        depth_img.img = depth_img.img.astype(np.float32) / 500.0

        depth_img.img -= depth_img.img.mean()
        # depth_img.resize((self.output_size, self.output_size))
        depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img

    def get_rgb(self, img, boundingbox=None, norm=True):
        rgb_img = image.Image(img)
        # rgb_img.make_boundingbox(boundingbox=boundingbox)
        if boundingbox:
            self.get_boundingbox(boundingbox)
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        # rgb_img.resize((self.output_size, self.output_size))
        if norm:
                rgb_img.normalise()
                rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_data(self,boundingbox=None ,rgb=None, depth=None):
        depth_img = None
        rgb_img = None
        
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(img=depth, boundingbox=boundingbox)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb, boundingbox=boundingbox)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         np.expand_dims(rgb_img, 0)),
                        1
                    )
                )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(np.expand_dims(rgb_img, 0))

        return x, depth_img, rgb_img
    
    def average_depth(self,boundingbox,depth):
        mean_value = self.get_depth_number(img=depth, boundingbox=boundingbox)
        return mean_value
