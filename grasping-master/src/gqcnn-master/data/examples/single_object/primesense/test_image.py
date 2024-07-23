from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import cv2

# im = Image.open("color_0.png")
# img = np.array(im)      # image类 转 numpy
# # ac = np.load('depth_0.npy')
# print(img)
# print(img.shape)
# img = img[:,:,1]        #第1通道
# # img = img.astype(np.float32)
# img = np.array(Image.fromarray(img.astype(np.uint8)).resize((256, 256)))
# print(img)
# print(img.shape)
# np.save("color.npy",img)


# im = Image.open("color_0.png")
# im.show() 


# # np.save("color.npy",im)
# img = np.array(im)      # image类 转 numpy
# img = img[:,:,0]        #第1通道
# img = img.astype(np.float32)
# plt.imshow(im)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.savefig('depthma.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.show()     
# im=Image.fromarray(img) # numpy 转 image类
# im.show() 


# # print(m)
# # m = Image.fromarray(np.uint8(m.transpose(1,2,0)))
# # m.show()




np.set_printoptions(threshold=np.inf)

m = np.load("depth_0.npy")
# m = m.resize((256, 256,1))
print(m)
print(m.shape)
m = m[120:-120, 210:-210,0]
# m = np.array(m.resize((480, 640,1)))
print(m)
print(m.shape)
np.save("color.npy",m)
# plt.imshow(m)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.show()     