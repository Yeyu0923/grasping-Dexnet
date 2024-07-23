import xml.etree.ElementTree as ET
import glob
import os

file_path = "/media/xfy/Elements/飞宇实验数据备份/01/color_image"
grasp_files = glob.glob(os.path.join(file_path,'*.xml'))

for i in grasp_files:
    tree = ET.parse(i)
    root = tree.getroot()
    output_file = i.replace('xml', 'txt')

    with open(output_file,'w') as f:
        for obj in root.findall('object'):
            cx = obj.find('robndbox/cx').text
            cy = obj.find('robndbox/cy').text
            w = obj.find('robndbox/w').text
            h = obj.find('robndbox/h').text
            angle = obj.find('robndbox/angle').text

            f.write(f"{cx};{cy};{angle};{w};{h}\n")

print("ok")
