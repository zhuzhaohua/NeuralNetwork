import scipy.misc
import scipy.ndimage
import glob
import numpy
import matplotlib.pyplot

our_own_dataset = []

# 循环打开文件
for image_file_name in glob.glob('./my_own_images/2828_my_own_?.png'):
    print("loading ... ", image_file_name)
    # 截取文件名中的数据，是图片所表示的数字
    label = int(image_file_name[-5:-4])
    # 解析图片 生成表示灰度的向量
    img_array = scipy.misc.imread(image_file_name, flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    # 数据标准化
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # print(numpy.min(img_data))
    # print(numpy.max(img_data))
    # 合并为测试用数组
    record = numpy.append(label, img_data)
    our_own_dataset.append(record)
    pass

# 查看第1个图像
new_data = our_own_dataset[0][1:].reshape(28, 28)

# 分别左右旋转10度
inputs_plus10_img = scipy.ndimage.rotate(new_data, 10.0, cval=0.01, order=1, reshape=False)
inputs_minus10_img = scipy.ndimage.rotate(new_data, -10.0, cval=0.01, order=1, reshape=False)


# 分别加载图形
matplotlib.pyplot.imshow(new_data, cmap='Greys', interpolation='None')
# matplotlib.pyplot.imshow(inputs_plus10_img, cmap='Greys', interpolation='None')
# matplotlib.pyplot.imshow(inputs_minus10_img, cmap='Greys', interpolation='None')
#
# 显示图形
matplotlib.pyplot.show()
