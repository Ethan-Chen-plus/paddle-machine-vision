# 作业四：撰写项目README并完成开源

## 评分标准
1.格式规范（有至少3个小标题，内容完整），一个小标题5分，最高20分

2.图文并茂，一张图5分，最高20分

3.有可运行的代码，且代码内有详细注释，20分

4.代码开源到github，15分

5.代码同步到gitee，5分

# 作业内容
把自己的项目描述放在对应区域中，形成一个完整流程项目，并且在最后声明自己项目的github和gitee链接









## 一、项目背景介绍

本次项目主要包括三个小部分：

1、初探图像分类：以农业数据集vegetable为代表的农作物分类

2、计算机视觉的医学应用：眼底血管数据集

3、图像风格迁移：马变斑马

这次，我们将以计算机视觉为主线，从一些经典的角度对深度学习视觉处理产生一些更深的认识。

## 二、数据介绍
#### 1、vegetable

数据集链接：https://aistudio.baidu.com/aistudio/datasetdetail/129400


数据集大小：10M

使用tree查看文件目录结构：

```
.
|--cuke
| |--1515826947475.jpg
| |--1515826951490.jpg
| |--1515826952893.jpg
| |--...
|--lettuce
| |--1515827008819.jpg
| |--1515827009249.jpg
| |--1515827009552.jpg
| |--1515827009827.jpg
| |--1515827010211.jpg
| |--...
|--lotus_root
| |--1515827047952.jpg
| |--1515827048269.jpg
| |--1515827048546.jpg
| |--1515827048831.jpg
| |--1515827049128.jpg
| |--1515827049620.jpg
| |--1515827051120.jpg
| |--...
|--train.txt
```

我们可以看到，数据集含有三个种类的图片：黄瓜、莴苣、藕。

![img](https://ai-studio-static-online.cdn.bcebos.com/a858a23ba7e245f0b5f16bf0f05cec1d14652ddf1eca47c0a007c7899d1e8f2a) ![img](https://ai-studio-static-online.cdn.bcebos.com/d23256906f7e4c0a916c3224f01b571132d81dc9c40f47659211abc115f6ac8b) 

同时，train.txt提供了已经进行好标注的图片，作为训练集。
![image.png](attachment:a8a7f533-9158-47fc-b9fd-5d323b339028.png)
我们可以采取图像分类的一般思路，进行数据分类和处理。

#### 2、 FundusVessels数据集介绍

本项目使用的数据集照片来自荷兰的糖尿病视网膜病变筛查项目。筛查人群包括400名年龄在25-90岁之间的糖尿病患者。但只有40张照片被选取，其中33张没有显示任何糖尿病视网膜病变的迹象，7张显示轻度早期糖尿病视网膜病变的迹象。

AI Studio上已经有[DRIVE糖尿病人眼底血管分割](https://aistudio.baidu.com/aistudio/datasetdetail/27737)数据集了，但是该数据集的数据量非常少，只有20张训练集。

因此，我在处理数据时做了一些处理来增加我的训练集数据量。

## 数据集图片格式转换

原数据集里的眼底图像：

![img](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252135098.png)

原数据集手工分好的的血管图像：

![img](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252135610.png)


#### 3、horse2zebra

数据集链接：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip 

horse2zebra是用于在Python中做CycleGAN学习测试的由马图片生成斑马图片的数据集，包括训练集和测试集（普通的马和斑马）。

数据集目录结构

```
.
|--horse2zebra
| |--testA
| | |--n02381460_1000.jpg
| | |--n02381460_1010.jpg
| | |--n02381460_1030.jpg
| | |--...
| |--testB
| | |--n02391049_80.jpg
| | |--n02391049_100.jpg
| | |--n02391049_130.jpg
| | |--...
| |--trainA
| | |--n02381460_2.jpg
| | |--n02381460_11.jpg
| | |--n02381460_14.jpg
| | |--...
| |--trainB
| | |--n02391049_2.jpg
| | |--n02391049_4.jpg
| | |--n02391049_7.jpg
| | |--...
```
数据集包含了testA、testB、trainA、trainB四个文件夹，里面分别含有马、斑马的训练集和测试集。

![image.png](attachment:73bb52f6-8d6b-4cf9-9c6b-d8bd6fd4dc23.png)
原版书籍集是用[Torch](https://so.csdn.net/so/search?q=Torch&spm=1001.2101.3001.7020)实现的图像到图像的转换（pix2pix），而不用输入输出数据对，例如：

　　![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwNDExLzQ2ODMzZTZiMWM0MTQ3NzRhZDhkMzNlMjY4YmJlZmZjLmpwZWc?x-oss-process=image/format,png)

　　这个程序包包含CycleGAN，pix2pix，以及其他方法，例如：BiGAN/ALI以及苹果的论文：S+U learning.

　　代码作者：Jun-YanZhu和TaesungPark

　　PyTorch版本即将上线

　　**应用** **莫奈油画转换成照片**

　　![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwNDExLzZmMWFmYWFhZGM1ODQyOTU5MTBmMWQxYzZmMDQyM2U1X3RoLmpwZWc?x-oss-process=image/format,png)

　　**画集风格转换**

　　![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwNDExLzhjYWZlNTY0YTQxMzRiZjQ5ZjA0MDgyZTE1OTI4NjhiX3RoLmpwZWc?x-oss-process=image/format,png)

　　**目标变形**

　　![img](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252126538.png)

　　**季节转换**

　　![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwNDExLzY4ZjQxOWVlMWJiNjQ1YzQ4OWI5MWVlNjBjZWZkNWM1LmpwZWc?x-oss-process=image/format,png)

　　**照片增强：iPhone照片转DSLR（数码单反相机）照片**

　　![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwNDExLzZmNTcxNDRjOGZlYjQ4NmZhNzM0NjM5Zjk3MTdlZTIzLmpwZWc?x-oss-process=image/format,png)





## 三、模型介绍
#### 1、运用GoogleNet算法进行vegetables图片分类
算法原理如下

![image.png](attachment:5da0a245-1bd2-4acc-88f2-b4f5ccc8df70.png)

> 算法核心思想：inception


inception模块的基本机构如下图，整个inception结构就是由多个这样的inception模块串联起来的。inception结构的主要贡献有两个：一是使用1x1的卷积来进行升降维；二是在多个尺寸上同时进行卷积再聚合。

![img](https://pic3.zhimg.com/80/v2-fa6813ae7f80db92580404ef652800e6_720w.jpg)
图1:inception模块

**1、1x1卷积**

可以看到图1中有多个黄色的1x1卷积模块，这样的卷积有什么用处呢？

**作用1：**在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。这个观点来自于Network in Network(NIN, [https://arxiv.org/pdf/1312.4400.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1312.4400.pdf))，图1里三个1x1卷积都起到了该作用。

![img](https://pic3.zhimg.com/80/v2-c98a6c44c6d943fb8dbacc29513dcf6e_720w.jpg)
图2:线性卷积和NIN结构对比

图2左侧是是传统的卷积层结构（线性卷积），在一个尺度上只有一次卷积；右图是Network in Network结构（NIN结构），先进行一次普通的卷积（比如3x3），紧跟再进行一次1x1的卷积，对于某个像素点来说1x1卷积等效于该像素点在所有特征上进行一次全连接的计算，所以右侧图的1x1卷积画成了全连接层的形式，需要注意的是NIN结构中无论是第一个3x3卷积还是新增的1x1卷积，后面都紧跟着激活函数（比如relu）。将两个卷积串联，就能组合出更多的非线性特征。举个例子，假设第1个3x3卷积＋激活函数近似于$f_1(x)=ax^2+bx+c$，第二个1x1卷积＋激活函数近似于$f_2(x)=mx^2+nx+q$，那$f_1(x)$和$f_2(f_1(x))$比哪个非线性更强，更能模拟非线性的特征？答案是显而易见的。NIN的结构和传统的神经网络中多层的结构有些类似，后者的多层是跨越了不同尺寸的感受野（通过层与层中间加pool层），从而在更高尺度上提取出特征；NIN结构是在同一个尺度上的多层（中间没有pool层），从而在相同的感受野范围能提取更强的非线性。

**作用2：**使用1x1卷积进行降维，降低了计算复杂度。图2中间3x3卷积和5x5卷积前的1x1卷积都起到了这个作用。当某个卷积层输入的特征数较多，对这个输入进行卷积运算将产生巨大的计算量；如果对输入先进行降维，减少特征数后再做卷积计算量就会显著减少。下图是优化前后两种方案的乘法次数比较，同样是输入一组有192个特征、32x32大小，输出256组特征的数据，第一张图直接用3x3卷积实现，需要192x256x3x3x32x32=452984832次乘法；第二张图先用1x1的卷积降到96个特征，再用3x3卷积恢复出256组特征，需要192x96x1x1x32x32+96x256x3x3x32x32=245366784次乘法，使用1x1卷积降维的方法节省了一半的计算量。有人会问，用1x1卷积降到96个特征后特征数不就减少了么，会影响最后训练的效果么？答案是否定的，只要最后输出的特征数不变（256组），中间的降维类似于压缩的效果，并不影响最终训练的结果。

![img](https://pic2.zhimg.com/80/v2-d047d9d4b1a67cfb91501c71c1eb7315_720w.jpg)图3:增加了1x1卷积后降低了计算量

**2、多个尺寸上进行卷积再聚合**

图2可以看到对输入做了4个分支，分别用不同尺寸的filter进行卷积或池化，最后再在特征维度上拼接到一起。这种全新的结构有什么好处呢？Szegedy从多个角度进行了解释：

**解释1：** 在直观感觉上在多个尺度上同时进行卷积，能提取到不同尺度的特征。特征更为丰富也意味着最后分类判断时更加准确。

**解释2：** 利用稀疏矩阵分解成密集矩阵计算的原理来加快收敛速度。举个例子下图左侧是个稀疏矩阵（很多元素都为0，不均匀分布在矩阵中），和一个2x2的矩阵进行卷积，需要对稀疏矩阵中的每一个元素进行计算；如果像右图那样把稀疏矩阵分解成2个子密集矩阵，再和2x2矩阵进行卷积，稀疏矩阵中0较多的区域就可以不用计算，计算量就大大降低。**这个原理应用到inception上就是要在特征维度上进行分解！** 传统的卷积层的输入数据只和一种尺度（比如3x3）的卷积核进行卷积，输出固定维度（比如256个特征）的数据，所有256个输出特征基本上是均匀分布在$3x3$尺度范围上，这可以理解成输出了一个稀疏分布的特征集；而$inception$模块在多个尺度上提取特征（比如$1$x$1$，$3$x$3$，$5$x$5$），输出的256个特征就不再是均匀分布，而是相关性强的特征聚集在一起（比如$1$x$1$的的96个特征聚集在一起，3x3的96个特征聚集在一起，$5$x$5$的64个特征聚集在一起），这可以理解成多个密集分布的子特征集。这样的特征集中因为相关性较强的特征聚集在了一起，不相关的非关键特征就被弱化，同样是输出256个特征，$inception$方法输出的特征“冗余”的信息较少。用这样的“纯”的特征集层层传递最后作为反向计算的输入，自然收敛的速度更快。

![img](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252140098.jpeg)
图4: 将稀疏矩阵分解成子密集矩阵来进行计算

**解释3：** Hebbin赫布原理。Hebbin原理是神经科学上的一个理论，解释了在学习的过程中脑中的神经元所发生的变化，用一句话概括就是*fire togethter, wire together*。赫布认为“两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋”。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水。用在inception结构中就是要把相关性强的特征汇聚到一起。这有点类似上面的解释2，把1x1，3x3，5x5的特征分开。因为训练收敛的最终目的就是要提取出独立的特征，所以预先把相关性强的特征汇聚，就能起到加速收敛的作用。

在inception模块中有一个分支使用了max pooling，作者认为pooling也能起到提取特征的作用，所以也加入模块中。注意这个pooling的stride=1，pooling后没有减少数据的尺寸。

#### 2、语义分割：眼底血管数据集
本次采用的PaddleSeg提供的Matting算法便是对MODNet算法的复现，并在原著基础上提供了多个不同主干网络的预训练模型如RestNet50_vd、HRNet_w18 来满足用户在边缘端、服务端等不同场景部署的需求。
**** Matting算法基本结构 ****

基于深度学习的Matting分为两大类：

**1.一种是** **基于辅助信息输入**。即除了原图和标注图像外，还需要输入其他的信息辅助预测。最常见的辅助信息是Trimap，即将图片划分为前景，背景及过度区域三部分。另外也有以背景或交互点作为辅助信息。

**2. 一种是** **不依赖任何辅助信息**，直接实现Alpha预测。

本文将分别对两类Matting算法展开介绍，和小伙伴们一起梳理Matting的发展历程。

![6576fe342959eda8047eb972599db82f.png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252154551.png)



***\*DIM -Matting\****

DIM（Deep Image Matting）第一次阐述了在给定图像和辅助信息Trimap的情况下，可以通过端到端的方式学习到Alpha。其网络分为两个阶段，第一阶段是深度卷积编码-解码网络， 第二阶段是一个小型卷积神经网络，用来减少编码-解码网络引起的细节损失，提升Alpha预测的准确性和边缘效果。在DIM之后诞生了大量的基于Trimap的Matting网络。

![cfefe8e3e62d883781449556471eb9ae.png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252157116.png)

图片来源：Xu, Ning, et al. "Deep image matting." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.



***\*BGMV2：\****

***\*以背景作为辅助信息\****

BGMv2(Background Matting v2) 改变思路，利用背景图像取代Trimap来辅助网络进行预测，有效避免了Trimap获取费时费力的问题，并将网络分为Base网络和Refiner两部分。在计算量大的Base网络阶段对低分辨率进行初步预测，在Refiner阶段利用Error Map对高分辨率图像相应的切片进行Refine。通过此实现了高分辨率图像的实时预测。

![0acfc6027d2a66d83a502344b4975678.png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252154476.png)

图片来源：Lin, Shanchuan, et al. "Real-time high-resolution background matting." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.



***\*MODNet\****

辅助信息的获取极大限制了Matting的应用，为了提升Matting的应用性，针对Portrait Matting领域MODNet摒弃了辅助信息，直接实现Alpha预测，实现了实时Matting，极大提升了基于深度学习Matting的应用价值。MODNet将Matting分解成三个子目标进行优化，通过任务分解提升Alpha预测的准确率。

![ac9230bbd5e729eb8e0763cdfc70e80c.png](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252154536.png)

图片来源：Ke Z, Li K, Zhou Y, et al. Is a Green Screen Really Necessary for Real-Time Portrait Matting?[J]. arXiv preprint arXiv:2011.11961, 2020.

#### 3、图像风格迁移：horse2zebra

**风格迁移（style transfer）**最近两年非常火，可谓是深度学习领域很有创意的研究成果。它主要是通过神经网络，将一幅艺术风格画（style image）和一张普通的照片（content image）巧妙地融合，形成一张非常有意思的图片。

![image-20220225220200994](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252202154.png)

##### 1. 风格迁移开山之作

2015年，Gatys等人发表了文章[1]《[A Neural Algorithm of Artistic Style](http://melonteam.com/https:/arxiv.org/pdf/1508.06576.pdf)》，首次使用深度学习进行艺术画风格学习。把风格图像Xs的绘画风格融入到内容图像Xc，得到一幅新的图像Xn。则新的图像Xn：即要保持内容图像Xc的原始图像内容（内容画是一部汽车，融合后应仍是一部汽车，不能变成摩托车），又要保持风格图像Xs的特有风格（比如纹理、色调、笔触等）。

##### 1.1 内容损失（Content Loss）

在CNN网络中，一般认为较低层的特征描述了图像的具体视觉特征（即纹理、颜色等），较高层的特征则是较为抽象的图像内容描述。 所以要比较两幅图像的**内容相似性**，可以比较两幅图像在CNN网络中`高层特征`的相似性（欧式距离）。

![image-20220225220233818](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252202856.png)

##### 1.2 风格损失（Style Loss）

而要比较两幅图像的**风格相似性**，则可以比较它们在CNN网络中较`低层特征`的相似性。不过值得注意的是，不能像内容相似性计算一样，简单的采用欧式距离度量，因为低层特征包含较多的图像局部特征（即空间信息过于显著），比如两幅风格相似但内容完全不同的图像，若直接计算它们的欧式距离，则可能会产生较大的误差，认为它们风格不相似。论文中使用了`Gram矩阵`，用于计算不同响应层之间的联系，即在保留低层特征的同时去除图像内容的影响，只比较风格的相似性。

![image-20220225220246161](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252202199.png)

那么风格的相似性计算可以用如下公式表示：

![image-20220225220256523](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252202564.png)

![image-20220225220643752](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252206829.png)

##### 1.3 总损失（Total Loss）

这样对两幅图像进行“内容+风格”的相似度评价，可以采用如下的损失函数：

![image-20220225220708719](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252207812.png)

##### 1.4 训练过程

文章使用了著名的VGG19网络[3]来进行训练（包含16个卷积层和5个池化层，但实际训练中未使用任何全连接层，并使用平均池化average- pooling替代最大池化max-pooling）。

![image-20220225220811284](https://gitee.com/qmckw/pic/raw/master/PicGO-updata-img/202202252208491.png)






## 四、模型训练
#### 1、vegetables图像分类


#####  1.1 导入有关包并下载数据集


```python
# coding=utf-8

import pandas as pd
import paddle.fluid as fluid
import rarfile
import sys
import tarfile
import shutil
import os
import numpy as np
import hashlib
import paddle
import json
import six
from sklearn.model_selection import train_test_split
import zipfile
import requests
import time
from multiprocessing import cpu_count
paddle.enable_static()

WORK_SPACE = '/home/aistudio'
DATA_SPACE = '/home/aistudio/data'
PRE_MODEL_SPACE = '/home/aistudio/pre_model'
MODEL_SPACE = '/home/aistudio/model'

x_shape = [3, 224, 224]
y_shape = [1]
class_dim = 3

    
def compressed_data_import_1():
    """
    download and decompress dataset
    :param
    :return:
        dataset
    """

    def md5file(fname):
        hash_md5 = hashlib.md5()
        f = open(fname, "rb")
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
        f.close()
        return hash_md5.hexdigest()

    def download(url, save_dir, filename=None, md5sum=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if md5sum is not None:
            filename = os.path.join(save_dir, url.split('/')[-1])
        else:
            filename = os.path.join(save_dir, filename)
        retry = 0
        retry_limit = 3
        while not os.path.exists(filename):
            if os.path.exists(filename):
                if md5sum is not None and md5file(filename) == md5sum:
                    break
            if retry < retry_limit:
                retry += 1
            else:
                raise RuntimeError("Cannot download file within retry limit {0}".format(retry_limit))
            sys.stderr.write("Cache file %s not found, downloading ... \n" % (filename))
            r = requests.get(url, stream=True)
            total_length = r.headers.get('content-length')
            
            if total_length is None:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            else:
                with open(filename, 'wb') as f:
                    dl = 0
                    total_length = int(total_length)
                    for data in r.iter_content(chunk_size=4096):
                        if six.PY2:
                            data = six.b(data)
                        dl += len(data)
                        f.write(data)
                    sys.stderr.write("file downloaded success! \n")
        return filename

    def un_zip(file_name):
        """unzip zip file"""
        zip_file = zipfile.ZipFile(file_name)
        un_zip_file = DATA_SPACE
        if os.path.isdir(un_zip_file):
            pass
        else:
            os.mkdir(un_zip_file)
        for names in zip_file.namelist():
            zip_file.extract(names, un_zip_file)
        zip_file.close()
        return un_zip_file

    try:
        flow_start = {
            'flow id':'d938e6e0-9575-11ec-be23-3731348851c2',
            'startTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_start))
        sys.stdout.flush()

        # init params
        is_preset = 1
        file = 'https://ai-studio-static-online.bj.bcebos.com/preset_dataset/vegetables.zip'
        file_type = 'zip'
        global class_dim
        class_dim = 3

        if is_preset == 1:
            md5sum = '90835e76aa00b6c6bf1ed3b8cba96df5'
            # download
            download_file_name = download(file, DATA_SPACE, md5sum=md5sum)
        else:
            # download
            file_name = '${codeInfo.property.fileName}'
            download_file_name = download(file, DATA_SPACE, filename=file_name)

        # decompress
        data_root_path = un_zip(download_file_name)

        # image numbers
        label_list_file = data_root_path + '/train.txt'
        all_images_num = sum(1 for line in open(label_list_file))

        # image label
        if(os.path.exists(data_root_path + '/label_list')):
            all_class_file = data_root_path + '/label_list'
            all_class_sum = sum(1 for line in open(all_class_file))

        # response
        flow_end = {
            'flow id':'d938e6e0-9575-11ec-be23-3731348851c2',
            'endTime':int(round(time.time() * 1000)),
            'response':{
                'properties': {
                    'compressedData':{
                        'allImagesNum': all_images_num,
                        'allClassSum': all_class_sum if(os.path.exists(data_root_path + '/label_list')) else class_dim
                    }
                }
            }
        }
        print(json.dumps(flow_end))
        sys.stdout.flush()

        dataset = {
            'x_shape': [1],
            'y_shape': [1],
            'type': 'image',
            'file_path': label_list_file,
            'data_dir': data_root_path,
            'class_num': all_class_sum if(os.path.exists(data_root_path + '/label_list')) else class_dim,
            'image_num': all_images_num,
            'label_path': '' if not os.path.exists(data_root_path + '/label_list') else data_root_path + '/label_list'
        }
        return dataset
    except Exception as e:
        flow_error = {
            'flow id':'d938e6e0-9575-11ec-be23-3731348851c2',
            'errorTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_error))
        sys.stdout.flush()
        raise e
    
    

```

##### 1.2 数据分割


```python
def data_segment_1(dataset):
    """
    :param
        dataset
    :return:
        dataset
        dataset
    """

    try:
        flow_start = {
            'flow id':'ddec83e0-9575-11ec-be23-3731348851c2',
            'startTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_start))
        sys.stdout.flush()
        
        # init params
        test_size = 20/float(100)

        preprocessed_file_path = dataset['file_path']
        x_shape = dataset['x_shape']
        y_shape = dataset['y_shape']

        preprocessed_dataset = pd.read_csv(preprocessed_file_path, sep=' ', header=None)
        preprocessed_x = preprocessed_dataset.iloc[:, 0:x_shape[0]]
        preprocessed_y = preprocessed_dataset.iloc[:, -y_shape[0]:]

        x_train, x_test, y_train, y_test = train_test_split(preprocessed_x, preprocessed_y, test_size=test_size)
        train_data = np.hstack((x_train, y_train))
        test_data = np.hstack((x_test, y_test))

        segmented_file_path_1 = DATA_SPACE + '/segmented_dataset_1_1.txt'
        segmented_file_path_2 = DATA_SPACE + '/segmented_dataset_1_2.txt'
        pd.DataFrame(train_data).to_csv(segmented_file_path_1, sep=' ', index=False, header=False)
        pd.DataFrame(test_data).to_csv(segmented_file_path_2, sep=' ', index=False, header=False)

        flow_end = {
            'flow id':'ddec83e0-9575-11ec-be23-3731348851c2',
            'endTime':int(round(time.time() * 1000)),
            'response':{
                'trainSize': train_data.shape,
                'testSize': test_data.shape
            }
        }
        print(json.dumps(flow_end))
        sys.stdout.flush()

        dataset1 = {
            'x_shape': x_shape,
            'y_shape': y_shape,
            'type': dataset['type'],
            'file_path': segmented_file_path_1,
            'data_dir': dataset['data_dir'],
            'class_num': dataset['class_num'] if dataset['class_num'] else -1,
            'image_num': train_data.shape[0] if dataset['type'] == 'image' else 0,
            'label_path': dataset['label_path']
        }
        dataset2 = {
            'x_shape': x_shape,
            'y_shape': y_shape,
            'type': dataset['type'],
            'file_path': segmented_file_path_2,
            'data_dir': dataset['data_dir'],
            'class_num': dataset['class_num'] if dataset['class_num'] else -1,
            'image_num': test_data.shape[0] if dataset['type'] == 'image' else 0,
            'label_path': dataset['label_path']
        }
        return dataset1, dataset2
    except Exception as e:
        flow_error = {
            'flow id':'ddec83e0-9575-11ec-be23-3731348851c2',
            'errorTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_error))
        sys.stdout.flush()
        raise e
    

```

##### 1.3 调用googlenet算法


```python
def googlenet_1():
    """googlenet"""
    flow_start = {
        'flow id':'e7d968a0-9575-11ec-be23-3731348851c2',
        'startTime':int(round(time.time() * 1000))
    }
    print(json.dumps(flow_start))
    sys.stdout.flush()

    # init params
    global x_shape, y_shape
    x_shape = [3, 224, 224]
    y_shape = [1]

    def conv_layer(input, num_filters, filter_size, stride=1, groups=1, act=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv))
        conv = fluid.layers.conv2d(input=input, num_filters=num_filters, filter_size=filter_size,
                                   stride=stride, padding=(filter_size - 1) // 2, groups=groups, act=act,
                                   param_attr=param_attr, bias_attr=False)
        return conv

    def xavier(channels, filter_size):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv))
        return param_attr

    def inception(name, input, channels, filter1, filter3R, filter3, filter5R, filter5, proj):
        conv1 = conv_layer(input=input, num_filters=filter1, filter_size=1, stride=1, act=None)
        conv3r = conv_layer(input=input, num_filters=filter3R, filter_size=1, stride=1, act=None)
        conv3 = conv_layer(input=conv3r, num_filters=filter3, filter_size=3, stride=1, act=None)
        conv5r = conv_layer(input=input, num_filters=filter5R, filter_size=1, stride=1, act=None)
        conv5 = conv_layer(input=conv5r, num_filters=filter5, filter_size=5, stride=1, act=None)
        pool = fluid.layers.pool2d(input=input, pool_size=3, pool_stride=1, pool_padding=1, pool_type='max')
        convprj = fluid.layers.conv2d(input=pool, filter_size=1, num_filters=proj, stride=1, padding=0)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def network(input, class_dim=1000):
        conv = conv_layer(input=input, num_filters=64, filter_size=7, stride=2, act=None)
        pool = fluid.layers.pool2d(input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = conv_layer(input=pool, num_filters=64, filter_size=1, stride=1, act=None)
        conv = conv_layer(input=conv, num_filters=192, filter_size=3, stride=1, act=None)
        pool = fluid.layers.pool2d(input=conv, pool_size=3, pool_type='max', pool_stride=2)
        
        ince3a = inception("ince3a", pool, 192, 64, 96, 128, 16, 32, 32)
        ince3b = inception("ince3b", ince3a, 256, 128, 128, 192, 32, 96, 64)
        pool3 = fluid.layers.pool2d(input=ince3b, pool_size=3, pool_type='max', pool_stride=2)
        
        ince4a = inception("ince4a", pool3, 480, 192, 96, 208, 16, 48, 64)
        ince4b = inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64, 64)
        ince4c = inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64, 64)
        ince4d = inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64, 64)
        ince4e = inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128, 128)
        pool4 = fluid.layers.pool2d(input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = inception("ince5a", pool4, 832, 256, 160, 320, 32, 128, 128)
        ince5b = inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128, 128)
        pool5 = fluid.layers.pool2d(input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout, size=class_dim, act='softmax', param_attr=xavier(1024, 1))

        pool_o1 = fluid.layers.pool2d(input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = conv_layer(input=pool_o1, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o1 = fluid.layers.fc(input=conv_o1, size=1024, act='relu', param_attr=xavier(2048, 1))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1, size=class_dim, act='softmax', param_attr=xavier(1024, 1))

        pool_o2 = fluid.layers.pool2d(input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = conv_layer(input=pool_o2, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o2 = fluid.layers.fc(input=conv_o2, size=1024, act='relu', param_attr=xavier(2048, 1))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2, size=class_dim, act='softmax', param_attr=xavier(1024, 1))

        # last fc layer is "out"
        return out, out1, out2

    image = fluid.layers.data(name='x', shape=x_shape, dtype='float32')
    predict = network(image, class_dim)
    
    model = {
        'name': 'GoogleNet',
        'x_shape': x_shape,
        'y_shape': y_shape,
        'class_dim': class_dim,
        'predict': predict,
        'network_type': 'Classification'
    }

    flow_end = {
        'flow id':'e7d968a0-9575-11ec-be23-3731348851c2',
        'endTime':int(round(time.time() * 1000)),
        'response':{}
    }
    print(json.dumps(flow_end))
    sys.stdout.flush()
    
    return model
    
    
# define image reader

```

##### 1.4 读取数据并开始训练


```python
def image_dataset_reader(data_dir, file_path, is_train):
    # define image mapper
    def image_dataset_mapper(sample):
        x, y, is_train = sample
        is_color = True if len(x_shape) == 3 and x_shape[0] == 3 else False
        x = paddle.dataset.image.load_image(file=x, is_color=is_color)
        x = paddle.dataset.image.simple_transform(im=x, resize_size=x_shape[1]+5,
                                                  crop_size=x_shape[1], is_color=is_color, is_train=is_train)
        x = x.flatten().astype('float32') / 255.0
        return x, y
    
    def reader():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            del lines[len(lines) - 1]
            for line in lines:
                x, y = line.split(' ')
                yield data_dir + '/' + x, int(y), is_train
    
    return paddle.reader.xmap_readers(image_dataset_mapper, reader, cpu_count(), 1024)


def image_classification_train_1(train_dataset, test_dataset, model):
    """image classification train"""

    try:
        flow_start = {
            'flow id':'e0290660-9575-11ec-be23-3731348851c2',
            'startTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_start))
        sys.stdout.flush()

        # init params
        buf_size = 8 * 2
        batch_size = 8
        epoch = 100
        learning_rate = 0.01
        x_shape = model['x_shape']
        y_shape = model['y_shape']
        train_images = train_dataset['image_num']
        
        # optimizer
        def optimizer_setting():
            if epoch > 10:
                iters = train_images // batch_size
                lr_epochs = [(i + 1) * 10 for i in range(epoch//10)]
                boundaries = [i * iters for i in lr_epochs]
                values = [learning_rate * (0.1**i) for i in range(len(boundaries) + 1)]
                optimizer = fluid.optimizer.RMSProp(
                    learning_rate=fluid.layers.piecewise_decay(boundaries, values),
                    regularization=fluid.regularizer.L2Decay(0.00001), )
            else:
                optimizer = fluid.optimizer.RMSProp(
                    learning_rate=learning_rate,
                    regularization=fluid.regularizer.L2Decay(0.00005), )
            return optimizer

        # define network
        x = fluid.layers.data(name='x', shape=x_shape, dtype='float32')
        y = fluid.layers.data(name='y', shape=y_shape, dtype='int64')
        if model['name'] == 'GoogleNet':
            y_predict, y_predict1, y_predict2 = model['predict']
            cost0 = fluid.layers.cross_entropy(input=y_predict, label=y)
            cost1 = fluid.layers.cross_entropy(input=y_predict1, label=y)
            cost2 = fluid.layers.cross_entropy(input=y_predict2, label=y)
            avg_cost0 = fluid.layers.mean(x=cost0)
            avg_cost1 = fluid.layers.mean(x=cost1)
            avg_cost2 = fluid.layers.mean(x=cost2)
            avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        else:
            y_predict = model['predict']
            cost = fluid.layers.cross_entropy(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(x=cost)
        accuracy = fluid.layers.accuracy(input=y_predict, label=y, k=1)
        fetch_list=[avg_cost, accuracy]

        # clone test prog
        test_program = fluid.default_main_program().clone(for_test=True)
        
        # optimizer
        # optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
        optimizer = optimizer_setting()
        optimizer.minimize(avg_cost)

        # define executor
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place=place)
        exe.run(program=fluid.default_startup_program())

        # data reader
        if train_dataset['type'] != 'image':
            print('输入数据集无法转换为算法所需要的格式')
            sys.stdout.flush()
            raise Exception()
        train_reader = paddle.batch(
            reader=paddle.reader.shuffle(reader=image_dataset_reader(train_dataset['data_dir'], train_dataset['file_path'], True), buf_size=buf_size),
            batch_size=batch_size)
        if test_dataset:
            if test_dataset['type'] != 'image':
                print('输入数据集无法转换为算法所需要的格式')
                sys.stdout.flush()
                raise Exception()
            test_reader = paddle.batch(reader=image_dataset_reader(test_dataset['data_dir'], test_dataset['file_path'], False),
                                       batch_size=batch_size)

        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        
        train_info = []
        test_info = []

        # start train
        for pass_id in range(epoch):
            epoch_train_info = []
            for batch_id, data in enumerate(train_reader()):
                train_result = exe.run(program=fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=fetch_list)
                epoch_train_info.append("Train-Pass:" + str(pass_id) + " Batch:" + str(batch_id)
                                        + " Cost:" + str(train_result[0][0]) + " Accuracy:" + str(train_result[1][0]))
            if epoch_train_info:
                train_info.append(epoch_train_info[-1])
                print(epoch_train_info[-1])
                sys.stdout.flush()
            if test_dataset:
                test_costs = []
                test_accs = []
                for batch_id, data in enumerate(test_reader()):
                    test_result = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=fetch_list)
                    test_costs.append(test_result[0][0])
                    test_accs.append(test_result[1][0])
                test_cost = sum(test_costs) / len(test_costs)
                test_acc = sum(test_accs) / len(test_accs)
                test_info.append("Train-Pass:" + str(pass_id) + " Test-Cost:" + str(test_cost) + " Test-Accuracy:" + str(test_acc))
                print('Train-Pass:%d Test-Cost:%f Test-Accuracy:%f' % (pass_id, test_cost, test_acc))
                sys.stdout.flush()

        # save model
        model_save_sub_dir =  model['name'] + '_' + str(1)
        model_save_dir = MODEL_SPACE + '/' + model_save_sub_dir
        fluid.io.save_inference_model(dirname=model_save_dir,
                                     feeded_var_names=['x'],
                                     target_vars=[y_predict],
                                     executor=exe)

        flow_end = {
            'flow id': 'e0290660-9575-11ec-be23-3731348851c2',
            'endTime': int(round(time.time() * 1000)),
            'response': {
                'train': train_info,
                'test': test_info,
                'savedModelInfo': {
                    'aliasId': 'e0290660-9575-11ec-be23-3731348851c2',
                    'itemId': 1,
                    'modelType': model['network_type'],
                    'modelName': model['name'],
                    'xShape': model['x_shape'],
                    'modelDir': model_save_sub_dir
                }
            }
        }
        print(json.dumps(flow_end))
        sys.stdout.flush()

        model['dirname'] = model_save_dir
        model['executor'] = exe
        model['buf_size'] = buf_size
        model['batch_size'] = batch_size
        model['epoch'] = epoch
        model['train_program'] = fluid.default_main_program()
        model['test_program'] = test_program
        model['feeder'] = feeder
        model['fetch_list'] = fetch_list
        return model
    except Exception as e:
        flow_error = {
            'flow id':'e0290660-9575-11ec-be23-3731348851c2',
            'errorTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_error))
        sys.stdout.flush()
        raise e
    
    
def image_classification_test_1(test_dataset, model):
    """image classification test"""

    try:
        flow_start = {
            'flow id':'e362d4a0-9575-11ec-be23-3731348851c2',
            'startTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_start))
        sys.stdout.flush()

        # init params
        x_shape = model['x_shape']
        y_shape = model['y_shape']
        executor = model['executor']
        test_program = model['test_program']
        feeder = model['feeder']
        fetch_list = model['fetch_list']
        batch_size = model['batch_size']

        # test reader
        test_reader = paddle.batch(reader=image_dataset_reader(test_dataset['data_dir'],test_dataset['file_path'], False),
                                   batch_size=batch_size)

        test_info = []
        test_costs = []
        test_accs = []
        for batch_id, data in enumerate(test_reader()):
            test_result = executor.run(program=test_program,
                                       feed=feeder.feed(data),
                                       fetch_list=fetch_list)
            test_costs.append(test_result[0][0])
            test_accs.append(test_result[1][0])

        test_cost = sum(test_costs) / len(test_costs)
        test_acc = sum(test_accs) / len(test_accs)
        test_info.append("Test-Cost:" + str(test_cost) + " Test-Accuracy:" + str(test_acc))
        print('Test-Cost:%f Test-Accuracy:%f' % (test_cost, test_acc))
        sys.stdout.flush()

        flow_end = {
            'flow id':'e362d4a0-9575-11ec-be23-3731348851c2',
            'endTime':int(round(time.time() * 1000)),
            'response':{
                'test': test_info
            }
        }
        print(json.dumps(flow_end))
        sys.stdout.flush()

        return model
    except Exception as e:
        flow_error = {
            'flow id':'e362d4a0-9575-11ec-be23-3731348851c2',
            'errorTime':int(round(time.time() * 1000))
        }
        print(json.dumps(flow_error))
        sys.stdout.flush()
        raise e
    

if __name__ == '__main__':
    var_output100110 = compressed_data_import_1()
    var_output100310, var_output100311 = data_segment_1(var_output100110)
    var_output250510 = googlenet_1()
    var_output300210 = image_classification_train_1(var_output100310, var_output100311, var_output250510)
    var_output400210 = image_classification_test_1(var_output100311, var_output300210)
```

    {"flow id": "d938e6e0-9575-11ec-be23-3731348851c2", "startTime": 1645798469390}
    

    Cache file /home/aistudio/data/vegetables.zip not found, downloading ... 
    

    {"flow id": "d938e6e0-9575-11ec-be23-3731348851c2", "endTime": 1645798470392, "response": {"properties": {"compressedData": {"allImagesNum": 300, "allClassSum": 3}}}}
    {"flow id": "ddec83e0-9575-11ec-be23-3731348851c2", "startTime": 1645798470393}
    {"flow id": "ddec83e0-9575-11ec-be23-3731348851c2", "endTime": 1645798470402, "response": {"trainSize": [240, 2], "testSize": [60, 2]}}
    {"flow id": "e7d968a0-9575-11ec-be23-3731348851c2", "startTime": 1645798470403}
    

    file downloaded success! 
    

    {"flow id": "e7d968a0-9575-11ec-be23-3731348851c2", "endTime": 1645798470545, "response": {}}
    {"flow id": "e0290660-9575-11ec-be23-3731348851c2", "startTime": 1645798470545}
    

    W0225 22:14:30.971613   229 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0225 22:14:30.976670   229 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    

    Train-Pass:0 Batch:29 Cost:73.14286 Accuracy:0.2857143
    Train-Pass:0 Test-Cost:67.433334 Test-Accuracy:0.338542
    Train-Pass:1 Batch:29 Cost:67.65715 Accuracy:0.2857143
    Train-Pass:1 Test-Cost:59.166667 Test-Accuracy:0.427083
    Train-Pass:2 Batch:29 Cost:83.200005 Accuracy:0.0
    Train-Pass:2 Test-Cost:70.033335 Test-Accuracy:0.307292
    Train-Pass:3 Batch:29 Cost:58.51429 Accuracy:0.42857143
    Train-Pass:3 Test-Cost:70.100001 Test-Accuracy:0.281250
    Train-Pass:4 Batch:29 Cost:66.74286 Accuracy:0.42857143
    Train-Pass:4 Test-Cost:67.266667 Test-Accuracy:0.364583
    Train-Pass:5 Batch:29 Cost:79.54286 Accuracy:0.14285715
    Train-Pass:5 Test-Cost:68.033332 Test-Accuracy:0.307292
    Train-Pass:6 Batch:29 Cost:58.51429 Accuracy:0.42857143
    Train-Pass:6 Test-Cost:65.366668 Test-Accuracy:0.380208
    Train-Pass:7 Batch:29 Cost:75.88571 Accuracy:0.2857143
    Train-Pass:7 Test-Cost:67.633335 Test-Accuracy:0.354167
    Train-Pass:8 Batch:29 Cost:47.88485 Accuracy:0.42857143
    Train-Pass:8 Test-Cost:57.368449 Test-Accuracy:0.307292
    Train-Pass:9 Batch:29 Cost:47.85202 Accuracy:0.42857143
    Train-Pass:9 Test-Cost:51.280324 Test-Accuracy:0.411458
    Train-Pass:10 Batch:29 Cost:59.7823 Accuracy:0.2857143
    Train-Pass:10 Test-Cost:53.769757 Test-Accuracy:0.354167
    Train-Pass:11 Batch:29 Cost:63.410152 Accuracy:0.14285715
    Train-Pass:11 Test-Cost:47.759420 Test-Accuracy:0.416667
    Train-Pass:12 Batch:29 Cost:66.166016 Accuracy:0.14285715
    Train-Pass:12 Test-Cost:46.594699 Test-Accuracy:0.442708
    Train-Pass:13 Batch:29 Cost:38.736935 Accuracy:0.5714286
    Train-Pass:13 Test-Cost:54.638546 Test-Accuracy:0.322917
    Train-Pass:14 Batch:29 Cost:53.352787 Accuracy:0.42857143
    Train-Pass:14 Test-Cost:50.265920 Test-Accuracy:0.369792
    Train-Pass:15 Batch:29 Cost:54.278954 Accuracy:0.2857143
    Train-Pass:15 Test-Cost:56.551185 Test-Accuracy:0.291667
    Train-Pass:16 Batch:29 Cost:54.278328 Accuracy:0.2857143
    Train-Pass:16 Test-Cost:49.775012 Test-Accuracy:0.411458
    Train-Pass:17 Batch:29 Cost:63.421204 Accuracy:0.14285715
    Train-Pass:17 Test-Cost:51.100402 Test-Accuracy:0.380208
    Train-Pass:18 Batch:29 Cost:68.88622 Accuracy:0.14285715
    Train-Pass:18 Test-Cost:50.300138 Test-Accuracy:0.411458
    Train-Pass:19 Batch:29 Cost:51.524876 Accuracy:0.2857143
    Train-Pass:19 Test-Cost:46.631316 Test-Accuracy:0.437500
    Train-Pass:20 Batch:29 Cost:68.90619 Accuracy:0.14285715
    Train-Pass:20 Test-Cost:41.733178 Test-Accuracy:0.468750
    Train-Pass:21 Batch:29 Cost:59.753925 Accuracy:0.2857143
    Train-Pass:21 Test-Cost:38.599825 Test-Accuracy:0.520833
    Train-Pass:22 Batch:29 Cost:45.125366 Accuracy:0.42857143
    Train-Pass:22 Test-Cost:42.468172 Test-Accuracy:0.479167
    Train-Pass:23 Batch:29 Cost:45.115177 Accuracy:0.42857143
    Train-Pass:23 Test-Cost:36.895847 Test-Accuracy:0.536458
    Train-Pass:24 Batch:29 Cost:47.877518 Accuracy:0.42857143
    Train-Pass:24 Test-Cost:37.934782 Test-Accuracy:0.546875
    Train-Pass:25 Batch:29 Cost:48.782093 Accuracy:0.2857143
    Train-Pass:25 Test-Cost:50.464557 Test-Accuracy:0.338542
    Train-Pass:26 Batch:29 Cost:47.877464 Accuracy:0.42857143
    Train-Pass:26 Test-Cost:38.866389 Test-Accuracy:0.494792
    Train-Pass:27 Batch:29 Cost:57.020813 Accuracy:0.2857143
    Train-Pass:27 Test-Cost:37.799614 Test-Accuracy:0.520833
    Train-Pass:28 Batch:29 Cost:71.55658 Accuracy:0.14285715
    Train-Pass:28 Test-Cost:54.240968 Test-Accuracy:0.401042
    Train-Pass:29 Batch:29 Cost:26.84831 Accuracy:0.71428573
    Train-Pass:29 Test-Cost:38.564502 Test-Accuracy:0.526042
    Train-Pass:30 Batch:29 Cost:57.916515 Accuracy:0.14285715
    Train-Pass:30 Test-Cost:37.264500 Test-Accuracy:0.541667
    Train-Pass:31 Batch:29 Cost:42.38267 Accuracy:0.42857143
    Train-Pass:31 Test-Cost:33.034643 Test-Accuracy:0.578125
    Train-Pass:32 Batch:29 Cost:47.858856 Accuracy:0.42857143
    Train-Pass:32 Test-Cost:28.301307 Test-Accuracy:0.645833
    Train-Pass:33 Batch:29 Cost:35.991684 Accuracy:0.5714286
    Train-Pass:33 Test-Cost:33.499567 Test-Accuracy:0.583333
    Train-Pass:34 Batch:29 Cost:63.410194 Accuracy:0.14285715
    Train-Pass:34 Test-Cost:33.699661 Test-Accuracy:0.598958
    Train-Pass:35 Batch:29 Cost:26.821304 Accuracy:0.71428573
    Train-Pass:35 Test-Cost:34.799659 Test-Accuracy:0.567708
    Train-Pass:36 Batch:29 Cost:41.48638 Accuracy:0.5714286
    Train-Pass:36 Test-Cost:31.299657 Test-Accuracy:0.598958
    Train-Pass:37 Batch:29 Cost:54.267868 Accuracy:0.2857143
    Train-Pass:37 Test-Cost:29.534723 Test-Accuracy:0.671875
    Train-Pass:38 Batch:29 Cost:41.477913 Accuracy:0.5714286
    Train-Pass:38 Test-Cost:38.756240 Test-Accuracy:0.510417
    Train-Pass:39 Batch:29 Cost:53.28497 Accuracy:0.42857143
    Train-Pass:39 Test-Cost:32.889573 Test-Accuracy:0.583333
    Train-Pass:40 Batch:29 Cost:24.09644 Accuracy:0.71428573
    Train-Pass:40 Test-Cost:30.494171 Test-Accuracy:0.598958
    Train-Pass:41 Batch:29 Cost:63.429756 Accuracy:0.14285715
    Train-Pass:41 Test-Cost:29.895904 Test-Accuracy:0.630208
    Train-Pass:42 Batch:29 Cost:53.3631 Accuracy:0.42857143
    Train-Pass:42 Test-Cost:31.994072 Test-Accuracy:0.583333
    Train-Pass:43 Batch:29 Cost:47.87738 Accuracy:0.42857143
    Train-Pass:43 Test-Cost:30.994170 Test-Accuracy:0.598958
    Train-Pass:44 Batch:29 Cost:20.439833 Accuracy:0.85714287
    Train-Pass:44 Test-Cost:30.991305 Test-Accuracy:0.598958
    Train-Pass:45 Batch:29 Cost:57.02024 Accuracy:0.2857143
    Train-Pass:45 Test-Cost:29.989669 Test-Accuracy:0.614583
    Train-Pass:46 Batch:29 Cost:45.1155 Accuracy:0.42857143
    Train-Pass:46 Test-Cost:28.524637 Test-Accuracy:0.625000
    Train-Pass:47 Batch:29 Cost:42.391666 Accuracy:0.42857143
    Train-Pass:47 Test-Cost:31.491403 Test-Accuracy:0.598958
    Train-Pass:48 Batch:29 Cost:29.532034 Accuracy:0.71428573
    Train-Pass:48 Test-Cost:29.954601 Test-Accuracy:0.619792
    Train-Pass:49 Batch:29 Cost:57.0197 Accuracy:0.2857143
    Train-Pass:49 Test-Cost:30.989668 Test-Accuracy:0.598958
    Train-Pass:50 Batch:29 Cost:50.61073 Accuracy:0.42857143
    Train-Pass:50 Test-Cost:30.989668 Test-Accuracy:0.598958
    Train-Pass:51 Batch:29 Cost:24.09644 Accuracy:0.71428573
    Train-Pass:51 Test-Cost:28.824637 Test-Accuracy:0.625000
    Train-Pass:52 Batch:29 Cost:39.6393 Accuracy:0.42857143
    Train-Pass:52 Test-Cost:29.989473 Test-Accuracy:0.614583
    Train-Pass:53 Batch:29 Cost:32.29284 Accuracy:0.71428573
    Train-Pass:53 Test-Cost:29.489570 Test-Accuracy:0.614583
    Train-Pass:54 Batch:29 Cost:3.0673323 Accuracy:1.0
    Train-Pass:54 Test-Cost:27.524637 Test-Accuracy:0.640625
    Train-Pass:55 Batch:29 Cost:44.170116 Accuracy:0.5714286
    Train-Pass:55 Test-Cost:30.691304 Test-Accuracy:0.598958
    Train-Pass:56 Batch:29 Cost:35.932034 Accuracy:0.5714286
    Train-Pass:56 Test-Cost:30.689571 Test-Accuracy:0.598958
    Train-Pass:57 Batch:29 Cost:45.115505 Accuracy:0.42857143
    Train-Pass:57 Test-Cost:30.389570 Test-Accuracy:0.598958
    Train-Pass:58 Batch:29 Cost:29.532034 Accuracy:0.71428573
    Train-Pass:58 Test-Cost:30.189667 Test-Accuracy:0.598958
    Train-Pass:59 Batch:29 Cost:66.11351 Accuracy:0.14285715
    Train-Pass:59 Test-Cost:32.354602 Test-Accuracy:0.572917
    Train-Pass:60 Batch:29 Cost:24.096977 Accuracy:0.71428573
    Train-Pass:60 Test-Cost:30.189571 Test-Accuracy:0.598958
    Train-Pass:61 Batch:29 Cost:38.734524 Accuracy:0.5714286
    Train-Pass:61 Test-Cost:28.524637 Test-Accuracy:0.625000
    Train-Pass:62 Batch:29 Cost:59.753586 Accuracy:0.2857143
    Train-Pass:62 Test-Cost:32.354601 Test-Accuracy:0.572917
    Train-Pass:63 Batch:29 Cost:35.982697 Accuracy:0.5714286
    Train-Pass:63 Test-Cost:32.356238 Test-Accuracy:0.572917
    Train-Pass:64 Batch:29 Cost:42.350517 Accuracy:0.42857143
    Train-Pass:64 Test-Cost:30.189668 Test-Accuracy:0.598958
    Train-Pass:65 Batch:29 Cost:14.963094 Accuracy:0.85714287
    Train-Pass:65 Test-Cost:32.856335 Test-Accuracy:0.572917
    Train-Pass:66 Batch:29 Cost:24.10595 Accuracy:0.71428573
    Train-Pass:66 Test-Cost:30.189570 Test-Accuracy:0.598958
    Train-Pass:67 Batch:29 Cost:26.848272 Accuracy:0.71428573
    Train-Pass:67 Test-Cost:30.689570 Test-Accuracy:0.598958
    Train-Pass:68 Batch:29 Cost:54.268406 Accuracy:0.2857143
    Train-Pass:68 Test-Cost:30.689668 Test-Accuracy:0.598958
    Train-Pass:69 Batch:29 Cost:14.963094 Accuracy:0.85714287
    Train-Pass:69 Test-Cost:32.356237 Test-Accuracy:0.572917
    Train-Pass:70 Batch:29 Cost:26.830862 Accuracy:0.71428573
    Train-Pass:70 Test-Cost:28.524637 Test-Accuracy:0.625000
    Train-Pass:71 Batch:29 Cost:51.516037 Accuracy:0.2857143
    Train-Pass:71 Test-Cost:30.689668 Test-Accuracy:0.598958
    Train-Pass:72 Batch:29 Cost:24.08693 Accuracy:0.71428573
    Train-Pass:72 Test-Cost:32.354601 Test-Accuracy:0.572917
    Train-Pass:73 Batch:29 Cost:35.99167 Accuracy:0.5714286
    Train-Pass:73 Test-Cost:30.189571 Test-Accuracy:0.598958
    Train-Pass:74 Batch:29 Cost:45.125015 Accuracy:0.42857143
    Train-Pass:74 Test-Cost:30.187935 Test-Accuracy:0.598958
    Train-Pass:75 Batch:29 Cost:33.20766 Accuracy:0.5714286
    Train-Pass:75 Test-Cost:32.354601 Test-Accuracy:0.572917
    Train-Pass:76 Batch:29 Cost:33.257782 Accuracy:0.5714286
    Train-Pass:76 Test-Cost:28.524638 Test-Accuracy:0.625000
    Train-Pass:77 Batch:29 Cost:56.970116 Accuracy:0.2857143
    Train-Pass:77 Test-Cost:30.689668 Test-Accuracy:0.598958
    Train-Pass:78 Batch:29 Cost:47.83623 Accuracy:0.42857143
    Train-Pass:78 Test-Cost:30.189668 Test-Accuracy:0.598958
    Train-Pass:79 Batch:29 Cost:54.27631 Accuracy:0.2857143
    Train-Pass:79 Test-Cost:32.354601 Test-Accuracy:0.572917
    Train-Pass:80 Batch:29 Cost:51.543495 Accuracy:0.2857143
    Train-Pass:80 Test-Cost:32.356140 Test-Accuracy:0.572917
    Train-Pass:81 Batch:29 Cost:39.62979 Accuracy:0.42857143
    Train-Pass:81 Test-Cost:32.354602 Test-Accuracy:0.572917
    Train-Pass:82 Batch:29 Cost:42.391666 Accuracy:0.42857143
    Train-Pass:82 Test-Cost:30.189668 Test-Accuracy:0.598958
    Train-Pass:83 Batch:29 Cost:38.716576 Accuracy:0.5714286
    Train-Pass:83 Test-Cost:30.689570 Test-Accuracy:0.598958
    Train-Pass:84 Batch:29 Cost:33.23876 Accuracy:0.5714286
    Train-Pass:84 Test-Cost:30.189668 Test-Accuracy:0.598958
    Train-Pass:85 Batch:29 Cost:12.22921 Accuracy:0.85714287
    Train-Pass:85 Test-Cost:32.356140 Test-Accuracy:0.572917
    Train-Pass:86 Batch:29 Cost:42.38269 Accuracy:0.42857143
    Train-Pass:86 Test-Cost:30.689668 Test-Accuracy:0.598958
    Train-Pass:87 Batch:29 Cost:33.24881 Accuracy:0.5714286
    Train-Pass:87 Test-Cost:30.189570 Test-Accuracy:0.598958
    Train-Pass:88 Batch:29 Cost:24.11546 Accuracy:0.71428573
    Train-Pass:88 Test-Cost:30.691402 Test-Accuracy:0.598958
    Train-Pass:89 Batch:29 Cost:47.877914 Accuracy:0.42857143
    Train-Pass:89 Test-Cost:30.691304 Test-Accuracy:0.598958
    Train-Pass:90 Batch:29 Cost:42.36367 Accuracy:0.42857143
    Train-Pass:90 Test-Cost:32.856237 Test-Accuracy:0.572917
    Train-Pass:91 Batch:29 Cost:63.401222 Accuracy:0.14285715
    Train-Pass:91 Test-Cost:32.856335 Test-Accuracy:0.572917
    Train-Pass:92 Batch:29 Cost:51.525013 Accuracy:0.2857143
    Train-Pass:92 Test-Cost:30.191207 Test-Accuracy:0.598958
    Train-Pass:93 Batch:29 Cost:57.001755 Accuracy:0.2857143
    Train-Pass:93 Test-Cost:30.689668 Test-Accuracy:0.598958
    Train-Pass:94 Batch:29 Cost:35.99167 Accuracy:0.5714286
    Train-Pass:94 Test-Cost:30.189571 Test-Accuracy:0.598958
    Train-Pass:95 Batch:29 Cost:54.277916 Accuracy:0.2857143
    Train-Pass:95 Test-Cost:30.689570 Test-Accuracy:0.598958
    Train-Pass:96 Batch:29 Cost:48.79113 Accuracy:0.2857143
    Train-Pass:96 Test-Cost:30.689570 Test-Accuracy:0.598958
    

#### 2、眼底数据语义分割
##### 2.1、血管标签图像二值化
如果直接将格式转换后的格式送入模型，会发现最多有256个标签，这是因为PaddleSeg采用单通道的标注图片，每一种像素值代表一种类别，像素标注类别需要从0开始递增，例如0，1，2，3表示有4种类别，所以标注类别最多为256类。

但其实我们只需要找到血管的位置，因此血管就作为一个类，其背景作为另一个类别，这样总共有2个类别。下面来看一下如何使用opencv做图像二值化处理。


```python
# 使用opencv读取图像
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("work2/0.png") # 读取的图片路径

plt.imshow(img)
plt.show()
```


    
![png](main_files/main_15_0.png)
    



```python
# 使用opencv读取图像
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("work2/0.png") # 读取的图片路径
# 转换为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray)
plt.show()
        
```


    
![png](main_files/main_16_0.png)
    



```python
# 使用opencv读取图像
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("work2/0.png") # 读取的图片路径
# 转换为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 将灰度极差的一半作为阈值
difference = (img_gray.max() - img_gray.min()) // 2
# 将图像二值化
_, img_binary = cv2.threshold(img_gray, difference, 1, cv2.THRESH_BINARY)
print("阈值：", _)
plt.imshow(img_binary)
plt.show()
```

    阈值： 127.0
    


    
![png](main_files/main_17_1.png)
    


##### 2.2 导入数据并进行处理


```python
# 解压数据集
!unzip data/data56918/FundusVessels.zip
```

    Archive:  data/data56918/FundusVessels.zip
       creating: FundusVessels/
       creating: FundusVessels/Annotations/
      inflating: FundusVessels/Annotations/0.png  
      inflating: FundusVessels/Annotations/1.png  
      inflating: FundusVessels/Annotations/10.png  
      inflating: FundusVessels/Annotations/100.png  
      inflating: FundusVessels/Annotations/101.png  
      inflating: FundusVessels/Annotations/102.png  
      inflating: FundusVessels/Annotations/103.png  
      inflating: FundusVessels/Annotations/104.png  
      inflating: FundusVessels/Annotations/105.png  
      inflating: FundusVessels/Annotations/106.png  
      inflating: FundusVessels/Annotations/107.png  
      inflating: FundusVessels/Annotations/108.png  
      inflating: FundusVessels/Annotations/109.png  
      inflating: FundusVessels/Annotations/11.png  
      inflating: FundusVessels/Annotations/110.png  
      inflating: FundusVessels/Annotations/111.png  
      inflating: FundusVessels/Annotations/112.png  
      inflating: FundusVessels/Annotations/113.png  
      inflating: FundusVessels/Annotations/114.png  
      inflating: FundusVessels/Annotations/115.png  
      inflating: FundusVessels/Annotations/116.png  
      inflating: FundusVessels/Annotations/117.png  
      inflating: FundusVessels/Annotations/118.png  
      inflating: FundusVessels/Annotations/119.png  
      inflating: FundusVessels/Annotations/12.png  
      inflating: FundusVessels/Annotations/120.png  
      inflating: FundusVessels/Annotations/121.png  
      inflating: FundusVessels/Annotations/122.png  
      inflating: FundusVessels/Annotations/123.png  
      inflating: FundusVessels/Annotations/124.png  
      inflating: FundusVessels/Annotations/125.png  
      inflating: FundusVessels/Annotations/126.png  
      inflating: FundusVessels/Annotations/127.png  
      inflating: FundusVessels/Annotations/128.png  
      inflating: FundusVessels/Annotations/129.png  
      inflating: FundusVessels/Annotations/13.png  
      inflating: FundusVessels/Annotations/130.png  
      inflating: FundusVessels/Annotations/131.png  
      inflating: FundusVessels/Annotations/132.png  
      inflating: FundusVessels/Annotations/133.png  
      inflating: FundusVessels/Annotations/134.png  
      inflating: FundusVessels/Annotations/135.png  
      inflating: FundusVessels/Annotations/136.png  
      inflating: FundusVessels/Annotations/137.png  
      inflating: FundusVessels/Annotations/138.png  
      inflating: FundusVessels/Annotations/139.png  
      inflating: FundusVessels/Annotations/14.png  
      inflating: FundusVessels/Annotations/140.png  
      inflating: FundusVessels/Annotations/141.png  
      inflating: FundusVessels/Annotations/142.png  
      inflating: FundusVessels/Annotations/143.png  
      inflating: FundusVessels/Annotations/144.png  
      inflating: FundusVessels/Annotations/145.png  
      inflating: FundusVessels/Annotations/146.png  
      inflating: FundusVessels/Annotations/147.png  
      inflating: FundusVessels/Annotations/148.png  
      inflating: FundusVessels/Annotations/149.png  
      inflating: FundusVessels/Annotations/15.png  
      inflating: FundusVessels/Annotations/150.png  
      inflating: FundusVessels/Annotations/151.png  
      inflating: FundusVessels/Annotations/152.png  
      inflating: FundusVessels/Annotations/153.png  
      inflating: FundusVessels/Annotations/154.png  
      inflating: FundusVessels/Annotations/155.png  
      inflating: FundusVessels/Annotations/156.png  
      inflating: FundusVessels/Annotations/157.png  
      inflating: FundusVessels/Annotations/158.png  
      inflating: FundusVessels/Annotations/159.png  
      inflating: FundusVessels/Annotations/16.png  
      inflating: FundusVessels/Annotations/160.png  
      inflating: FundusVessels/Annotations/161.png  
      inflating: FundusVessels/Annotations/162.png  
      inflating: FundusVessels/Annotations/163.png  
      inflating: FundusVessels/Annotations/164.png  
      inflating: FundusVessels/Annotations/165.png  
      inflating: FundusVessels/Annotations/166.png  
      inflating: FundusVessels/Annotations/167.png  
      inflating: FundusVessels/Annotations/168.png  
      inflating: FundusVessels/Annotations/169.png  
      inflating: FundusVessels/Annotations/17.png  
      inflating: FundusVessels/Annotations/170.png  
      inflating: FundusVessels/Annotations/171.png  
      inflating: FundusVessels/Annotations/172.png  
      inflating: FundusVessels/Annotations/173.png  
      inflating: FundusVessels/Annotations/174.png  
      inflating: FundusVessels/Annotations/175.png  
      inflating: FundusVessels/Annotations/176.png  
      inflating: FundusVessels/Annotations/177.png  
      inflating: FundusVessels/Annotations/178.png  
      inflating: FundusVessels/Annotations/179.png  
      inflating: FundusVessels/Annotations/18.png  
      inflating: FundusVessels/Annotations/180.png  
      inflating: FundusVessels/Annotations/181.png  
      inflating: FundusVessels/Annotations/182.png  
      inflating: FundusVessels/Annotations/183.png  
      inflating: FundusVessels/Annotations/184.png  
      inflating: FundusVessels/Annotations/185.png  
      inflating: FundusVessels/Annotations/186.png  
      inflating: FundusVessels/Annotations/187.png  
      inflating: FundusVessels/Annotations/188.png  
      inflating: FundusVessels/Annotations/189.png  
      inflating: FundusVessels/Annotations/19.png  
      inflating: FundusVessels/Annotations/190.png  
      inflating: FundusVessels/Annotations/191.png  
      inflating: FundusVessels/Annotations/192.png  
      inflating: FundusVessels/Annotations/193.png  
      inflating: FundusVessels/Annotations/194.png  
      inflating: FundusVessels/Annotations/195.png  
      inflating: FundusVessels/Annotations/196.png  
      inflating: FundusVessels/Annotations/197.png  
      inflating: FundusVessels/Annotations/198.png  
      inflating: FundusVessels/Annotations/199.png  
      inflating: FundusVessels/Annotations/2.png  
      inflating: FundusVessels/Annotations/20.png  
      inflating: FundusVessels/Annotations/21.png  
      inflating: FundusVessels/Annotations/22.png  
      inflating: FundusVessels/Annotations/23.png  
      inflating: FundusVessels/Annotations/24.png  
      inflating: FundusVessels/Annotations/25.png  
      inflating: FundusVessels/Annotations/26.png  
      inflating: FundusVessels/Annotations/27.png  
      inflating: FundusVessels/Annotations/28.png  
      inflating: FundusVessels/Annotations/29.png  
      inflating: FundusVessels/Annotations/3.png  
      inflating: FundusVessels/Annotations/30.png  
      inflating: FundusVessels/Annotations/31.png  
      inflating: FundusVessels/Annotations/32.png  
      inflating: FundusVessels/Annotations/33.png  
      inflating: FundusVessels/Annotations/34.png  
      inflating: FundusVessels/Annotations/35.png  
      inflating: FundusVessels/Annotations/36.png  
      inflating: FundusVessels/Annotations/37.png  
      inflating: FundusVessels/Annotations/38.png  
      inflating: FundusVessels/Annotations/39.png  
      inflating: FundusVessels/Annotations/4.png  
      inflating: FundusVessels/Annotations/40.png  
      inflating: FundusVessels/Annotations/41.png  
      inflating: FundusVessels/Annotations/42.png  
      inflating: FundusVessels/Annotations/43.png  
      inflating: FundusVessels/Annotations/44.png  
      inflating: FundusVessels/Annotations/45.png  
      inflating: FundusVessels/Annotations/46.png  
      inflating: FundusVessels/Annotations/47.png  
      inflating: FundusVessels/Annotations/48.png  
      inflating: FundusVessels/Annotations/49.png  
      inflating: FundusVessels/Annotations/5.png  
      inflating: FundusVessels/Annotations/50.png  
      inflating: FundusVessels/Annotations/51.png  
      inflating: FundusVessels/Annotations/52.png  
      inflating: FundusVessels/Annotations/53.png  
      inflating: FundusVessels/Annotations/54.png  
      inflating: FundusVessels/Annotations/55.png  
      inflating: FundusVessels/Annotations/56.png  
      inflating: FundusVessels/Annotations/57.png  
      inflating: FundusVessels/Annotations/58.png  
      inflating: FundusVessels/Annotations/59.png  
      inflating: FundusVessels/Annotations/6.png  
      inflating: FundusVessels/Annotations/60.png  
      inflating: FundusVessels/Annotations/61.png  
      inflating: FundusVessels/Annotations/62.png  
      inflating: FundusVessels/Annotations/63.png  
      inflating: FundusVessels/Annotations/64.png  
      inflating: FundusVessels/Annotations/65.png  
      inflating: FundusVessels/Annotations/66.png  
      inflating: FundusVessels/Annotations/67.png  
      inflating: FundusVessels/Annotations/68.png  
      inflating: FundusVessels/Annotations/69.png  
      inflating: FundusVessels/Annotations/7.png  
      inflating: FundusVessels/Annotations/70.png  
      inflating: FundusVessels/Annotations/71.png  
      inflating: FundusVessels/Annotations/72.png  
      inflating: FundusVessels/Annotations/73.png  
      inflating: FundusVessels/Annotations/74.png  
      inflating: FundusVessels/Annotations/75.png  
      inflating: FundusVessels/Annotations/76.png  
      inflating: FundusVessels/Annotations/77.png  
      inflating: FundusVessels/Annotations/78.png  
      inflating: FundusVessels/Annotations/79.png  
      inflating: FundusVessels/Annotations/8.png  
      inflating: FundusVessels/Annotations/80.png  
      inflating: FundusVessels/Annotations/81.png  
      inflating: FundusVessels/Annotations/82.png  
      inflating: FundusVessels/Annotations/83.png  
      inflating: FundusVessels/Annotations/84.png  
      inflating: FundusVessels/Annotations/85.png  
      inflating: FundusVessels/Annotations/86.png  
      inflating: FundusVessels/Annotations/87.png  
      inflating: FundusVessels/Annotations/88.png  
      inflating: FundusVessels/Annotations/89.png  
      inflating: FundusVessels/Annotations/9.png  
      inflating: FundusVessels/Annotations/90.png  
      inflating: FundusVessels/Annotations/91.png  
      inflating: FundusVessels/Annotations/92.png  
      inflating: FundusVessels/Annotations/93.png  
      inflating: FundusVessels/Annotations/94.png  
      inflating: FundusVessels/Annotations/95.png  
      inflating: FundusVessels/Annotations/96.png  
      inflating: FundusVessels/Annotations/97.png  
      inflating: FundusVessels/Annotations/98.png  
      inflating: FundusVessels/Annotations/99.png  
       creating: FundusVessels/JPEGImages/
      inflating: FundusVessels/JPEGImages/0.jpg  
      inflating: FundusVessels/JPEGImages/1.jpg  
      inflating: FundusVessels/JPEGImages/10.jpg  
      inflating: FundusVessels/JPEGImages/100.jpg  
      inflating: FundusVessels/JPEGImages/101.jpg  
      inflating: FundusVessels/JPEGImages/102.jpg  
      inflating: FundusVessels/JPEGImages/103.jpg  
      inflating: FundusVessels/JPEGImages/104.jpg  
      inflating: FundusVessels/JPEGImages/105.jpg  
      inflating: FundusVessels/JPEGImages/106.jpg  
      inflating: FundusVessels/JPEGImages/107.jpg  
      inflating: FundusVessels/JPEGImages/108.jpg  
      inflating: FundusVessels/JPEGImages/109.jpg  
      inflating: FundusVessels/JPEGImages/11.jpg  
      inflating: FundusVessels/JPEGImages/110.jpg  
      inflating: FundusVessels/JPEGImages/111.jpg  
      inflating: FundusVessels/JPEGImages/112.jpg  
      inflating: FundusVessels/JPEGImages/113.jpg  
      inflating: FundusVessels/JPEGImages/114.jpg  
      inflating: FundusVessels/JPEGImages/115.jpg  
      inflating: FundusVessels/JPEGImages/116.jpg  
      inflating: FundusVessels/JPEGImages/117.jpg  
      inflating: FundusVessels/JPEGImages/118.jpg  
      inflating: FundusVessels/JPEGImages/119.jpg  
      inflating: FundusVessels/JPEGImages/12.jpg  
      inflating: FundusVessels/JPEGImages/120.jpg  
      inflating: FundusVessels/JPEGImages/121.jpg  
      inflating: FundusVessels/JPEGImages/122.jpg  
      inflating: FundusVessels/JPEGImages/123.jpg  
      inflating: FundusVessels/JPEGImages/124.jpg  
      inflating: FundusVessels/JPEGImages/125.jpg  
      inflating: FundusVessels/JPEGImages/126.jpg  
      inflating: FundusVessels/JPEGImages/127.jpg  
      inflating: FundusVessels/JPEGImages/128.jpg  
      inflating: FundusVessels/JPEGImages/129.jpg  
      inflating: FundusVessels/JPEGImages/13.jpg  
      inflating: FundusVessels/JPEGImages/130.jpg  
      inflating: FundusVessels/JPEGImages/131.jpg  
      inflating: FundusVessels/JPEGImages/132.jpg  
      inflating: FundusVessels/JPEGImages/133.jpg  
      inflating: FundusVessels/JPEGImages/134.jpg  
      inflating: FundusVessels/JPEGImages/135.jpg  
      inflating: FundusVessels/JPEGImages/136.jpg  
      inflating: FundusVessels/JPEGImages/137.jpg  
      inflating: FundusVessels/JPEGImages/138.jpg  
      inflating: FundusVessels/JPEGImages/139.jpg  
      inflating: FundusVessels/JPEGImages/14.jpg  
      inflating: FundusVessels/JPEGImages/140.jpg  
      inflating: FundusVessels/JPEGImages/141.jpg  
      inflating: FundusVessels/JPEGImages/142.jpg  
      inflating: FundusVessels/JPEGImages/143.jpg  
      inflating: FundusVessels/JPEGImages/144.jpg  
      inflating: FundusVessels/JPEGImages/145.jpg  
      inflating: FundusVessels/JPEGImages/146.jpg  
      inflating: FundusVessels/JPEGImages/147.jpg  
      inflating: FundusVessels/JPEGImages/148.jpg  
      inflating: FundusVessels/JPEGImages/149.jpg  
      inflating: FundusVessels/JPEGImages/15.jpg  
      inflating: FundusVessels/JPEGImages/150.jpg  
      inflating: FundusVessels/JPEGImages/151.jpg  
      inflating: FundusVessels/JPEGImages/152.jpg  
      inflating: FundusVessels/JPEGImages/153.jpg  
      inflating: FundusVessels/JPEGImages/154.jpg  
      inflating: FundusVessels/JPEGImages/155.jpg  
      inflating: FundusVessels/JPEGImages/156.jpg  
      inflating: FundusVessels/JPEGImages/157.jpg  
      inflating: FundusVessels/JPEGImages/158.jpg  
      inflating: FundusVessels/JPEGImages/159.jpg  
      inflating: FundusVessels/JPEGImages/16.jpg  
      inflating: FundusVessels/JPEGImages/160.jpg  
      inflating: FundusVessels/JPEGImages/161.jpg  
      inflating: FundusVessels/JPEGImages/162.jpg  
      inflating: FundusVessels/JPEGImages/163.jpg  
      inflating: FundusVessels/JPEGImages/164.jpg  
      inflating: FundusVessels/JPEGImages/165.jpg  
      inflating: FundusVessels/JPEGImages/166.jpg  
      inflating: FundusVessels/JPEGImages/167.jpg  
      inflating: FundusVessels/JPEGImages/168.jpg  
      inflating: FundusVessels/JPEGImages/169.jpg  
      inflating: FundusVessels/JPEGImages/17.jpg  
      inflating: FundusVessels/JPEGImages/170.jpg  
      inflating: FundusVessels/JPEGImages/171.jpg  
      inflating: FundusVessels/JPEGImages/172.jpg  
      inflating: FundusVessels/JPEGImages/173.jpg  
      inflating: FundusVessels/JPEGImages/174.jpg  
      inflating: FundusVessels/JPEGImages/175.jpg  
      inflating: FundusVessels/JPEGImages/176.jpg  
      inflating: FundusVessels/JPEGImages/177.jpg  
      inflating: FundusVessels/JPEGImages/178.jpg  
      inflating: FundusVessels/JPEGImages/179.jpg  
      inflating: FundusVessels/JPEGImages/18.jpg  
      inflating: FundusVessels/JPEGImages/180.jpg  
      inflating: FundusVessels/JPEGImages/181.jpg  
      inflating: FundusVessels/JPEGImages/182.jpg  
      inflating: FundusVessels/JPEGImages/183.jpg  
      inflating: FundusVessels/JPEGImages/184.jpg  
      inflating: FundusVessels/JPEGImages/185.jpg  
      inflating: FundusVessels/JPEGImages/186.jpg  
      inflating: FundusVessels/JPEGImages/187.jpg  
      inflating: FundusVessels/JPEGImages/188.jpg  
      inflating: FundusVessels/JPEGImages/189.jpg  
      inflating: FundusVessels/JPEGImages/19.jpg  
      inflating: FundusVessels/JPEGImages/190.jpg  
      inflating: FundusVessels/JPEGImages/191.jpg  
      inflating: FundusVessels/JPEGImages/192.jpg  
      inflating: FundusVessels/JPEGImages/193.jpg  
      inflating: FundusVessels/JPEGImages/194.jpg  
      inflating: FundusVessels/JPEGImages/195.jpg  
      inflating: FundusVessels/JPEGImages/196.jpg  
      inflating: FundusVessels/JPEGImages/197.jpg  
      inflating: FundusVessels/JPEGImages/198.jpg  
      inflating: FundusVessels/JPEGImages/199.jpg  
      inflating: FundusVessels/JPEGImages/2.jpg  
      inflating: FundusVessels/JPEGImages/20.jpg  
      inflating: FundusVessels/JPEGImages/21.jpg  
      inflating: FundusVessels/JPEGImages/22.jpg  
      inflating: FundusVessels/JPEGImages/23.jpg  
      inflating: FundusVessels/JPEGImages/24.jpg  
      inflating: FundusVessels/JPEGImages/25.jpg  
      inflating: FundusVessels/JPEGImages/26.jpg  
      inflating: FundusVessels/JPEGImages/27.jpg  
      inflating: FundusVessels/JPEGImages/28.jpg  
      inflating: FundusVessels/JPEGImages/29.jpg  
      inflating: FundusVessels/JPEGImages/3.jpg  
      inflating: FundusVessels/JPEGImages/30.jpg  
      inflating: FundusVessels/JPEGImages/31.jpg  
      inflating: FundusVessels/JPEGImages/32.jpg  
      inflating: FundusVessels/JPEGImages/33.jpg  
      inflating: FundusVessels/JPEGImages/34.jpg  
      inflating: FundusVessels/JPEGImages/35.jpg  
      inflating: FundusVessels/JPEGImages/36.jpg  
      inflating: FundusVessels/JPEGImages/37.jpg  
      inflating: FundusVessels/JPEGImages/38.jpg  
      inflating: FundusVessels/JPEGImages/39.jpg  
      inflating: FundusVessels/JPEGImages/4.jpg  
      inflating: FundusVessels/JPEGImages/40.jpg  
      inflating: FundusVessels/JPEGImages/41.jpg  
      inflating: FundusVessels/JPEGImages/42.jpg  
      inflating: FundusVessels/JPEGImages/43.jpg  
      inflating: FundusVessels/JPEGImages/44.jpg  
      inflating: FundusVessels/JPEGImages/45.jpg  
      inflating: FundusVessels/JPEGImages/46.jpg  
      inflating: FundusVessels/JPEGImages/47.jpg  
      inflating: FundusVessels/JPEGImages/48.jpg  
      inflating: FundusVessels/JPEGImages/49.jpg  
      inflating: FundusVessels/JPEGImages/5.jpg  
      inflating: FundusVessels/JPEGImages/50.jpg  
      inflating: FundusVessels/JPEGImages/51.jpg  
      inflating: FundusVessels/JPEGImages/52.jpg  
      inflating: FundusVessels/JPEGImages/53.jpg  
      inflating: FundusVessels/JPEGImages/54.jpg  
      inflating: FundusVessels/JPEGImages/55.jpg  
      inflating: FundusVessels/JPEGImages/56.jpg  
      inflating: FundusVessels/JPEGImages/57.jpg  
      inflating: FundusVessels/JPEGImages/58.jpg  
      inflating: FundusVessels/JPEGImages/59.jpg  
      inflating: FundusVessels/JPEGImages/6.jpg  
      inflating: FundusVessels/JPEGImages/60.jpg  
      inflating: FundusVessels/JPEGImages/61.jpg  
      inflating: FundusVessels/JPEGImages/62.jpg  
      inflating: FundusVessels/JPEGImages/63.jpg  
      inflating: FundusVessels/JPEGImages/64.jpg  
      inflating: FundusVessels/JPEGImages/65.jpg  
      inflating: FundusVessels/JPEGImages/66.jpg  
      inflating: FundusVessels/JPEGImages/67.jpg  
      inflating: FundusVessels/JPEGImages/68.jpg  
      inflating: FundusVessels/JPEGImages/69.jpg  
      inflating: FundusVessels/JPEGImages/7.jpg  
      inflating: FundusVessels/JPEGImages/70.jpg  
      inflating: FundusVessels/JPEGImages/71.jpg  
      inflating: FundusVessels/JPEGImages/72.jpg  
      inflating: FundusVessels/JPEGImages/73.jpg  
      inflating: FundusVessels/JPEGImages/74.jpg  
      inflating: FundusVessels/JPEGImages/75.jpg  
      inflating: FundusVessels/JPEGImages/76.jpg  
      inflating: FundusVessels/JPEGImages/77.jpg  
      inflating: FundusVessels/JPEGImages/78.jpg  
      inflating: FundusVessels/JPEGImages/79.jpg  
      inflating: FundusVessels/JPEGImages/8.jpg  
      inflating: FundusVessels/JPEGImages/80.jpg  
      inflating: FundusVessels/JPEGImages/81.jpg  
      inflating: FundusVessels/JPEGImages/82.jpg  
      inflating: FundusVessels/JPEGImages/83.jpg  
      inflating: FundusVessels/JPEGImages/84.jpg  
      inflating: FundusVessels/JPEGImages/85.jpg  
      inflating: FundusVessels/JPEGImages/86.jpg  
      inflating: FundusVessels/JPEGImages/87.jpg  
      inflating: FundusVessels/JPEGImages/88.jpg  
      inflating: FundusVessels/JPEGImages/89.jpg  
      inflating: FundusVessels/JPEGImages/9.jpg  
      inflating: FundusVessels/JPEGImages/90.jpg  
      inflating: FundusVessels/JPEGImages/91.jpg  
      inflating: FundusVessels/JPEGImages/92.jpg  
      inflating: FundusVessels/JPEGImages/93.jpg  
      inflating: FundusVessels/JPEGImages/94.jpg  
      inflating: FundusVessels/JPEGImages/95.jpg  
      inflating: FundusVessels/JPEGImages/96.jpg  
      inflating: FundusVessels/JPEGImages/97.jpg  
      inflating: FundusVessels/JPEGImages/98.jpg  
      inflating: FundusVessels/JPEGImages/99.jpg  
    


```python
# 生成图像列表
import os

path_origin = 'FundusVessels/JPEGImages/'
path_seg = 'FundusVessels/Annotations/'
pic_dir = os.listdir(path_origin)

f_train = open('train_list.txt', 'w')
f_val = open('val_list.txt', 'w')

for i in range(len(pic_dir)):
    if i % 30 != 0:
        f_train.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')
    else:
        f_val.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')

f_train.close()
f_val.close()
```


```python
# 解压从PaddleSeg Github仓库下载好的压缩包
!unzip -o work2/PaddleSeg.zip
# 运行脚本需在PaddleSeg目录下
%cd PaddleSeg
# 安装所需依赖项
!pip install -r requirements.txt
```

    Archive:  work2/PaddleSeg.zip
       creating: PaddleSeg/
       creating: PaddleSeg/pretrained_model/
      inflating: PaddleSeg/pretrained_model/download_model.py  
      inflating: PaddleSeg/.copyright.hook  
       creating: PaddleSeg/tutorial/
      inflating: PaddleSeg/tutorial/finetune_icnet.md  
       creating: PaddleSeg/tutorial/imgs/
      inflating: PaddleSeg/tutorial/imgs/optic_icnet.png  
      inflating: PaddleSeg/tutorial/imgs/optic_deeplab.png  
      inflating: PaddleSeg/tutorial/imgs/optic_unet.png  
      inflating: PaddleSeg/tutorial/imgs/optic.png  
      inflating: PaddleSeg/tutorial/imgs/optic_pspnet.png  
      inflating: PaddleSeg/tutorial/imgs/optic_hrnet.png  
      inflating: PaddleSeg/tutorial/finetune_hrnet.md  
      inflating: PaddleSeg/tutorial/finetune_unet.md  
      inflating: PaddleSeg/tutorial/finetune_pspnet.md  
      inflating: PaddleSeg/tutorial/finetune_fast_scnn.md  
      inflating: PaddleSeg/tutorial/finetune_deeplabv3plus.md  
      inflating: PaddleSeg/tutorial/finetune_ocrnet.md  
      inflating: PaddleSeg/LICENSE       
      inflating: PaddleSeg/requirements.txt  
       creating: PaddleSeg/test/
      inflating: PaddleSeg/test/test_utils.py  
       creating: PaddleSeg/test/ci/
      inflating: PaddleSeg/test/ci/test_download_dataset.sh  
      inflating: PaddleSeg/test/ci/check_code_style.sh  
      inflating: PaddleSeg/test/local_test_pet.py  
      inflating: PaddleSeg/test/local_test_cityscapes.py  
       creating: PaddleSeg/test/configs/
      inflating: PaddleSeg/test/configs/unet_pet.yaml  
      inflating: PaddleSeg/test/configs/deeplabv3p_xception65_cityscapes.yaml  
      inflating: PaddleSeg/.pre-commit-config.yaml  
       creating: PaddleSeg/deploy/
       creating: PaddleSeg/deploy/serving/
       creating: PaddleSeg/deploy/serving/tools/
       creating: PaddleSeg/deploy/serving/tools/images/
      inflating: PaddleSeg/deploy/serving/tools/images/2.jpg  
      inflating: PaddleSeg/deploy/serving/tools/images/3.jpg  
      inflating: PaddleSeg/deploy/serving/tools/images/1.jpg  
      inflating: PaddleSeg/deploy/serving/tools/image_seg_client.py  
       creating: PaddleSeg/deploy/serving/seg-serving/
      inflating: PaddleSeg/deploy/serving/seg-serving/CMakeLists.txt  
       creating: PaddleSeg/deploy/serving/seg-serving/proto/
      inflating: PaddleSeg/deploy/serving/seg-serving/proto/CMakeLists.txt  
      inflating: PaddleSeg/deploy/serving/seg-serving/proto/image_seg.proto  
       creating: PaddleSeg/deploy/serving/seg-serving/op/
      inflating: PaddleSeg/deploy/serving/seg-serving/op/image_seg_op.cpp  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/CMakeLists.txt  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/image_seg_op.h  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/write_json_op.cpp  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/write_json_op.h  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/reader_op.h  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/seg_conf.cpp  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/reader_op.cpp  
      inflating: PaddleSeg/deploy/serving/seg-serving/op/seg_conf.h  
       creating: PaddleSeg/deploy/serving/seg-serving/scripts/
      inflating: PaddleSeg/deploy/serving/seg-serving/scripts/start.sh  
       creating: PaddleSeg/deploy/serving/seg-serving/data/
       creating: PaddleSeg/deploy/serving/seg-serving/data/model/
       creating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/
      inflating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/fluid_reload_flag  
      inflating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/fluid_time_file  
       creating: PaddleSeg/deploy/serving/seg-serving/conf/
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/workflow.prototxt  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/seg_conf.yaml  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/gflags.conf  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/seg_conf2.yaml  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/resource.prototxt  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/service.prototxt  
      inflating: PaddleSeg/deploy/serving/seg-serving/conf/model_toolkit.prototxt  
      inflating: PaddleSeg/deploy/serving/requirements.txt  
      inflating: PaddleSeg/deploy/serving/COMPILE_GUIDE.md  
      inflating: PaddleSeg/deploy/serving/UBUNTU.md  
      inflating: PaddleSeg/deploy/serving/README.md  
       creating: PaddleSeg/deploy/python/
      inflating: PaddleSeg/deploy/python/requirements.txt  
       creating: PaddleSeg/deploy/python/docs/
      inflating: PaddleSeg/deploy/python/docs/compile_paddle_with_tensorrt.md  
      inflating: PaddleSeg/deploy/python/docs/PaddleSeg_Infer_Benchmark.md  
      inflating: PaddleSeg/deploy/python/README.md  
      inflating: PaddleSeg/deploy/python/infer.py  
       creating: PaddleSeg/deploy/lite/
       creating: PaddleSeg/deploy/lite/example/
      inflating: PaddleSeg/deploy/lite/example/human_2.png  
      inflating: PaddleSeg/deploy/lite/example/human_3.png  
      inflating: PaddleSeg/deploy/lite/example/human_1.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/proguard-rules.pro  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/local.properties  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/gradle-wrapper.jar  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/gradle-wrapper.properties  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradlew  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/.gitignore  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/build.gradle  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradlew.bat  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/demo/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/demo/ExampleInstrumentedTest.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/demo/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/demo/ExampleUnitTest.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/ic_launcher.png  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/ic_launcher_round.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable-v24/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable-v24/ic_launcher_foreground.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/ic_launcher.png  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/ic_launcher_round.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable/ic_launcher_background.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/layout/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/layout/activity_main.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/ic_launcher.png  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/ic_launcher_round.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/colors.xml  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/arrays.xml  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/styles.xml  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/strings.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/xml/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/xml/settings.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/menu/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/menu/menu_action_options.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/ic_launcher.png  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/ic_launcher_round.png  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/AndroidManifest.xml  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/AppCompatPreferenceActivity.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/config/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/config/Config.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/preprocess/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/preprocess/Preprocess.java  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/MainActivity.java  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/Utils.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/visual/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/visual/Visualize.java  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/Predictor.java  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/SettingsActivity.java  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/images/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/images/human.jpg  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/labels/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/labels/label_list  
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/
       creating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/gradle-wrapper.jar  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/gradle-wrapper.properties  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradlew  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/.gitignore  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/build.gradle  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle.properties  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradlew.bat  
      inflating: PaddleSeg/deploy/lite/human_segmentation_demo/settings.gradle  
      inflating: PaddleSeg/deploy/lite/README.md  
      inflating: PaddleSeg/deploy/README.md  
       creating: PaddleSeg/deploy/cpp/
       creating: PaddleSeg/deploy/cpp/tools/
      inflating: PaddleSeg/deploy/cpp/tools/visualize.py  
      inflating: PaddleSeg/deploy/cpp/CMakeLists.txt  
      inflating: PaddleSeg/deploy/cpp/LICENSE  
       creating: PaddleSeg/deploy/cpp/images/
       creating: PaddleSeg/deploy/cpp/images/humanseg/
      inflating: PaddleSeg/deploy/cpp/images/humanseg/demo3.jpeg  
      inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2.jpeg  
      inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2_jpeg_recover.png  
      inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2.jpeg_result.png  
      inflating: PaddleSeg/deploy/cpp/images/humanseg/demo1.jpeg  
      inflating: PaddleSeg/deploy/cpp/INSTALL.md  
       creating: PaddleSeg/deploy/cpp/utils/
      inflating: PaddleSeg/deploy/cpp/utils/utils.h  
      inflating: PaddleSeg/deploy/cpp/utils/seg_conf_parser.h  
       creating: PaddleSeg/deploy/cpp/docs/
      inflating: PaddleSeg/deploy/cpp/docs/vis_result.png  
      inflating: PaddleSeg/deploy/cpp/docs/demo.jpg  
      inflating: PaddleSeg/deploy/cpp/docs/windows_vs2015_build.md  
      inflating: PaddleSeg/deploy/cpp/docs/vis.md  
      inflating: PaddleSeg/deploy/cpp/docs/demo_jpg.png  
      inflating: PaddleSeg/deploy/cpp/docs/windows_vs2019_build.md  
      inflating: PaddleSeg/deploy/cpp/docs/linux_build.md  
      inflating: PaddleSeg/deploy/cpp/README.md  
       creating: PaddleSeg/deploy/cpp/external-cmake/
      inflating: PaddleSeg/deploy/cpp/external-cmake/yaml-cpp.cmake  
      inflating: PaddleSeg/deploy/cpp/demo.cpp  
       creating: PaddleSeg/deploy/cpp/predictor/
      inflating: PaddleSeg/deploy/cpp/predictor/seg_predictor.cpp  
      inflating: PaddleSeg/deploy/cpp/predictor/seg_predictor.h  
       creating: PaddleSeg/deploy/cpp/conf/
      inflating: PaddleSeg/deploy/cpp/conf/humanseg.yaml  
       creating: PaddleSeg/deploy/cpp/preprocessor/
      inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor.cpp  
      inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor_seg.cpp  
      inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor_seg.h  
      inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor.h  
      inflating: PaddleSeg/deploy/cpp/CMakeSettings.json  
       creating: PaddleSeg/dataset/
      inflating: PaddleSeg/dataset/download_mini_deepglobe_road_extraction.py  
      inflating: PaddleSeg/dataset/download_pet.py  
      inflating: PaddleSeg/dataset/README.md  
      inflating: PaddleSeg/dataset/download_and_convert_voc2012.py  
      inflating: PaddleSeg/dataset/download_cityscapes.py  
      inflating: PaddleSeg/dataset/download_optic.py  
      inflating: PaddleSeg/dataset/convert_voc2012.py  
       creating: PaddleSeg/dygraph/
       creating: PaddleSeg/dygraph/benchmark/
      inflating: PaddleSeg/dygraph/benchmark/hrnet.py  
      inflating: PaddleSeg/dygraph/benchmark/deeplabv3p.py  
       creating: PaddleSeg/dygraph/tools/
      inflating: PaddleSeg/dygraph/tools/conver_cityscapes.py  
      inflating: PaddleSeg/dygraph/tools/voc_augment.py  
       creating: PaddleSeg/dygraph/core/
      inflating: PaddleSeg/dygraph/core/val.py  
      inflating: PaddleSeg/dygraph/core/__init__.py  
      inflating: PaddleSeg/dygraph/core/train.py  
      inflating: PaddleSeg/dygraph/core/infer.py  
       creating: PaddleSeg/dygraph/cvlibs/
      inflating: PaddleSeg/dygraph/cvlibs/__init__.py  
      inflating: PaddleSeg/dygraph/cvlibs/manager.py  
      inflating: PaddleSeg/dygraph/val.py  
       creating: PaddleSeg/dygraph/datasets/
      inflating: PaddleSeg/dygraph/datasets/cityscapes.py  
      inflating: PaddleSeg/dygraph/datasets/ade.py  
      inflating: PaddleSeg/dygraph/datasets/__init__.py  
      inflating: PaddleSeg/dygraph/datasets/dataset.py  
      inflating: PaddleSeg/dygraph/datasets/voc.py  
      inflating: PaddleSeg/dygraph/datasets/optic_disc_seg.py  
      inflating: PaddleSeg/dygraph/__init__.py  
       creating: PaddleSeg/dygraph/utils/
      inflating: PaddleSeg/dygraph/utils/metrics.py  
      inflating: PaddleSeg/dygraph/utils/timer.py  
      inflating: PaddleSeg/dygraph/utils/download.py  
      inflating: PaddleSeg/dygraph/utils/__init__.py  
      inflating: PaddleSeg/dygraph/utils/logger.py  
      inflating: PaddleSeg/dygraph/utils/utils.py  
      inflating: PaddleSeg/dygraph/utils/get_environ_info.py  
       creating: PaddleSeg/dygraph/models/
      inflating: PaddleSeg/dygraph/models/model_utils.py  
      inflating: PaddleSeg/dygraph/models/fcn.py  
      inflating: PaddleSeg/dygraph/models/deeplab.py  
      inflating: PaddleSeg/dygraph/models/unet.py  
      inflating: PaddleSeg/dygraph/models/__init__.py  
       creating: PaddleSeg/dygraph/models/architectures/
      inflating: PaddleSeg/dygraph/models/architectures/layer_utils.py  
      inflating: PaddleSeg/dygraph/models/architectures/hrnet.py  
      inflating: PaddleSeg/dygraph/models/architectures/xception_deeplab.py  
      inflating: PaddleSeg/dygraph/models/architectures/__init__.py  
      inflating: PaddleSeg/dygraph/models/architectures/mobilenetv3.py  
      inflating: PaddleSeg/dygraph/models/architectures/resnet_vd.py  
      inflating: PaddleSeg/dygraph/models/pspnet.py  
      inflating: PaddleSeg/dygraph/README.md  
       creating: PaddleSeg/dygraph/transforms/
      inflating: PaddleSeg/dygraph/transforms/transforms.py  
      inflating: PaddleSeg/dygraph/transforms/__init__.py  
      inflating: PaddleSeg/dygraph/transforms/functional.py  
      inflating: PaddleSeg/dygraph/train.py  
      inflating: PaddleSeg/dygraph/infer.py  
       creating: PaddleSeg/pdseg/
      inflating: PaddleSeg/pdseg/export_model.py  
      inflating: PaddleSeg/pdseg/metrics.py  
       creating: PaddleSeg/pdseg/tools/
      inflating: PaddleSeg/pdseg/tools/create_dataset_list.py  
      inflating: PaddleSeg/pdseg/tools/__init__.py  
      inflating: PaddleSeg/pdseg/tools/labelme2seg.py  
      inflating: PaddleSeg/pdseg/tools/jingling2seg.py  
      inflating: PaddleSeg/pdseg/tools/gray2pseudo_color.py  
      inflating: PaddleSeg/pdseg/check.py  
      inflating: PaddleSeg/pdseg/solver.py  
      inflating: PaddleSeg/pdseg/__init__.py  
       creating: PaddleSeg/pdseg/utils/
      inflating: PaddleSeg/pdseg/utils/collect.py  
      inflating: PaddleSeg/pdseg/utils/config.py  
      inflating: PaddleSeg/pdseg/utils/timer.py  
      inflating: PaddleSeg/pdseg/utils/__init__.py  
      inflating: PaddleSeg/pdseg/utils/fp16_utils.py  
      inflating: PaddleSeg/pdseg/utils/dist_utils.py  
      inflating: PaddleSeg/pdseg/utils/load_model_utils.py  
       creating: PaddleSeg/pdseg/models/
      inflating: PaddleSeg/pdseg/models/model_builder.py  
      inflating: PaddleSeg/pdseg/models/__init__.py  
       creating: PaddleSeg/pdseg/models/modeling/
      inflating: PaddleSeg/pdseg/models/modeling/hrnet.py  
      inflating: PaddleSeg/pdseg/models/modeling/deeplab.py  
      inflating: PaddleSeg/pdseg/models/modeling/unet.py  
      inflating: PaddleSeg/pdseg/models/modeling/icnet.py  
      inflating: PaddleSeg/pdseg/models/modeling/__init__.py  
      inflating: PaddleSeg/pdseg/models/modeling/fast_scnn.py  
      inflating: PaddleSeg/pdseg/models/modeling/pspnet.py  
      inflating: PaddleSeg/pdseg/models/modeling/ocrnet.py  
       creating: PaddleSeg/pdseg/models/libs/
      inflating: PaddleSeg/pdseg/models/libs/__init__.py  
      inflating: PaddleSeg/pdseg/models/libs/model_libs.py  
       creating: PaddleSeg/pdseg/models/backbone/
      inflating: PaddleSeg/pdseg/models/backbone/vgg.py  
      inflating: PaddleSeg/pdseg/models/backbone/mobilenet_v2.py  
      inflating: PaddleSeg/pdseg/models/backbone/__init__.py  
      inflating: PaddleSeg/pdseg/models/backbone/mobilenet_v3.py  
      inflating: PaddleSeg/pdseg/models/backbone/resnet.py  
      inflating: PaddleSeg/pdseg/models/backbone/resnet_vd.py  
      inflating: PaddleSeg/pdseg/models/backbone/xception.py  
      inflating: PaddleSeg/pdseg/vis.py  
      inflating: PaddleSeg/pdseg/reader.py  
      inflating: PaddleSeg/pdseg/loss.py  
      inflating: PaddleSeg/pdseg/data_utils.py  
      inflating: PaddleSeg/pdseg/data_aug.py  
      inflating: PaddleSeg/pdseg/train.py  
      inflating: PaddleSeg/pdseg/eval.py  
      inflating: PaddleSeg/pdseg/lovasz_losses.py  
       creating: PaddleSeg/docs/
      inflating: PaddleSeg/docs/data_aug.md  
       creating: PaddleSeg/docs/imgs/
      inflating: PaddleSeg/docs/imgs/deepglobe.png  
      inflating: PaddleSeg/docs/imgs/usage_vis_demo.jpg  
      inflating: PaddleSeg/docs/imgs/fast-scnn.png  
      inflating: PaddleSeg/docs/imgs/lovasz-hinge.png  
      inflating: PaddleSeg/docs/imgs/loss_comparison.png  
      inflating: PaddleSeg/docs/imgs/lovasz-softmax.png  
      inflating: PaddleSeg/docs/imgs/rangescale.png  
      inflating: PaddleSeg/docs/imgs/qq_group2.png  
      inflating: PaddleSeg/docs/imgs/pspnet2.png  
      inflating: PaddleSeg/docs/imgs/hrnet.png  
      inflating: PaddleSeg/docs/imgs/lovasz-hinge-vis.png  
      inflating: PaddleSeg/docs/imgs/file_list.png  
      inflating: PaddleSeg/docs/imgs/unet.png  
       creating: PaddleSeg/docs/imgs/annotation/
      inflating: PaddleSeg/docs/imgs/annotation/image-4-1.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-6-2.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-4-2.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-10.jpg  
      inflating: PaddleSeg/docs/imgs/annotation/image-11.png  
      inflating: PaddleSeg/docs/imgs/annotation/jingling-5.png  
      inflating: PaddleSeg/docs/imgs/annotation/jingling-4.png  
      inflating: PaddleSeg/docs/imgs/annotation/jingling-1.png  
      inflating: PaddleSeg/docs/imgs/annotation/jingling-3.png  
      inflating: PaddleSeg/docs/imgs/annotation/jingling-2.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-1.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-3.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-2.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-6.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-7.png  
      inflating: PaddleSeg/docs/imgs/annotation/image-5.png  
      inflating: PaddleSeg/docs/imgs/file_list2.png  
      inflating: PaddleSeg/docs/imgs/warmup_with_poly_decay_example.png  
      inflating: PaddleSeg/docs/imgs/data_aug_example.png  
      inflating: PaddleSeg/docs/imgs/dice3.png  
      inflating: PaddleSeg/docs/imgs/dice2.png  
      inflating: PaddleSeg/docs/imgs/softmax_loss.png  
      inflating: PaddleSeg/docs/imgs/gn.png  
      inflating: PaddleSeg/docs/imgs/dice.png  
      inflating: PaddleSeg/docs/imgs/pspnet.png  
      inflating: PaddleSeg/docs/imgs/icnet.png  
      inflating: PaddleSeg/docs/imgs/visualdl_scalar.png  
      inflating: PaddleSeg/docs/imgs/aug_method.png  
      inflating: PaddleSeg/docs/imgs/data_aug_flip_mirror.png  
      inflating: PaddleSeg/docs/imgs/poly_decay_example.png  
      inflating: PaddleSeg/docs/imgs/piecewise_decay_example.png  
      inflating: PaddleSeg/docs/imgs/VOC2012.png  
      inflating: PaddleSeg/docs/imgs/visualdl_image.png  
      inflating: PaddleSeg/docs/imgs/cosine_decay_example.png  
      inflating: PaddleSeg/docs/imgs/data_aug_flow.png  
      inflating: PaddleSeg/docs/imgs/deeplabv3p.png  
      inflating: PaddleSeg/docs/usage.md  
      inflating: PaddleSeg/docs/lovasz_loss.md  
      inflating: PaddleSeg/docs/data_prepare.md  
       creating: PaddleSeg/docs/annotation/
      inflating: PaddleSeg/docs/annotation/labelme2seg.md  
       creating: PaddleSeg/docs/annotation/cityscapes_demo/
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/cityscapes_demo_dataset.yaml  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/train_list.txt  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/val_list.txt  
       creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/stuttgart_000021_000019_gtFine_labelTrainIds.png  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/stuttgart_000072_000019_gtFine_labelTrainIds.png  
       creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/frankfurt_000001_063045_gtFine_labelTrainIds.png  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/frankfurt_000001_062250_gtFine_labelTrainIds.png  
       creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/stuttgart_000072_000019_leftImg8bit.png  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/stuttgart_000021_000019_leftImg8bit.png  
       creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/
       creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/frankfurt_000001_062250_leftImg8bit.png  
      inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/frankfurt_000001_063045_leftImg8bit.png  
      inflating: PaddleSeg/docs/annotation/jingling2seg.md  
       creating: PaddleSeg/docs/annotation/jingling_demo/
      inflating: PaddleSeg/docs/annotation/jingling_demo/jingling.jpg  
       creating: PaddleSeg/docs/annotation/jingling_demo/outputs/
       creating: PaddleSeg/docs/annotation/jingling_demo/outputs/annotations/
      inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/annotations/jingling.png  
      inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/jingling.json  
      inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/class_names.txt  
       creating: PaddleSeg/docs/annotation/labelme_demo/
      inflating: PaddleSeg/docs/annotation/labelme_demo/2011_000025.jpg  
      inflating: PaddleSeg/docs/annotation/labelme_demo/2011_000025.json  
      inflating: PaddleSeg/docs/annotation/labelme_demo/class_names.txt  
      inflating: PaddleSeg/docs/check.md  
      inflating: PaddleSeg/docs/dice_loss.md  
      inflating: PaddleSeg/docs/config.md  
      inflating: PaddleSeg/docs/deploy.md  
      inflating: PaddleSeg/docs/models.md  
       creating: PaddleSeg/docs/configs/
      inflating: PaddleSeg/docs/configs/model_hrnet_group.md  
      inflating: PaddleSeg/docs/configs/model_pspnet_group.md  
      inflating: PaddleSeg/docs/configs/model_deeplabv3p_group.md  
      inflating: PaddleSeg/docs/configs/.gitkeep  
      inflating: PaddleSeg/docs/configs/model_unet_group.md  
      inflating: PaddleSeg/docs/configs/model_group.md  
      inflating: PaddleSeg/docs/configs/test_group.md  
      inflating: PaddleSeg/docs/configs/train_group.md  
      inflating: PaddleSeg/docs/configs/dataloader_group.md  
      inflating: PaddleSeg/docs/configs/model_icnet_group.md  
      inflating: PaddleSeg/docs/configs/freeze_group.md  
      inflating: PaddleSeg/docs/configs/solver_group.md  
      inflating: PaddleSeg/docs/configs/basic_group.md  
      inflating: PaddleSeg/docs/configs/dataset_group.md  
      inflating: PaddleSeg/docs/multiple_gpus_train_and_mixed_precision_train.md  
      inflating: PaddleSeg/docs/model_export.md  
      inflating: PaddleSeg/docs/model_zoo.md  
      inflating: PaddleSeg/docs/loss_select.md  
       creating: PaddleSeg/contrib/
       creating: PaddleSeg/contrib/MechanicalIndustryMeter/
       creating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/
      inflating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.png  
      inflating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.JPG  
      inflating: PaddleSeg/contrib/MechanicalIndustryMeter/unet_mechanical_meter.yaml  
      inflating: PaddleSeg/contrib/MechanicalIndustryMeter/download_unet_mechanical_industry_meter.py  
      inflating: PaddleSeg/contrib/MechanicalIndustryMeter/download_mini_mechanical_industry_meter.py  
       creating: PaddleSeg/contrib/SpatialEmbeddings/
      inflating: PaddleSeg/contrib/SpatialEmbeddings/config.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/models.py  
       creating: PaddleSeg/contrib/SpatialEmbeddings/imgs/
      inflating: PaddleSeg/contrib/SpatialEmbeddings/imgs/kitti_0007_000518_ori.png  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/imgs/kitti_0007_000518_pred.png  
       creating: PaddleSeg/contrib/SpatialEmbeddings/utils/
      inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/util.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/data_util.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/__init__.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/palette.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/download_SpatialEmbeddings_kitti.py  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/README.md  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/infer.py  
       creating: PaddleSeg/contrib/SpatialEmbeddings/data/
       creating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/
       creating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/
      inflating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/kitti_0007_000518.png  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/kitti_0007_000512.png  
      inflating: PaddleSeg/contrib/SpatialEmbeddings/data/test.txt  
       creating: PaddleSeg/contrib/RoadLine/
      inflating: PaddleSeg/contrib/RoadLine/config.py  
       creating: PaddleSeg/contrib/RoadLine/imgs/
      inflating: PaddleSeg/contrib/RoadLine/imgs/RoadLine.jpg  
      inflating: PaddleSeg/contrib/RoadLine/imgs/RoadLine.png  
      inflating: PaddleSeg/contrib/RoadLine/__init__.py  
       creating: PaddleSeg/contrib/RoadLine/utils/
      inflating: PaddleSeg/contrib/RoadLine/utils/util.py  
      inflating: PaddleSeg/contrib/RoadLine/utils/__init__.py  
      inflating: PaddleSeg/contrib/RoadLine/utils/palette.py  
      inflating: PaddleSeg/contrib/RoadLine/infer.py  
      inflating: PaddleSeg/contrib/RoadLine/download_RoadLine.py  
      inflating: PaddleSeg/contrib/README.md  
       creating: PaddleSeg/contrib/LaneNet/
      inflating: PaddleSeg/contrib/LaneNet/requirements.txt  
       creating: PaddleSeg/contrib/LaneNet/imgs/
      inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_lane.png  
      inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_binary.png  
      inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_instance.png  
       creating: PaddleSeg/contrib/LaneNet/dataset/
      inflating: PaddleSeg/contrib/LaneNet/dataset/download_tusimple.py  
       creating: PaddleSeg/contrib/LaneNet/utils/
      inflating: PaddleSeg/contrib/LaneNet/utils/config.py  
      inflating: PaddleSeg/contrib/LaneNet/utils/__init__.py  
      inflating: PaddleSeg/contrib/LaneNet/utils/lanenet_postprocess.py  
      inflating: PaddleSeg/contrib/LaneNet/utils/generate_tusimple_dataset.py  
      inflating: PaddleSeg/contrib/LaneNet/utils/dist_utils.py  
      inflating: PaddleSeg/contrib/LaneNet/utils/load_model_utils.py  
       creating: PaddleSeg/contrib/LaneNet/models/
      inflating: PaddleSeg/contrib/LaneNet/models/model_builder.py  
      inflating: PaddleSeg/contrib/LaneNet/models/__init__.py  
       creating: PaddleSeg/contrib/LaneNet/models/modeling/
      inflating: PaddleSeg/contrib/LaneNet/models/modeling/__init__.py  
      inflating: PaddleSeg/contrib/LaneNet/models/modeling/lanenet.py  
      inflating: PaddleSeg/contrib/LaneNet/vis.py  
      inflating: PaddleSeg/contrib/LaneNet/README.md  
      inflating: PaddleSeg/contrib/LaneNet/reader.py  
      inflating: PaddleSeg/contrib/LaneNet/loss.py  
      inflating: PaddleSeg/contrib/LaneNet/data_aug.py  
       creating: PaddleSeg/contrib/LaneNet/configs/
      inflating: PaddleSeg/contrib/LaneNet/configs/lanenet.yaml  
      inflating: PaddleSeg/contrib/LaneNet/train.py  
      inflating: PaddleSeg/contrib/LaneNet/eval.py  
       creating: PaddleSeg/contrib/ACE2P/
      inflating: PaddleSeg/contrib/ACE2P/config.py  
       creating: PaddleSeg/contrib/ACE2P/imgs/
      inflating: PaddleSeg/contrib/ACE2P/imgs/result.jpg  
      inflating: PaddleSeg/contrib/ACE2P/imgs/net.jpg  
      inflating: PaddleSeg/contrib/ACE2P/imgs/117676_2149260.jpg  
      inflating: PaddleSeg/contrib/ACE2P/imgs/117676_2149260.png  
      inflating: PaddleSeg/contrib/ACE2P/download_ACE2P.py  
      inflating: PaddleSeg/contrib/ACE2P/__init__.py  
       creating: PaddleSeg/contrib/ACE2P/utils/
      inflating: PaddleSeg/contrib/ACE2P/utils/util.py  
      inflating: PaddleSeg/contrib/ACE2P/utils/__init__.py  
      inflating: PaddleSeg/contrib/ACE2P/utils/palette.py  
      inflating: PaddleSeg/contrib/ACE2P/README.md  
      inflating: PaddleSeg/contrib/ACE2P/reader.py  
      inflating: PaddleSeg/contrib/ACE2P/infer.py  
       creating: PaddleSeg/contrib/HumanSeg/
      inflating: PaddleSeg/contrib/HumanSeg/video_infer.py  
      inflating: PaddleSeg/contrib/HumanSeg/quant_online.py  
      inflating: PaddleSeg/contrib/HumanSeg/requirements.txt  
      inflating: PaddleSeg/contrib/HumanSeg/val.py  
       creating: PaddleSeg/contrib/HumanSeg/datasets/
      inflating: PaddleSeg/contrib/HumanSeg/datasets/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/datasets/dataset.py  
       creating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/
      inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/sharedmemory.py  
      inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/queue.py  
      inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/bg_replace.py  
      inflating: PaddleSeg/contrib/HumanSeg/quant_offline.py  
       creating: PaddleSeg/contrib/HumanSeg/utils/
      inflating: PaddleSeg/contrib/HumanSeg/utils/logging.py  
      inflating: PaddleSeg/contrib/HumanSeg/utils/metrics.py  
      inflating: PaddleSeg/contrib/HumanSeg/utils/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/utils/humanseg_postprocess.py  
      inflating: PaddleSeg/contrib/HumanSeg/utils/utils.py  
      inflating: PaddleSeg/contrib/HumanSeg/utils/post_quantization.py  
       creating: PaddleSeg/contrib/HumanSeg/models/
      inflating: PaddleSeg/contrib/HumanSeg/models/load_model.py  
      inflating: PaddleSeg/contrib/HumanSeg/models/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/models/humanseg.py  
      inflating: PaddleSeg/contrib/HumanSeg/export.py  
      inflating: PaddleSeg/contrib/HumanSeg/README.md  
       creating: PaddleSeg/contrib/HumanSeg/nets/
      inflating: PaddleSeg/contrib/HumanSeg/nets/hrnet.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/shufflenet_slim.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/seg_modules.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/deeplabv3p.py  
       creating: PaddleSeg/contrib/HumanSeg/nets/backbone/
      inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/mobilenet_v2.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/xception.py  
      inflating: PaddleSeg/contrib/HumanSeg/nets/libs.py  
       creating: PaddleSeg/contrib/HumanSeg/transforms/
      inflating: PaddleSeg/contrib/HumanSeg/transforms/transforms.py  
      inflating: PaddleSeg/contrib/HumanSeg/transforms/__init__.py  
      inflating: PaddleSeg/contrib/HumanSeg/transforms/functional.py  
      inflating: PaddleSeg/contrib/HumanSeg/train.py  
      inflating: PaddleSeg/contrib/HumanSeg/infer.py  
       creating: PaddleSeg/contrib/HumanSeg/data/
      inflating: PaddleSeg/contrib/HumanSeg/data/background.jpg  
      inflating: PaddleSeg/contrib/HumanSeg/data/download_data.py  
      inflating: PaddleSeg/contrib/HumanSeg/data/human_image.jpg  
       creating: PaddleSeg/contrib/HumanSeg/pretrained_weights/
      inflating: PaddleSeg/contrib/HumanSeg/pretrained_weights/download_pretrained_weights.py  
       creating: PaddleSeg/contrib/RemoteSensing/
      inflating: PaddleSeg/contrib/RemoteSensing/visualize_demo.py  
       creating: PaddleSeg/contrib/RemoteSensing/tools/
      inflating: PaddleSeg/contrib/RemoteSensing/tools/data_analyse_and_check.py  
      inflating: PaddleSeg/contrib/RemoteSensing/tools/cal_norm_coef.py  
      inflating: PaddleSeg/contrib/RemoteSensing/tools/data_distribution_vis.py  
      inflating: PaddleSeg/contrib/RemoteSensing/tools/create_dataset_list.py  
      inflating: PaddleSeg/contrib/RemoteSensing/tools/split_dataset_list.py  
      inflating: PaddleSeg/contrib/RemoteSensing/requirements.txt  
      inflating: PaddleSeg/contrib/RemoteSensing/__init__.py  
       creating: PaddleSeg/contrib/RemoteSensing/utils/
      inflating: PaddleSeg/contrib/RemoteSensing/utils/logging.py  
      inflating: PaddleSeg/contrib/RemoteSensing/utils/metrics.py  
      inflating: PaddleSeg/contrib/RemoteSensing/utils/pretrain_weights.py  
      inflating: PaddleSeg/contrib/RemoteSensing/utils/__init__.py  
      inflating: PaddleSeg/contrib/RemoteSensing/utils/utils.py  
       creating: PaddleSeg/contrib/RemoteSensing/models/
      inflating: PaddleSeg/contrib/RemoteSensing/models/load_model.py  
      inflating: PaddleSeg/contrib/RemoteSensing/models/hrnet.py  
      inflating: PaddleSeg/contrib/RemoteSensing/models/unet.py  
      inflating: PaddleSeg/contrib/RemoteSensing/models/__init__.py  
       creating: PaddleSeg/contrib/RemoteSensing/models/utils/
      inflating: PaddleSeg/contrib/RemoteSensing/models/utils/visualize.py  
      inflating: PaddleSeg/contrib/RemoteSensing/models/base.py  
       creating: PaddleSeg/contrib/RemoteSensing/docs/
       creating: PaddleSeg/contrib/RemoteSensing/docs/imgs/
      inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/dataset.png  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/visualdl.png  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/data_distribution.png  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/vis.png  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/data_prepare.md  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/transforms.md  
      inflating: PaddleSeg/contrib/RemoteSensing/docs/data_analyse_and_check.md  
      inflating: PaddleSeg/contrib/RemoteSensing/README.md  
       creating: PaddleSeg/contrib/RemoteSensing/nets/
      inflating: PaddleSeg/contrib/RemoteSensing/nets/hrnet.py  
      inflating: PaddleSeg/contrib/RemoteSensing/nets/unet.py  
      inflating: PaddleSeg/contrib/RemoteSensing/nets/__init__.py  
      inflating: PaddleSeg/contrib/RemoteSensing/nets/loss.py  
      inflating: PaddleSeg/contrib/RemoteSensing/nets/libs.py  
       creating: PaddleSeg/contrib/RemoteSensing/transforms/
      inflating: PaddleSeg/contrib/RemoteSensing/transforms/transforms.py  
      inflating: PaddleSeg/contrib/RemoteSensing/transforms/__init__.py  
      inflating: PaddleSeg/contrib/RemoteSensing/transforms/ops.py  
      inflating: PaddleSeg/contrib/RemoteSensing/predict_demo.py  
       creating: PaddleSeg/contrib/RemoteSensing/readers/
      inflating: PaddleSeg/contrib/RemoteSensing/readers/__init__.py  
      inflating: PaddleSeg/contrib/RemoteSensing/readers/reader.py  
      inflating: PaddleSeg/contrib/RemoteSensing/readers/base.py  
      inflating: PaddleSeg/contrib/RemoteSensing/train_demo.py  
      inflating: PaddleSeg/README.md     
      inflating: PaddleSeg/.gitignore    
      inflating: PaddleSeg/.style.yapf   
       creating: PaddleSeg/configs/
      inflating: PaddleSeg/configs/lovasz_hinge_deeplabv3p_mobilenet_road.yaml  
      inflating: PaddleSeg/configs/cityscape_fast_scnn.yaml  
      inflating: PaddleSeg/configs/unet_optic.yaml  
      inflating: PaddleSeg/configs/deepglobe_road_extraction.yaml  
      inflating: PaddleSeg/configs/hrnet_optic.yaml  
      inflating: PaddleSeg/configs/ocrnet_w18_bn_cityscapes.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_mobilenetv3_large_cityscapes.yaml  
      inflating: PaddleSeg/configs/icnet_optic.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_resnet50_vd_cityscapes.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_xception65_optic.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_xception65_cityscapes.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_mobilenet-1-0_pet.yaml  
      inflating: PaddleSeg/configs/fast_scnn_pet.yaml  
      inflating: PaddleSeg/configs/lovasz_softmax_deeplabv3p_mobilenet_pascal.yaml  
      inflating: PaddleSeg/configs/deeplabv3p_mobilenetv2_cityscapes.yaml  
      inflating: PaddleSeg/configs/pspnet_optic.yaml  
       creating: PaddleSeg/slim/
       creating: PaddleSeg/slim/nas/
      inflating: PaddleSeg/slim/nas/train_nas.py  
      inflating: PaddleSeg/slim/nas/model_builder.py  
      inflating: PaddleSeg/slim/nas/mobilenetv2_search_space.py  
      inflating: PaddleSeg/slim/nas/deeplab.py  
      inflating: PaddleSeg/slim/nas/README.md  
      inflating: PaddleSeg/slim/nas/eval_nas.py  
       creating: PaddleSeg/slim/distillation/
      inflating: PaddleSeg/slim/distillation/model_builder.py  
      inflating: PaddleSeg/slim/distillation/README.md  
      inflating: PaddleSeg/slim/distillation/train_distill.py  
      inflating: PaddleSeg/slim/distillation/cityscape_teacher.yaml  
      inflating: PaddleSeg/slim/distillation/cityscape.yaml  
       creating: PaddleSeg/slim/quantization/
      inflating: PaddleSeg/slim/quantization/export_model.py  
       creating: PaddleSeg/slim/quantization/images/
      inflating: PaddleSeg/slim/quantization/images/TransformPass.png  
      inflating: PaddleSeg/slim/quantization/images/ConvertToInt8Pass.png  
      inflating: PaddleSeg/slim/quantization/images/FreezePass.png  
      inflating: PaddleSeg/slim/quantization/images/TransformForMobilePass.png  
      inflating: PaddleSeg/slim/quantization/train_quant.py  
      inflating: PaddleSeg/slim/quantization/eval_quant.py  
      inflating: PaddleSeg/slim/quantization/README.md  
       creating: PaddleSeg/slim/prune/
      inflating: PaddleSeg/slim/prune/train_prune.py  
      inflating: PaddleSeg/slim/prune/README.md  
      inflating: PaddleSeg/slim/prune/eval_prune.py  
       creating: PaddleSeg/.git/
      inflating: PaddleSeg/.git/config   
       creating: PaddleSeg/.git/objects/
       creating: PaddleSeg/.git/objects/pack/
      inflating: PaddleSeg/.git/objects/pack/pack-0b5aa212c914b01e77066dc5729dffedd245ed42.pack  
      inflating: PaddleSeg/.git/objects/pack/pack-0b5aa212c914b01e77066dc5729dffedd245ed42.idx  
       creating: PaddleSeg/.git/objects/info/
      inflating: PaddleSeg/.git/HEAD     
       creating: PaddleSeg/.git/info/
      inflating: PaddleSeg/.git/info/exclude  
       creating: PaddleSeg/.git/logs/
      inflating: PaddleSeg/.git/logs/HEAD  
       creating: PaddleSeg/.git/logs/refs/
       creating: PaddleSeg/.git/logs/refs/heads/
       creating: PaddleSeg/.git/logs/refs/heads/release/
      inflating: PaddleSeg/.git/logs/refs/heads/release/v0.6.0  
       creating: PaddleSeg/.git/logs/refs/remotes/
       creating: PaddleSeg/.git/logs/refs/remotes/origin/
      inflating: PaddleSeg/.git/logs/refs/remotes/origin/HEAD  
      inflating: PaddleSeg/.git/description  
       creating: PaddleSeg/.git/hooks/
      inflating: PaddleSeg/.git/hooks/commit-msg.sample  
      inflating: PaddleSeg/.git/hooks/pre-rebase.sample  
      inflating: PaddleSeg/.git/hooks/pre-commit.sample  
      inflating: PaddleSeg/.git/hooks/applypatch-msg.sample  
      inflating: PaddleSeg/.git/hooks/fsmonitor-watchman.sample  
      inflating: PaddleSeg/.git/hooks/pre-receive.sample  
      inflating: PaddleSeg/.git/hooks/prepare-commit-msg.sample  
      inflating: PaddleSeg/.git/hooks/post-update.sample  
      inflating: PaddleSeg/.git/hooks/pre-applypatch.sample  
      inflating: PaddleSeg/.git/hooks/pre-push.sample  
      inflating: PaddleSeg/.git/hooks/update.sample  
       creating: PaddleSeg/.git/refs/
       creating: PaddleSeg/.git/refs/heads/
       creating: PaddleSeg/.git/refs/heads/release/
      inflating: PaddleSeg/.git/refs/heads/release/v0.6.0  
       creating: PaddleSeg/.git/refs/tags/
       creating: PaddleSeg/.git/refs/remotes/
       creating: PaddleSeg/.git/refs/remotes/origin/
      inflating: PaddleSeg/.git/refs/remotes/origin/HEAD  
      inflating: PaddleSeg/.git/index    
      inflating: PaddleSeg/.git/packed-refs  
      inflating: PaddleSeg/.travis.yml   
    /home/aistudio/PaddleSeg
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (1.21.0)
    Requirement already satisfied: yapf==0.26.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (0.26.0)
    Requirement already satisfied: flake8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (4.0.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (5.1.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (2.2.0)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (16.7.9)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.16.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.4)
    Requirement already satisfied: importlib-metadata in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (4.2.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.4.10)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (2.0.1)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (0.6.1)
    Requirement already satisfied: pycodestyle<2.9.0,>=2.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.8.0)
    Requirement already satisfied: pyflakes<2.5.0,>=2.4.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.4.0)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (2.2.3)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (3.14.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.0.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.1)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.5)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.8.53)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (2.24.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.7.1.1)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.19.5)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (8.2.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (7.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.16.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.11.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->pre-commit->-r requirements.txt (line 1)) (4.0.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->pre-commit->-r requirements.txt (line 1)) (3.7.0)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.9.9)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.0.7)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.10.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.25.6)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.9.11)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.0.1)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (56.2.0)
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m
    

##### 2.3 配置文件进行训练

这里我们需要使用两个文件:
- 训练文件：PaddleSeg/pdseg/train.py
- 训练的配置文件：PaddleSeg/configs/unet_optic.yaml

这里使用U-Net，大家可以尝试使用其他网络进行配置。

######  .yaml文件的配置

下面是笔者在查找有关资料后配置的.yaml文件，
~~~
# 数据集配置
DATASET:
    DATA_DIR: ""
    NUM_CLASSES: 2
    TEST_FILE_LIST: "train_list.txt"
    TRAIN_FILE_LIST: "train_list.txt"
    VAL_FILE_LIST: "val_list.txt"
    VIS_FILE_LIST: "train_list.txt"

# 预训练模型配置
MODEL:
    MODEL_NAME: "unet"
    DEFAULT_NORM_TYPE: "bn"

# 其他配置
TRAIN_CROP_SIZE: (565, 584)
EVAL_CROP_SIZE: (565, 584)
AUG:
    AUG_METHOD: "unpadding"
    FIX_RESIZE_SIZE: (565, 584)
BATCH_SIZE: 4
TRAIN:
    # PRETRAINED_MODEL_DIR: "./pretrained_model/unet_bn_coco/"
    MODEL_SAVE_DIR: "./saved_model/unet_optic/"
    SNAPSHOT_EPOCH: 2
TEST:
    TEST_MODEL: "./saved_model/unet_optic/final"
SOLVER:
    NUM_EPOCHS: 10
    LR: 0.001
    LR_POLICY: "poly"
    OPTIMIZER: "adam"

~~~

###### 开始训练

训练命令的格式参考：
~~~
python PaddleSeg/pdseg/train.py --cfg configs/unet_optic.yaml \
                      --use_gpu \
                      --do_eval \
                      --use_vdl \
                      --vdl_log_dir train_log \
                      BATCH_SIZE 4 \
                      SOLVER.LR 0.001
                      
~~~


```python
import os
os.chdir('/home/aistudio')
import paddle
#记得在train.py设置paddle.enable_static()

!python PaddleSeg/pdseg/train.py --cfg PaddleSeg/configs/unet_optic.yaml \
                      --use_gpu \
                      --do_eval \
                      --use_vdl \
                      --vdl_log_dir train_log \
                      BATCH_SIZE 4 \
                      SOLVER.LR 0.001
```

    {'AUG': {'AUG_METHOD': 'unpadding',
             'FIX_RESIZE_SIZE': (565, 584),
             'FLIP': False,
             'FLIP_RATIO': 0.5,
             'INF_RESIZE_VALUE': 500,
             'MAX_RESIZE_VALUE': 600,
             'MAX_SCALE_FACTOR': 2.0,
             'MIN_RESIZE_VALUE': 400,
             'MIN_SCALE_FACTOR': 0.5,
             'MIRROR': True,
             'RICH_CROP': {'ASPECT_RATIO': 0.33,
                           'BLUR': False,
                           'BLUR_RATIO': 0.1,
                           'BRIGHTNESS_JITTER_RATIO': 0.5,
                           'CONTRAST_JITTER_RATIO': 0.5,
                           'ENABLE': False,
                           'MAX_ROTATION': 15,
                           'MIN_AREA_RATIO': 0.5,
                           'SATURATION_JITTER_RATIO': 0.5},
             'SCALE_STEP_SIZE': 0.25,
             'TO_RGB': False},
     'BATCH_SIZE': 4,
     'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
     'DATASET': {'DATA_DIM': 3,
                 'DATA_DIR': '',
                 'IGNORE_INDEX': 255,
                 'IMAGE_TYPE': 'rgb',
                 'NUM_CLASSES': 2,
                 'PADDING_VALUE': [127.5, 127.5, 127.5],
                 'SEPARATOR': ' ',
                 'TEST_FILE_LIST': 'train_list.txt',
                 'TEST_TOTAL_IMAGES': 193,
                 'TRAIN_FILE_LIST': 'train_list.txt',
                 'TRAIN_TOTAL_IMAGES': 193,
                 'VAL_FILE_LIST': 'val_list.txt',
                 'VAL_TOTAL_IMAGES': 7,
                 'VIS_FILE_LIST': 'train_list.txt'},
     'EVAL_CROP_SIZE': (565, 584),
     'FREEZE': {'MODEL_FILENAME': '__model__',
                'PARAMS_FILENAME': '__params__',
                'SAVE_DIR': 'freeze_model'},
     'MEAN': [0.5, 0.5, 0.5],
     'MODEL': {'BN_MOMENTUM': 0.99,
               'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                           'BACKBONE': 'xception_65',
                           'BACKBONE_LR_MULT_LIST': None,
                           'DECODER': {'CONV_FILTERS': 256,
                                       'OUTPUT_IS_LOGITS': False,
                                       'USE_SUM_MERGE': False},
                           'DECODER_USE_SEP_CONV': True,
                           'DEPTH_MULTIPLIER': 1.0,
                           'ENABLE_DECODER': True,
                           'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                       'ASPP_CONVS_FILTERS': 256,
                                       'ASPP_RATIOS': None,
                                       'ASPP_WITH_CONCAT_PROJECTION': True,
                                       'ASPP_WITH_SE': False,
                                       'POOLING_CROP_SIZE': None,
                                       'POOLING_STRIDE': [1, 1],
                                       'SE_USE_QSIGMOID': False},
                           'ENCODER_WITH_ASPP': True,
                           'OUTPUT_STRIDE': 16},
               'DEFAULT_EPSILON': 1e-05,
               'DEFAULT_GROUP_NUMBER': 32,
               'DEFAULT_NORM_TYPE': 'bn',
               'FP16': False,
               'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                         'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                    'NUM_MODULES': 4},
                         'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                    'NUM_MODULES': 3}},
               'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
               'MODEL_NAME': 'unet',
               'MULTI_LOSS_WEIGHT': [1.0],
               'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
               'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
               'SCALE_LOSS': 'DYNAMIC',
               'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
     'NUM_TRAINERS': 1,
     'SLIM': {'KNOWLEDGE_DISTILL': False,
              'KNOWLEDGE_DISTILL_IS_TEACHER': False,
              'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
              'NAS_ADDRESS': '',
              'NAS_IS_SERVER': True,
              'NAS_PORT': 23333,
              'NAS_SEARCH_STEPS': 100,
              'NAS_SPACE_NAME': '',
              'NAS_START_EVAL_EPOCH': 0,
              'PREPROCESS': False,
              'PRUNE_PARAMS': '',
              'PRUNE_RATIOS': []},
     'SOLVER': {'BEGIN_EPOCH': 1,
                'CROSS_ENTROPY_WEIGHT': None,
                'DECAY_EPOCH': [10, 20],
                'GAMMA': 0.1,
                'LOSS': ['softmax_loss'],
                'LOSS_WEIGHT': {'BCE_LOSS': 1,
                                'DICE_LOSS': 1,
                                'LOVASZ_HINGE_LOSS': 1,
                                'LOVASZ_SOFTMAX_LOSS': 1,
                                'SOFTMAX_LOSS': 1},
                'LR': 0.001,
                'LR_POLICY': 'poly',
                'LR_WARMUP': False,
                'LR_WARMUP_STEPS': 2000,
                'MOMENTUM': 0.9,
                'MOMENTUM2': 0.999,
                'NUM_EPOCHS': 10,
                'OPTIMIZER': 'adam',
                'POWER': 0.9,
                'WEIGHT_DECAY': 4e-05},
     'STD': [0.5, 0.5, 0.5],
     'TEST': {'TEST_MODEL': './saved_model/unet_optic/final'},
     'TRAIN': {'MODEL_SAVE_DIR': './saved_model/unet_optic/',
               'PRETRAINED_MODEL_DIR': '',
               'RESUME_MODEL_DIR': '',
               'SNAPSHOT_EPOCH': 2,
               'SYNC_BATCH_NORM': False},
     'TRAINER_ID': 0,
     'TRAIN_CROP_SIZE': (565, 584)}
    #Device count: 1
    batch_size_per_dev: 4
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:341: UserWarning: /home/aistudio/PaddleSeg/pdseg/loss.py:79
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    W0226 21:33:56.674086  1864 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0226 21:33:56.678936  1864 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    Pretrained model dir  not exists, training from scratch...
    Use multi-thread reader
    epoch=1 step=10 lr=0.00098 loss=0.4076 step/sec=2.834 | ETA 00:02:45
    epoch=1 step=20 lr=0.00097 loss=0.2499 step/sec=3.165 | ETA 00:02:25
    epoch=1 step=30 lr=0.00095 loss=0.2200 step/sec=3.171 | ETA 00:02:21
    epoch=1 step=40 lr=0.00093 loss=0.2107 step/sec=3.168 | ETA 00:02:18
    epoch=2 step=50 lr=0.00091 loss=0.1868 step/sec=3.070 | ETA 00:02:20
    epoch=2 step=60 lr=0.00089 loss=0.1741 step/sec=3.165 | ETA 00:02:12
    epoch=2 step=70 lr=0.00087 loss=0.1868 step/sec=3.154 | ETA 00:02:09
    epoch=2 step=80 lr=0.00085 loss=0.1773 step/sec=3.150 | ETA 00:02:06
    epoch=2 step=90 lr=0.00084 loss=0.1839 step/sec=3.148 | ETA 00:02:03
    Save model checkpoint to ./saved_model/unet_optic/2
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/2
    [EVAL]step=1 loss=0.54110 acc=0.8682 IoU=0.5495 step/sec=3.30 | ETA 00:00:14
    [EVAL]step=2 loss=0.44164 acc=0.8744 IoU=0.5501 step/sec=7.82 | ETA 00:00:06
    [EVAL]#image=7 acc=0.8744 IoU=0.5501
    [EVAL]Category IoU: [0.8695 0.2308]
    [EVAL]Category Acc: [0.9030 0.5136]
    [EVAL]Kappa:0.3108
    Save best model ./saved_model/unet_optic/2 to ./saved_model/unet_optic/best_model, mIoU = 0.5501
    load test model: ./saved_model/unet_optic/2
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 2
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 2
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 2
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 2
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 2
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 2
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 2
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 2
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 2
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 2
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 2
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 2
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 2
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 2
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 2
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 2
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 2
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 2
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 2
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 2
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 2
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 2
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 2
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 2
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 2
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 2
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 2
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 2
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 2
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 2
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 2
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 2
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 2
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 2
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 2
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 2
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 2
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 2
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 2
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 2
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 2
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 2
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 2
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 2
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 2
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 2
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 2
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 2
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 2
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 2
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 2
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 2
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 2
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 2
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 2
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 2
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 2
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 2
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 2
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 2
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 2
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 2
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 2
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 2
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 2
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 2
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 2
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 2
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 2
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 2
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 2
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 2
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 2
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 2
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 2
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 2
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 2
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 2
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 2
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 2
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 2
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 2
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 2
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 2
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 2
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 2
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 2
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 2
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 2
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 2
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 2
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 2
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 2
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 2
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 2
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 2
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 2
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 2
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 2
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 2
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 2
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 2
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 2
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 2
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 2
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 2
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 2
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 2
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 2
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 2
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 2
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 2
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 2
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 2
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 2
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 2
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 2
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 2
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 2
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 2
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 2
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 2
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 2
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 2
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 2
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 2
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 2
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 2
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 2
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 2
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 2
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 2
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 2
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 2
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 2
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 2
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 2
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 2
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 2
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 2
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 2
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 2
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 2
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 2
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 2
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 2
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 2
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 2
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 2
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 2
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 2
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 2
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 2
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 2
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 2
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 2
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 2
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 2
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 2
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 2
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 2
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 2
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 2
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 2
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 2
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 2
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 2
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 2
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 2
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 2
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 2
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 2
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 2
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 2
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 2
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 2
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 2
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 2
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 2
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 2
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 2
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 2
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 2
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 2
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 2
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 2
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 2
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 2
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 2
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 2
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 2
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 2
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 2
    epoch=3 step=100 lr=0.00082 loss=0.1738 step/sec=0.413 | ETA 00:15:19
    epoch=3 step=110 lr=0.00080 loss=0.1645 step/sec=3.156 | ETA 00:01:57
    epoch=3 step=120 lr=0.00078 loss=0.1667 step/sec=3.144 | ETA 00:01:54
    epoch=3 step=130 lr=0.00076 loss=0.1560 step/sec=3.139 | ETA 00:01:51
    epoch=3 step=140 lr=0.00074 loss=0.1578 step/sec=3.142 | ETA 00:01:48
    epoch=4 step=150 lr=0.00072 loss=0.1640 step/sec=3.044 | ETA 00:01:48
    epoch=4 step=160 lr=0.00070 loss=0.1556 step/sec=3.130 | ETA 00:01:42
    epoch=4 step=170 lr=0.00068 loss=0.1510 step/sec=3.133 | ETA 00:01:38
    epoch=4 step=180 lr=0.00066 loss=0.1560 step/sec=3.124 | ETA 00:01:36
    epoch=4 step=190 lr=0.00065 loss=0.1403 step/sec=3.126 | ETA 00:01:32
    Save model checkpoint to ./saved_model/unet_optic/4
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/4
    [EVAL]step=1 loss=0.24602 acc=0.9009 IoU=0.5799 step/sec=3.43 | ETA 00:00:13
    [EVAL]step=2 loss=0.25008 acc=0.9019 IoU=0.5806 step/sec=7.61 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9019 IoU=0.5806
    [EVAL]Category IoU: [0.8983 0.2630]
    [EVAL]Category Acc: [0.9035 0.8639]
    [EVAL]Kappa:0.3782
    Save best model ./saved_model/unet_optic/4 to ./saved_model/unet_optic/best_model, mIoU = 0.5806
    load test model: ./saved_model/unet_optic/4
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 4
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 4
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 4
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 4
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 4
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 4
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 4
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 4
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 4
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 4
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 4
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 4
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 4
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 4
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 4
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 4
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 4
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 4
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 4
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 4
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 4
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 4
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 4
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 4
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 4
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 4
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 4
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 4
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 4
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 4
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 4
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 4
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 4
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 4
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 4
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 4
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 4
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 4
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 4
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 4
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 4
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 4
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 4
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 4
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 4
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 4
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 4
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 4
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 4
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 4
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 4
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 4
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 4
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 4
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 4
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 4
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 4
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 4
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 4
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 4
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 4
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 4
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 4
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 4
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 4
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 4
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 4
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 4
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 4
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 4
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 4
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 4
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 4
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 4
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 4
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 4
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 4
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 4
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 4
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 4
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 4
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 4
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 4
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 4
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 4
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 4
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 4
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 4
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 4
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 4
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 4
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 4
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 4
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 4
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 4
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 4
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 4
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 4
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 4
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 4
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 4
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 4
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 4
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 4
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 4
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 4
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 4
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 4
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 4
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 4
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 4
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 4
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 4
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 4
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 4
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 4
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 4
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 4
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 4
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 4
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 4
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 4
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 4
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 4
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 4
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 4
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 4
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 4
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 4
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 4
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 4
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 4
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 4
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 4
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 4
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 4
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 4
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 4
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 4
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 4
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 4
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 4
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 4
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 4
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 4
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 4
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 4
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 4
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 4
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 4
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 4
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 4
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 4
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 4
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 4
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 4
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 4
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 4
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 4
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 4
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 4
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 4
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 4
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 4
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 4
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 4
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 4
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 4
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 4
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 4
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 4
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 4
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 4
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 4
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 4
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 4
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 4
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 4
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 4
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 4
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 4
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 4
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 4
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 4
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 4
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 4
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 4
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 4
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 4
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 4
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 4
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 4
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 4
    epoch=5 step=200 lr=0.00063 loss=0.1578 step/sec=0.416 | ETA 00:11:12
    epoch=5 step=210 lr=0.00061 loss=0.1491 step/sec=3.137 | ETA 00:01:26
    epoch=5 step=220 lr=0.00059 loss=0.1639 step/sec=3.134 | ETA 00:01:22
    epoch=5 step=230 lr=0.00057 loss=0.1447 step/sec=3.129 | ETA 00:01:19
    epoch=5 step=240 lr=0.00055 loss=0.1364 step/sec=3.130 | ETA 00:01:16
    epoch=6 step=250 lr=0.00053 loss=0.1493 step/sec=3.033 | ETA 00:01:15
    epoch=6 step=260 lr=0.00051 loss=0.1435 step/sec=3.126 | ETA 00:01:10
    epoch=6 step=270 lr=0.00049 loss=0.1447 step/sec=3.124 | ETA 00:01:07
    epoch=6 step=280 lr=0.00047 loss=0.1343 step/sec=3.122 | ETA 00:01:04
    Save model checkpoint to ./saved_model/unet_optic/6
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/6
    [EVAL]step=1 loss=0.24909 acc=0.9026 IoU=0.5693 step/sec=3.75 | ETA 00:00:12
    [EVAL]step=2 loss=0.23822 acc=0.9054 IoU=0.5830 step/sec=7.77 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9054 IoU=0.5830
    [EVAL]Category IoU: [0.9021 0.2640]
    [EVAL]Category Acc: [0.9029 0.9743]
    [EVAL]Kappa:0.3840
    Save best model ./saved_model/unet_optic/6 to ./saved_model/unet_optic/best_model, mIoU = 0.5830
    load test model: ./saved_model/unet_optic/6
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 6
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 6
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 6
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 6
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 6
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 6
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 6
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 6
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 6
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 6
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 6
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 6
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 6
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 6
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 6
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 6
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 6
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 6
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 6
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 6
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 6
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 6
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 6
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 6
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 6
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 6
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 6
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 6
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 6
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 6
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 6
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 6
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 6
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 6
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 6
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 6
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 6
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 6
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 6
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 6
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 6
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 6
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 6
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 6
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 6
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 6
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 6
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 6
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 6
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 6
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 6
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 6
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 6
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 6
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 6
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 6
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 6
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 6
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 6
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 6
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 6
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 6
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 6
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 6
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 6
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 6
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 6
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 6
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 6
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 6
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 6
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 6
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 6
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 6
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 6
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 6
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 6
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 6
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 6
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 6
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 6
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 6
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 6
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 6
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 6
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 6
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 6
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 6
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 6
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 6
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 6
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 6
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 6
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 6
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 6
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 6
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 6
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 6
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 6
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 6
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 6
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 6
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 6
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 6
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 6
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 6
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 6
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 6
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 6
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 6
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 6
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 6
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 6
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 6
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 6
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 6
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 6
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 6
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 6
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 6
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 6
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 6
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 6
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 6
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 6
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 6
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 6
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 6
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 6
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 6
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 6
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 6
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 6
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 6
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 6
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 6
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 6
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 6
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 6
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 6
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 6
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 6
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 6
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 6
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 6
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 6
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 6
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 6
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 6
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 6
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 6
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 6
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 6
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 6
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 6
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 6
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 6
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 6
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 6
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 6
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 6
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 6
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 6
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 6
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 6
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 6
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 6
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 6
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 6
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 6
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 6
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 6
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 6
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 6
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 6
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 6
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 6
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 6
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 6
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 6
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 6
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 6
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 6
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 6
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 6
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 6
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 6
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 6
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 6
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 6
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 6
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 6
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 6
    epoch=7 step=290 lr=0.00045 loss=0.1372 step/sec=0.414 | ETA 00:07:39
    epoch=7 step=300 lr=0.00043 loss=0.1437 step/sec=3.136 | ETA 00:00:57
    epoch=7 step=310 lr=0.00041 loss=0.1347 step/sec=3.132 | ETA 00:00:54
    epoch=7 step=320 lr=0.00039 loss=0.1403 step/sec=3.131 | ETA 00:00:51
    epoch=7 step=330 lr=0.00037 loss=0.1322 step/sec=3.126 | ETA 00:00:47
    epoch=8 step=340 lr=0.00035 loss=0.1413 step/sec=3.037 | ETA 00:00:46
    epoch=8 step=350 lr=0.00033 loss=0.1263 step/sec=3.114 | ETA 00:00:41
    epoch=8 step=360 lr=0.00031 loss=0.1418 step/sec=3.120 | ETA 00:00:38
    epoch=8 step=370 lr=0.00029 loss=0.1286 step/sec=3.121 | ETA 00:00:35
    epoch=8 step=380 lr=0.00026 loss=0.1303 step/sec=3.121 | ETA 00:00:32
    Save model checkpoint to ./saved_model/unet_optic/8
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/8
    [EVAL]step=1 loss=0.22327 acc=0.9124 IoU=0.6139 step/sec=3.76 | ETA 00:00:12
    [EVAL]step=2 loss=0.21361 acc=0.9148 IoU=0.6250 step/sec=7.66 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9148 IoU=0.6250
    [EVAL]Category IoU: [0.9109 0.3391]
    [EVAL]Category Acc: [0.9122 0.9694]
    [EVAL]Kappa:0.4713
    Save best model ./saved_model/unet_optic/8 to ./saved_model/unet_optic/best_model, mIoU = 0.6250
    load test model: ./saved_model/unet_optic/8
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 8
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 8
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 8
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 8
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 8
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 8
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 8
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 8
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 8
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 8
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 8
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 8
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 8
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 8
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 8
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 8
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 8
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 8
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 8
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 8
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 8
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 8
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 8
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 8
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 8
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 8
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 8
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 8
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 8
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 8
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 8
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 8
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 8
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 8
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 8
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 8
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 8
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 8
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 8
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 8
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 8
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 8
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 8
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 8
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 8
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 8
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 8
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 8
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 8
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 8
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 8
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 8
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 8
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 8
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 8
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 8
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 8
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 8
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 8
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 8
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 8
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 8
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 8
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 8
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 8
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 8
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 8
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 8
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 8
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 8
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 8
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 8
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 8
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 8
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 8
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 8
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 8
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 8
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 8
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 8
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 8
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 8
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 8
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 8
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 8
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 8
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 8
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 8
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 8
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 8
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 8
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 8
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 8
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 8
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 8
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 8
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 8
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 8
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 8
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 8
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 8
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 8
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 8
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 8
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 8
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 8
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 8
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 8
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 8
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 8
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 8
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 8
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 8
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 8
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 8
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 8
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 8
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 8
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 8
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 8
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 8
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 8
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 8
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 8
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 8
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 8
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 8
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 8
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 8
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 8
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 8
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 8
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 8
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 8
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 8
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 8
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 8
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 8
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 8
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 8
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 8
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 8
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 8
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 8
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 8
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 8
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 8
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 8
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 8
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 8
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 8
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 8
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 8
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 8
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 8
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 8
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 8
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 8
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 8
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 8
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 8
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 8
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 8
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 8
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 8
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 8
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 8
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 8
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 8
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 8
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 8
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 8
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 8
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 8
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 8
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 8
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 8
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 8
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 8
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 8
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 8
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 8
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 8
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 8
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 8
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 8
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 8
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 8
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 8
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 8
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 8
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 8
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 8
    epoch=9 step=390 lr=0.00024 loss=0.1224 step/sec=0.415 | ETA 00:03:36
    epoch=9 step=400 lr=0.00022 loss=0.1359 step/sec=3.138 | ETA 00:00:25
    epoch=9 step=410 lr=0.00020 loss=0.1295 step/sec=3.131 | ETA 00:00:22
    epoch=9 step=420 lr=0.00018 loss=0.1295 step/sec=3.127 | ETA 00:00:19
    epoch=9 step=430 lr=0.00016 loss=0.1317 step/sec=3.123 | ETA 00:00:16
    epoch=10 step=440 lr=0.00013 loss=0.1261 step/sec=3.031 | ETA 00:00:13
    epoch=10 step=450 lr=0.00011 loss=0.1230 step/sec=3.123 | ETA 00:00:09
    epoch=10 step=460 lr=0.00009 loss=0.1389 step/sec=3.125 | ETA 00:00:06
    epoch=10 step=470 lr=0.00006 loss=0.1321 step/sec=3.120 | ETA 00:00:03
    epoch=10 step=480 lr=0.00004 loss=0.1192 step/sec=3.121 | ETA 00:00:00
    Save model checkpoint to ./saved_model/unet_optic/10
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/10
    [EVAL]step=1 loss=0.18547 acc=0.9277 IoU=0.6886 step/sec=3.79 | ETA 00:00:12
    [EVAL]step=2 loss=0.16598 acc=0.9317 IoU=0.7055 step/sec=6.88 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9317 IoU=0.7055
    [EVAL]Category IoU: [0.9271 0.4839]
    [EVAL]Category Acc: [0.9317 0.9320]
    [EVAL]Kappa:0.6181
    Save best model ./saved_model/unet_optic/10 to ./saved_model/unet_optic/best_model, mIoU = 0.7055
    load test model: ./saved_model/unet_optic/10
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 10
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 10
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 10
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 10
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 10
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 10
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 10
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 10
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 10
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 10
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 10
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 10
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 10
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 10
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 10
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 10
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 10
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 10
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 10
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 10
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 10
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 10
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 10
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 10
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 10
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 10
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 10
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 10
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 10
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 10
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 10
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 10
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 10
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 10
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 10
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 10
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 10
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 10
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 10
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 10
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 10
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 10
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 10
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 10
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 10
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 10
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 10
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 10
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 10
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 10
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 10
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 10
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 10
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 10
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 10
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 10
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 10
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 10
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 10
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 10
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 10
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 10
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 10
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 10
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 10
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 10
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 10
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 10
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 10
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 10
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 10
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 10
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 10
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 10
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 10
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 10
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 10
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 10
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 10
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 10
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 10
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 10
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 10
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 10
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 10
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 10
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 10
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 10
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 10
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 10
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 10
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 10
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 10
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 10
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 10
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 10
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 10
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 10
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 10
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 10
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 10
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 10
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 10
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 10
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 10
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 10
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 10
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 10
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 10
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 10
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 10
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 10
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 10
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 10
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 10
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 10
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 10
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 10
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 10
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 10
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 10
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 10
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 10
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 10
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 10
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 10
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 10
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 10
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 10
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 10
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 10
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 10
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 10
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 10
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 10
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 10
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 10
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 10
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 10
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 10
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 10
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 10
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 10
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 10
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 10
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 10
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 10
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 10
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 10
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 10
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 10
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 10
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 10
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 10
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 10
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 10
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 10
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 10
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 10
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 10
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 10
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 10
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 10
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 10
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 10
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 10
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 10
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 10
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 10
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 10
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 10
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 10
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 10
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 10
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 10
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 10
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 10
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 10
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 10
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 10
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 10
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 10
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 10
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 10
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 10
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 10
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 10
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 10
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 10
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 10
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 10
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 10
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 10
    Save model checkpoint to ./saved_model/unet_optic/final
    

## 3 Horse2zebra
#### 3.1 模型准备



```python
#!wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip 
!unzip -oq /home/aistudio/data/horse2zebra.zip
```


```python
#!mkdir -p /home/aistudio/data/data10040/horse2zebra
!cp -r horse2zebra /home/aistudio/data/data10040/
```


```python
import os
image_path='/home/aistudio/data/data10040/horse2zebra/trainA'
file_names = os.listdir(image_path)
target_save_path='/home/aistudio/data/data10040/horse2zebra/'
fname=target_save_path+'trainA.txt'
fo=open(fname,'w+')
for line in file_names:
 fo.write('trainA/'+line+'\n')
#fo.write('\n'.join(file_names))
fo.close()
```


```python
import os
image_path='/home/aistudio/data/data10040/horse2zebra/trainB'
file_names = os.listdir(image_path)
target_save_path='/home/aistudio/data/data10040/horse2zebra/'
fname=target_save_path+'trainB.txt'
fo=open(fname,'w+')
for line in file_names:
 fo.write('trainB/'+line+'\n')
#fo.write('\n'.join(file_names))
fo.close()
```


```python
import paddle
from paddle.io import Dataset, DataLoader, IterableDataset
import numpy as np
import cv2
import random
import time
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore", category=Warning) # 过滤报警信息
BATCH_SIZE = 1
DATA_DIR = '/home/aistudio/data/data10040/horse2zebra/' # 设置训练集数据地址
#PLACE = paddle.CPUPlace() # 在cpu上训练
PLACE = paddle.CUDAPlace(0) # 在gpu上训练
```


```python
from PIL import Image
from paddle.vision.transforms import RandomCrop
# 处理图片数据：随机裁切、调整图片数据形状、归一化数据
def data_transform(img, output_size):
    h, w, _ = img.shape
    assert h == w and h >= output_size # check picture size
    # random crop
    rc = RandomCrop(224)
    img = rc(img)
    # normalize
    img = img / 255. * 2. - 1.
    # from [H,W,C] to [C,H,W]
    img = np.transpose(img, (2, 0, 1))
    # data type
    img = img.astype('float32') 
    return img
# 定义horse2zebra数据集对象
class H2ZDateset(Dataset):
    def __init__(self, data_dir):
        super(H2ZDateset, self).__init__()
        self.data_dir = data_dir
        self.pic_list_a = np.loadtxt(data_dir+'trainA.txt', dtype=np.str)
        np.random.shuffle(self.pic_list_a)
        self.pic_list_b = np.loadtxt(data_dir+'trainB.txt', dtype=np.str)
        np.random.shuffle(self.pic_list_b)
        self.pic_list_lenth = min(int(self.pic_list_a.shape[0]), int(self.pic_list_b.shape[0]))
    def __getitem__(self, idx):
        img_dir_a = self.data_dir+self.pic_list_a[idx]
        img_a = cv2.imread(img_dir_a)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_a = data_transform(img_a, 224)
        img_dir_b = self.data_dir+self.pic_list_b[idx]
        img_b = cv2.imread(img_dir_b)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        img_b = data_transform(img_b, 224)
        return np.array([img_a, img_b])
    def __len__(self):
        return self.pic_list_lenth
# 定义图片loader
h2zdateset = H2ZDateset(DATA_DIR)
loader = DataLoader(h2zdateset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False, num_workers=0, use_shared_memory=False)
data = next(loader())
data = np.transpose(data, (1, 0, 2, 3, 4))
print("读取的数据形状：", data.shape)
```

    W0226 18:48:52.900727   326 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0226 18:48:52.904693   326 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    WARNING: Detect dataset only contains single fileds, return format changed since Paddle 2.1. In Paddle <= 2.0, DataLoader add a list surround output data(e.g. return [data]), and in Paddle >= 2.1, DataLoader return the single filed directly (e.g. return data). For example, in following code: 
    
    import numpy as np
    from paddle.io import DataLoader, Dataset
    
    class RandomDataset(Dataset):
        def __getitem__(self, idx):
            data = np.random.random((2, 3)).astype('float32')
    
            return data
    
        def __len__(self):
            return 10
    
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=1)
    data = next(loader())
    
    In Paddle <= 2.0, data is in format '[Tensor(shape=(1, 2, 3), dtype=float32)]', and in Paddle >= 2.1, data is in format 'Tensor(shape=(1, 2, 3), dtype=float32)'
    
    

    读取的数据形状： (2, 1, 3, 224, 224)
    


```python
from PIL import Image
import os

# 打开图片
def open_pic(file_name='./data/data10040/horse2zebra/testA/n02381460_1300.jpg'):
    img = Image.open(file_name).resize((256, 256), Image.BILINEAR)
    img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    img = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
    return img

# 存储图片
def save_pics(pics, file_name='tmp', save_path='./output/pics/', save_root_path='./output/'):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(pics)):
        pics[i] = pics[i][0]
    pic = np.concatenate(tuple(pics), axis=2)
    pic = pic.transpose((1,2,0))
    pic = (pic + 1) / 2
    # plt.imshow(pic)
    pic = np.clip(pic * 256, 0, 255)
    img = Image.fromarray(pic.astype('uint8')).convert('RGB')
    img.save(save_path+file_name+'.jpg')
    
# 显示图片
def show_pics(pics):
    print(pics[0].shape)
    plt.figure(figsize=(3 * len(pics), 3), dpi=80)
    for i in range(len(pics)):
        pics[i] = (pics[i][0].transpose((1,2,0)) + 1) / 2
        plt.subplot(1, len(pics), i + 1)
        plt.imshow(pics[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 图片缓存队列
class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool = []
        self.count = 0
        self.pool_size = pool_size
    def pool_image(self, image):
        return image
        image = image.numpy()
        rtn = ''
        if self.count < self.pool_size:
            self.pool.append(image)
            self.count += 1
            rtn = image
        else:
            p = np.random.rand()
            if p > 0.5:
                random_id = np.random.randint(0, self.pool_size - 1)
                temp = self.pool[random_id]
                self.pool[random_id] = image
                rtn = temp
            else:
                rtn = image
        return paddle.to_tensor(rtn)
```


```python
show_pics([data[0], data[1]])
```

    (1, 3, 224, 224)
    


    
![png](main_files/main_33_1.png)
    



```python
import paddle
import paddle.nn as nn
import numpy as np

# 定义基础的“卷积层+实例归一化”块
class ConvIN(nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=1, bias_attr=None, 
        weight_attr=None):
        super(ConvIN, self).__init__()
        model = [
            nn.Conv2D(num_channels, num_filters, filter_size, stride=stride, padding=padding, 
                bias_attr=bias_attr, weight_attr=weight_attr),
            nn.InstanceNorm2D(num_filters),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# 定义CycleGAN的判别器
class Disc(nn.Layer):
    def __init__(self, weight_attr=nn.initializer.Normal(0., 0.02)):
        super(Disc, self).__init__()
        model = [
            ConvIN(3, 64, 4, stride=2, padding=1, bias_attr=True, weight_attr=weight_attr),
            ConvIN(64, 128, 4, stride=2, padding=1, bias_attr=False, weight_attr=weight_attr),
            ConvIN(128, 256, 4, stride=2, padding=1, bias_attr=False, weight_attr=weight_attr),
            ConvIN(256, 512, 4, stride=1, padding=1, bias_attr=False, weight_attr=weight_attr),
            nn.Conv2D(512, 1, 4, stride=1, padding=1, bias_attr=True, weight_attr=weight_attr)
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
```


```python
paddle.disable_static()
ci = ConvIN(3, 3, 3, weight_attr=nn.initializer.Normal(0., 0.02))
logit = ci(paddle.to_tensor(data[0]))
print('ConvIN块输出的特征图形状：', logit.shape)

d = Disc()
logit = d(paddle.to_tensor(data[0]))
print('判别器输出的特征图形状：', logit.shape)
```

    ConvIN块输出的特征图形状： [1, 3, 224, 224]
    判别器输出的特征图形状： [1, 1, 26, 26]
    


```python
# 定义基础的“转置卷积层+实例归一化”块
class ConvTransIN(nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding='same', padding_mode='constant', 
        bias_attr=None, weight_attr=None):
        super(ConvTransIN, self).__init__()
        model = [
            nn.Conv2DTranspose(num_channels, num_filters, filter_size, stride=stride, padding=padding, 
                bias_attr=bias_attr, weight_attr=weight_attr),
            nn.InstanceNorm2D(num_filters),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# 定义残差块
class Residual(nn.Layer):
    def __init__(self, dim, bias_attr=None, weight_attr=None):
        super(Residual, self).__init__()
        model = [
            nn.Conv2D(dim, dim, 3, stride=1, padding=1, padding_mode='reflect', bias_attr=bias_attr, 
                weight_attr=weight_attr),
            nn.InstanceNorm2D(dim),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return x + self.model(x)

# 定义CycleGAN的生成器
class Gen(nn.Layer):
    def __init__(self, base_dim=64, residual_num=7, downup_layer=2, weight_attr=nn.initializer.Normal(0., 0.02)):
        super(Gen, self).__init__()
        model=[
            nn.Conv2D(3, base_dim, 7, stride=1, padding=3, padding_mode='reflect', bias_attr=False, 
                weight_attr=weight_attr),
            nn.InstanceNorm2D(base_dim),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        # 下采样块（down sampling）
        for i in range(downup_layer):
            model += [
                ConvIN(base_dim * 2 ** i, base_dim * 2 ** (i + 1), 3, stride=2, padding=1, bias_attr=False, 
                    weight_attr=weight_attr),
            ]
        # 残差块（residual blocks）
        for i in range(residual_num):
            model += [
                Residual(base_dim * 2 ** downup_layer, True, weight_attr=nn.initializer.Normal(0., 0.02))
            ]
        # 上采样块（up sampling）
        for i in range(downup_layer):
            model += [
                ConvTransIN(base_dim * 2 ** (downup_layer - i), base_dim * 2 ** (downup_layer - i - 1), 3, 
                    stride=2, padding='same', padding_mode='constant', bias_attr=False, weight_attr=weight_attr),
            ]
        model += [
            nn.Conv2D(base_dim, 3, 7, stride=1, padding=3, padding_mode='reflect', bias_attr=True, 
                weight_attr=nn.initializer.Normal(0., 0.02)),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
```


```python
!pwd
```

    /home/aistudio
    


```python
cti = ConvTransIN(3, 3, 3, stride=2, padding='same', padding_mode='constant', bias_attr=False, 
    weight_attr=nn.initializer.Normal(0., 0.02))
logit = cti(paddle.to_tensor(data[0]))
print('ConvTransIN块输出的特征图形状：', logit.shape)

r = Residual(3, True, weight_attr=nn.initializer.Normal(0., 0.02))
logit = r(paddle.to_tensor(data[0]))
print('Residual块输出的特征图形状：', logit.shape)

g = Gen()
logit = g(paddle.to_tensor(data[0]))
print('生成器输出的特征图形状：', logit.shape)
```

    ConvTransIN块输出的特征图形状： [1, 3, 448, 448]
    Residual块输出的特征图形状： [1, 3, 224, 224]
    生成器输出的特征图形状： [1, 3, 224, 224]
    


```python

# 模型训练函数
def train(place, epoch_num=100, adv_weight=1, cycle_weight=10, identity_weight=10, \
          load_model=False, model_path='./model/', model_path_bkp='./model_bkp/', \
          print_interval=1, max_step=5, model_bkp_interval=2000):

    # 定义两对生成器、判别器对象
    g_a = Gen()
    g_b = Gen()
    d_a = Disc()
    d_b = Disc()

    # 定义数据读取器
    dataset = H2ZDateset(DATA_DIR)
    reader_ab = DataLoader(dataset, places=PLACE, shuffle=True, batch_size=BATCH_SIZE, drop_last=False, 
        num_workers=2)

    # 定义优化器
    g_a_optimizer = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=g_a.parameters())
    g_b_optimizer = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=g_b.parameters())
    d_a_optimizer = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=d_a.parameters())
    d_b_optimizer = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=d_b.parameters())

    # 定义图片缓存队列
    fa_pool, fb_pool = ImagePool(), ImagePool()

    # 定义总迭代次数为0
    total_step_num = np.array([0])

    # 加载存储的模型
    if load_model == True:
        ga_para_dict = paddle.load(model_path+'gen_b2a.pdparams')
        g_a.set_state_dict(ga_para_dict)

        gb_para_dict = paddle.load(model_path+'gen_a2b.pdparams')
        g_b.set_state_dict(gb_para_dict)

        da_para_dict = paddle.load(model_path+'dis_ga.pdparams')
        d_a.set_state_dict(da_para_dict)

        db_para_dict = paddle.load(model_path+'dis_gb.pdparams')
        d_b.set_state_dict(db_para_dict)

        total_step_num = np.load('./model/total_step_num.npy')
    
    # 定义本次训练开始时的迭代次数
    step = total_step_num[0]

    # 开始模型训练循环
    print('Start time :', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'start step:', step + 1)
    for epoch in range(epoch_num):
        for data_ab in reader_ab:
            step += 1

            # 设置模型为训练模式，针对bn、dropout等进行不同处理
            g_a.train()
            g_b.train()
            d_a.train()
            d_b.train()

            # 得到A、B组图片数据
            data_ab = np.transpose(data_ab, (1, 0, 2, 3, 4))
            img_ra = paddle.to_tensor(data_ab[0])
            img_rb = paddle.to_tensor(data_ab[1])

            # 训练判别器DA
            d_loss_ra = paddle.mean((d_a(img_ra.detach()) - 1) ** 2)
            d_loss_fa = paddle.mean(d_a(fa_pool.pool_image(g_a(img_rb.detach()))) ** 2)
            da_loss = (d_loss_ra + d_loss_fa) * 0.5
            d_a_optimizer.clear_grad() # 清除梯度
            da_loss.backward() # 反向更新梯度
            d_a_optimizer.step() # 更新模型权重

            # 训练判别器DB
            d_loss_rb = paddle.mean((d_b(img_rb.detach()) - 1) ** 2)
            d_loss_fb = paddle.mean(d_b(fb_pool.pool_image(g_b(img_ra.detach()))) ** 2)
            db_loss = (d_loss_rb + d_loss_fb) * 0.5
            d_b_optimizer.clear_grad()
            db_loss.backward()
            d_b_optimizer.step()

            # 训练生成器GA
            ga_gan_loss = paddle.mean((d_a(g_a(img_rb.detach())) - 1) ** 2)
            ga_cyc_loss = paddle.mean(paddle.abs(img_rb.detach() - g_b(g_a(img_rb.detach()))))
            ga_ide_loss = paddle.mean(paddle.abs(img_ra.detach() - g_a(img_ra.detach())))
            ga_loss = ga_gan_loss * adv_weight + ga_cyc_loss * cycle_weight + ga_ide_loss * identity_weight
            g_a_optimizer.clear_grad()
            ga_loss.backward()
            g_a_optimizer.step()

            # 训练生成器GB
            gb_gan_loss = paddle.mean((d_b(g_b(img_ra.detach())) - 1) ** 2)
            gb_cyc_loss = paddle.mean(paddle.abs(img_ra.detach() - g_a(g_b(img_ra.detach()))))
            gb_ide_loss = paddle.mean(paddle.abs(img_rb.detach() - g_b(img_rb.detach())))
            gb_loss = gb_gan_loss * adv_weight + gb_cyc_loss * cycle_weight + gb_ide_loss * identity_weight
            g_b_optimizer.clear_grad()
            gb_loss.backward()
            g_b_optimizer.step()
            
            # 存储训练过程中生成的图片
            if step in range(1, 101):
                pic_save_interval = 1
            elif step in range(101, 1001):
                pic_save_interval = 10
            elif step in range(1001, 10001):
                pic_save_interval = 100
            else:
                pic_save_interval = 500
            if step % pic_save_interval == 0:
                save_pics([img_ra.numpy(), g_b(img_ra).numpy(), g_a(g_b(img_ra)).numpy(), g_b(img_rb).numpy(), \
                            img_rb.numpy(), g_a(img_rb).numpy(), g_b(g_a(img_rb)).numpy(), g_a(img_ra).numpy()], \
                            str(step))
                test_pic = open_pic()
                test_pic_pp = paddle.to_tensor(test_pic)
                save_pics([test_pic, g_b(test_pic_pp).numpy()], str(step), save_path='./output/pics_test/')

            # 打印训练过程中的loss值和生成的图片
            if step % print_interval == 0:
                print([step], \
                        'DA:', da_loss.numpy(), d_loss_ra.numpy(), d_loss_fa.numpy(), \
                        'DB:', db_loss.numpy(), d_loss_rb.numpy(), d_loss_fb.numpy(), \
                        'GA:', ga_loss.numpy(), \
                        'GB:', gb_loss.numpy(), \
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                show_pics([img_ra.numpy(), g_b(img_ra).numpy(), g_a(g_b(img_ra)).numpy(), g_b(img_rb).numpy()])
                show_pics([img_rb.numpy(), g_a(img_rb).numpy(), g_b(g_a(img_rb)).numpy(), g_a(img_ra).numpy()])

            # 定期备份模型
            if step % model_bkp_interval == 0:
                paddle.save(g_a.state_dict(), model_path_bkp+'gen_b2a.pdparams')
                paddle.save(g_b.state_dict(), model_path_bkp+'gen_a2b.pdparams')
                paddle.save(d_a.state_dict(), model_path_bkp+'dis_ga.pdparams')
                paddle.save(d_b.state_dict(), model_path_bkp+'dis_gb.pdparams')
                np.save(model_path_bkp+'total_step_num', np.array([step]))

            # 完成训练时存储模型
            if step >= max_step + total_step_num[0]:
                paddle.save(g_a.state_dict(), model_path+'gen_b2a.pdparams')
                paddle.save(g_b.state_dict(), model_path+'gen_a2b.pdparams')
                paddle.save(d_a.state_dict(), model_path+'dis_ga.pdparams')
                paddle.save(d_b.state_dict(), model_path+'dis_gb.pdparams')
                np.save(model_path+'total_step_num', np.array([step]))
                print('End time :', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'End Step:', step)
                return

# 从头训练
# train(PLACE)

# 继续训练
train(PLACE, print_interval=1, max_step=5, load_model=True)
# train(PLACE, print_interval=500, max_step=200000, load_model=True)
```

    Start time : 2022-02-26 19:03:10 start step: 28031
    [28031] DA: [0.15572916] [0.15879211] [0.15266621] DB: [0.10216073] [0.03328097] [0.17104049] GA: [2.99264] GB: [2.603099] 2022-02-26 19:03:10
    (1, 3, 224, 224)
    


    
![png](main_files/main_39_1.png)
    


    (1, 3, 224, 224)
    


    
![png](main_files/main_39_3.png)
    


    [28032] DA: [0.22164321] [0.19200392] [0.2512825] DB: [0.03340015] [0.02800788] [0.03879242] GA: [2.3247375] GB: [2.4660404] 2022-02-26 19:03:11
    (1, 3, 224, 224)
    


    
![png](main_files/main_39_5.png)
    


    (1, 3, 224, 224)
    


    
![png](main_files/main_39_7.png)
    


    [28033] DA: [0.1730434] [0.08551465] [0.26057217] DB: [0.392854] [0.768643] [0.017065] GA: [2.6929455] GB: [2.9628701] 2022-02-26 19:03:12
    (1, 3, 224, 224)
    


    
![png](main_files/main_39_9.png)
    


    (1, 3, 224, 224)
    


    
![png](main_files/main_39_11.png)
    


    [28034] DA: [0.03355002] [0.01687377] [0.05022626] DB: [0.06274176] [0.10998107] [0.01550244] GA: [3.4384751] GB: [2.8524084] 2022-02-26 19:03:13
    (1, 3, 224, 224)
    


    
![png](main_files/main_39_13.png)
    


    (1, 3, 224, 224)
    


    
![png](main_files/main_39_15.png)
    


    [28035] DA: [0.18592301] [0.29941323] [0.0724328] DB: [0.05006506] [0.01937424] [0.08075589] GA: [3.6906383] GB: [2.39774] 2022-02-26 19:03:13
    (1, 3, 224, 224)
    


    
![png](main_files/main_39_17.png)
    


    (1, 3, 224, 224)
    


    
![png](main_files/main_39_19.png)
    


    End time : 2022-02-26 19:03:14 End Step: 28035
    


```python
def infer(img_path, place, model_path='./model/'):
    # 定义生成器对象
    g_b = Gen()

    # 设置模型为训练模式，针对bn、dropout等进行不同处理
    g_b.eval()

    # 读取存储的模型
    gb_para_dict = paddle.load(model_path+'gen_a2b.pdparams')
    g_b.set_state_dict(gb_para_dict)
    
    # 读取图片数据
    img_a = cv2.imread(img_path)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_a = data_transform(img_a, 224)
    img_a = paddle.to_tensor(np.array([img_a]))
    
    # 正向计算进行推理
    img_b = g_b(img_a)

    # 打印输出输入、输出图片
    print(img_a.numpy().shape, img_a.numpy().dtype)
    show_pics([img_a.numpy(), img_b.numpy()])

infer('./data/data10040/horse2zebra/testA/n02381460_1300.jpg', PLACE)
```

    (1, 3, 224, 224) float32
    (1, 3, 224, 224)
    


    
![png](main_files/main_40_1.png)
    



```python

```

## 五、模型评估
鉴于vegetables和horse2zebra模型已经在前面的作业完成了评估工作，此处以眼底血管数据集为例，展示模型评估过程



```python
!python PaddleSeg/pdseg/train.py --cfg PaddleSeg/configs/unet_optic.yaml \
                      --use_gpu \
                      --do_eval \
                      --use_vdl \
                      --vdl_log_dir train_log \
                      BATCH_SIZE 4 \
                      SOLVER.LR 0.001
```

    {'AUG': {'AUG_METHOD': 'unpadding',
             'FIX_RESIZE_SIZE': (565, 584),
             'FLIP': False,
             'FLIP_RATIO': 0.5,
             'INF_RESIZE_VALUE': 500,
             'MAX_RESIZE_VALUE': 600,
             'MAX_SCALE_FACTOR': 2.0,
             'MIN_RESIZE_VALUE': 400,
             'MIN_SCALE_FACTOR': 0.5,
             'MIRROR': True,
             'RICH_CROP': {'ASPECT_RATIO': 0.33,
                           'BLUR': False,
                           'BLUR_RATIO': 0.1,
                           'BRIGHTNESS_JITTER_RATIO': 0.5,
                           'CONTRAST_JITTER_RATIO': 0.5,
                           'ENABLE': False,
                           'MAX_ROTATION': 15,
                           'MIN_AREA_RATIO': 0.5,
                           'SATURATION_JITTER_RATIO': 0.5},
             'SCALE_STEP_SIZE': 0.25,
             'TO_RGB': False},
     'BATCH_SIZE': 4,
     'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
     'DATASET': {'DATA_DIM': 3,
                 'DATA_DIR': '',
                 'IGNORE_INDEX': 255,
                 'IMAGE_TYPE': 'rgb',
                 'NUM_CLASSES': 2,
                 'PADDING_VALUE': [127.5, 127.5, 127.5],
                 'SEPARATOR': ' ',
                 'TEST_FILE_LIST': 'train_list.txt',
                 'TEST_TOTAL_IMAGES': 193,
                 'TRAIN_FILE_LIST': 'train_list.txt',
                 'TRAIN_TOTAL_IMAGES': 193,
                 'VAL_FILE_LIST': 'val_list.txt',
                 'VAL_TOTAL_IMAGES': 7,
                 'VIS_FILE_LIST': 'train_list.txt'},
     'EVAL_CROP_SIZE': (565, 584),
     'FREEZE': {'MODEL_FILENAME': '__model__',
                'PARAMS_FILENAME': '__params__',
                'SAVE_DIR': 'freeze_model'},
     'MEAN': [0.5, 0.5, 0.5],
     'MODEL': {'BN_MOMENTUM': 0.99,
               'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                           'BACKBONE': 'xception_65',
                           'BACKBONE_LR_MULT_LIST': None,
                           'DECODER': {'CONV_FILTERS': 256,
                                       'OUTPUT_IS_LOGITS': False,
                                       'USE_SUM_MERGE': False},
                           'DECODER_USE_SEP_CONV': True,
                           'DEPTH_MULTIPLIER': 1.0,
                           'ENABLE_DECODER': True,
                           'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                       'ASPP_CONVS_FILTERS': 256,
                                       'ASPP_RATIOS': None,
                                       'ASPP_WITH_CONCAT_PROJECTION': True,
                                       'ASPP_WITH_SE': False,
                                       'POOLING_CROP_SIZE': None,
                                       'POOLING_STRIDE': [1, 1],
                                       'SE_USE_QSIGMOID': False},
                           'ENCODER_WITH_ASPP': True,
                           'OUTPUT_STRIDE': 16},
               'DEFAULT_EPSILON': 1e-05,
               'DEFAULT_GROUP_NUMBER': 32,
               'DEFAULT_NORM_TYPE': 'bn',
               'FP16': False,
               'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                         'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                    'NUM_MODULES': 4},
                         'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                    'NUM_MODULES': 3}},
               'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
               'MODEL_NAME': 'unet',
               'MULTI_LOSS_WEIGHT': [1.0],
               'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
               'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
               'SCALE_LOSS': 'DYNAMIC',
               'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
     'NUM_TRAINERS': 1,
     'SLIM': {'KNOWLEDGE_DISTILL': False,
              'KNOWLEDGE_DISTILL_IS_TEACHER': False,
              'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
              'NAS_ADDRESS': '',
              'NAS_IS_SERVER': True,
              'NAS_PORT': 23333,
              'NAS_SEARCH_STEPS': 100,
              'NAS_SPACE_NAME': '',
              'NAS_START_EVAL_EPOCH': 0,
              'PREPROCESS': False,
              'PRUNE_PARAMS': '',
              'PRUNE_RATIOS': []},
     'SOLVER': {'BEGIN_EPOCH': 1,
                'CROSS_ENTROPY_WEIGHT': None,
                'DECAY_EPOCH': [10, 20],
                'GAMMA': 0.1,
                'LOSS': ['softmax_loss'],
                'LOSS_WEIGHT': {'BCE_LOSS': 1,
                                'DICE_LOSS': 1,
                                'LOVASZ_HINGE_LOSS': 1,
                                'LOVASZ_SOFTMAX_LOSS': 1,
                                'SOFTMAX_LOSS': 1},
                'LR': 0.001,
                'LR_POLICY': 'poly',
                'LR_WARMUP': False,
                'LR_WARMUP_STEPS': 2000,
                'MOMENTUM': 0.9,
                'MOMENTUM2': 0.999,
                'NUM_EPOCHS': 10,
                'OPTIMIZER': 'adam',
                'POWER': 0.9,
                'WEIGHT_DECAY': 4e-05},
     'STD': [0.5, 0.5, 0.5],
     'TEST': {'TEST_MODEL': './saved_model/unet_optic/final'},
     'TRAIN': {'MODEL_SAVE_DIR': './saved_model/unet_optic/',
               'PRETRAINED_MODEL_DIR': '',
               'RESUME_MODEL_DIR': '',
               'SNAPSHOT_EPOCH': 2,
               'SYNC_BATCH_NORM': False},
     'TRAINER_ID': 0,
     'TRAIN_CROP_SIZE': (565, 584)}
    #Device count: 1
    batch_size_per_dev: 4
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:341: UserWarning: /home/aistudio/PaddleSeg/pdseg/loss.py:79
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    W0226 21:13:14.769402   895 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0226 21:13:14.774269   895 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    Pretrained model dir  not exists, training from scratch...
    Use multi-thread reader
    epoch=1 step=10 lr=0.00098 loss=0.4524 step/sec=2.839 | ETA 00:02:45
    epoch=1 step=20 lr=0.00097 loss=0.2706 step/sec=3.182 | ETA 00:02:24
    epoch=1 step=30 lr=0.00095 loss=0.2244 step/sec=3.181 | ETA 00:02:21
    epoch=1 step=40 lr=0.00093 loss=0.2253 step/sec=3.174 | ETA 00:02:18
    epoch=2 step=50 lr=0.00091 loss=0.1978 step/sec=3.078 | ETA 00:02:19
    epoch=2 step=60 lr=0.00089 loss=0.1929 step/sec=3.169 | ETA 00:02:12
    epoch=2 step=70 lr=0.00087 loss=0.1752 step/sec=3.160 | ETA 00:02:09
    epoch=2 step=80 lr=0.00085 loss=0.1873 step/sec=3.157 | ETA 00:02:06
    epoch=2 step=90 lr=0.00084 loss=0.1901 step/sec=3.156 | ETA 00:02:03
    Save model checkpoint to ./saved_model/unet_optic/2
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/2
    [EVAL]step=1 loss=0.45712 acc=0.8930 IoU=0.5474 step/sec=3.88 | ETA 00:00:12
    [EVAL]step=2 loss=0.43932 acc=0.8902 IoU=0.5288 step/sec=7.85 | ETA 00:00:05
    [EVAL]#image=7 acc=0.8902 IoU=0.5288
    [EVAL]Category IoU: [0.8877 0.1699]
    [EVAL]Category Acc: [0.8919 0.8288]
    [EVAL]Kappa:0.2573
    Save best model ./saved_model/unet_optic/2 to ./saved_model/unet_optic/best_model, mIoU = 0.5288
    load test model: ./saved_model/unet_optic/2
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 2
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 2
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 2
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 2
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 2
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 2
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 2
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 2
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 2
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 2
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 2
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 2
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 2
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 2
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 2
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 2
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 2
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 2
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 2
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 2
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 2
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 2
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 2
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 2
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 2
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 2
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 2
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 2
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 2
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 2
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 2
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 2
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 2
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 2
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 2
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 2
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 2
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 2
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 2
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 2
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 2
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 2
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 2
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 2
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 2
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 2
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 2
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 2
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 2
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 2
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 2
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 2
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 2
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 2
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 2
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 2
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 2
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 2
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 2
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 2
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 2
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 2
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 2
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 2
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 2
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 2
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 2
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 2
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 2
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 2
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 2
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 2
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 2
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 2
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 2
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 2
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 2
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 2
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 2
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 2
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 2
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 2
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 2
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 2
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 2
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 2
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 2
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 2
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 2
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 2
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 2
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 2
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 2
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 2
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 2
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 2
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 2
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 2
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 2
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 2
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 2
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 2
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 2
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 2
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 2
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 2
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 2
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 2
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 2
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 2
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 2
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 2
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 2
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 2
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 2
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 2
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 2
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 2
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 2
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 2
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 2
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 2
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 2
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 2
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 2
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 2
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 2
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 2
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 2
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 2
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 2
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 2
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 2
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 2
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 2
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 2
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 2
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 2
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 2
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 2
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 2
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 2
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 2
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 2
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 2
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 2
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 2
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 2
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 2
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 2
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 2
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 2
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 2
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 2
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 2
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 2
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 2
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 2
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 2
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 2
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 2
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 2
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 2
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 2
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 2
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 2
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 2
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 2
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 2
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 2
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 2
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 2
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 2
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 2
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 2
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 2
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 2
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 2
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 2
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 2
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 2
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 2
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 2
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 2
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 2
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 2
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 2
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 2
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 2
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 2
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 2
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 2
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 2
    epoch=3 step=100 lr=0.00082 loss=0.1791 step/sec=0.426 | ETA 00:14:52
    epoch=3 step=110 lr=0.00080 loss=0.1780 step/sec=3.157 | ETA 00:01:57
    epoch=3 step=120 lr=0.00078 loss=0.1805 step/sec=3.152 | ETA 00:01:54
    epoch=3 step=130 lr=0.00076 loss=0.1692 step/sec=3.149 | ETA 00:01:51
    epoch=3 step=140 lr=0.00074 loss=0.1526 step/sec=3.148 | ETA 00:01:48
    epoch=4 step=150 lr=0.00072 loss=0.1575 step/sec=3.040 | ETA 00:01:48
    epoch=4 step=160 lr=0.00070 loss=0.1507 step/sec=3.142 | ETA 00:01:41
    epoch=4 step=170 lr=0.00068 loss=0.1591 step/sec=3.133 | ETA 00:01:38
    epoch=4 step=180 lr=0.00066 loss=0.1526 step/sec=3.137 | ETA 00:01:35
    epoch=4 step=190 lr=0.00065 loss=0.1581 step/sec=3.126 | ETA 00:01:32
    Save model checkpoint to ./saved_model/unet_optic/4
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/4
    [EVAL]step=1 loss=0.24719 acc=0.9022 IoU=0.6015 step/sec=3.73 | ETA 00:00:12
    [EVAL]step=2 loss=0.22161 acc=0.9072 IoU=0.6178 step/sec=7.79 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9072 IoU=0.6178
    [EVAL]Category IoU: [0.9027 0.3329]
    [EVAL]Category Acc: [0.9137 0.8014]
    [EVAL]Kappa:0.4562
    Save best model ./saved_model/unet_optic/4 to ./saved_model/unet_optic/best_model, mIoU = 0.6178
    load test model: ./saved_model/unet_optic/4
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 4
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 4
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 4
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 4
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 4
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 4
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 4
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 4
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 4
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 4
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 4
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 4
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 4
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 4
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 4
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 4
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 4
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 4
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 4
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 4
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 4
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 4
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 4
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 4
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 4
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 4
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 4
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 4
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 4
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 4
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 4
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 4
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 4
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 4
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 4
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 4
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 4
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 4
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 4
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 4
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 4
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 4
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 4
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 4
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 4
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 4
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 4
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 4
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 4
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 4
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 4
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 4
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 4
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 4
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 4
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 4
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 4
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 4
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 4
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 4
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 4
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 4
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 4
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 4
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 4
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 4
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 4
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 4
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 4
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 4
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 4
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 4
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 4
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 4
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 4
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 4
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 4
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 4
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 4
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 4
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 4
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 4
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 4
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 4
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 4
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 4
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 4
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 4
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 4
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 4
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 4
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 4
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 4
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 4
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 4
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 4
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 4
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 4
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 4
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 4
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 4
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 4
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 4
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 4
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 4
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 4
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 4
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 4
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 4
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 4
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 4
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 4
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 4
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 4
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 4
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 4
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 4
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 4
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 4
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 4
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 4
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 4
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 4
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 4
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 4
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 4
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 4
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 4
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 4
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 4
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 4
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 4
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 4
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 4
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 4
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 4
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 4
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 4
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 4
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 4
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 4
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 4
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 4
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 4
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 4
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 4
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 4
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 4
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 4
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 4
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 4
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 4
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 4
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 4
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 4
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 4
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 4
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 4
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 4
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 4
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 4
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 4
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 4
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 4
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 4
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 4
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 4
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 4
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 4
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 4
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 4
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 4
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 4
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 4
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 4
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 4
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 4
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 4
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 4
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 4
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 4
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 4
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 4
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 4
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 4
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 4
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 4
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 4
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 4
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 4
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 4
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 4
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 4
    epoch=5 step=200 lr=0.00063 loss=0.1446 step/sec=0.415 | ETA 00:11:15
    epoch=5 step=210 lr=0.00061 loss=0.1456 step/sec=3.140 | ETA 00:01:25
    epoch=5 step=220 lr=0.00059 loss=0.1589 step/sec=3.137 | ETA 00:01:22
    epoch=5 step=230 lr=0.00057 loss=0.1501 step/sec=3.129 | ETA 00:01:19
    epoch=5 step=240 lr=0.00055 loss=0.1523 step/sec=3.127 | ETA 00:01:16
    epoch=6 step=250 lr=0.00053 loss=0.1554 step/sec=3.028 | ETA 00:01:15
    epoch=6 step=260 lr=0.00051 loss=0.1415 step/sec=3.118 | ETA 00:01:10
    epoch=6 step=270 lr=0.00049 loss=0.1386 step/sec=3.119 | ETA 00:01:07
    epoch=6 step=280 lr=0.00047 loss=0.1369 step/sec=3.122 | ETA 00:01:04
    Save model checkpoint to ./saved_model/unet_optic/6
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/6
    [EVAL]step=1 loss=0.23677 acc=0.9057 IoU=0.6014 step/sec=3.39 | ETA 00:00:14
    [EVAL]step=2 loss=0.22824 acc=0.9078 IoU=0.6091 step/sec=7.00 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9078 IoU=0.6091
    [EVAL]Category IoU: [0.9037 0.3145]
    [EVAL]Category Acc: [0.9103 0.8596]
    [EVAL]Kappa:0.4386
    load test model: ./saved_model/unet_optic/6
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 6
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 6
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 6
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 6
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 6
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 6
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 6
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 6
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 6
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 6
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 6
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 6
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 6
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 6
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 6
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 6
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 6
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 6
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 6
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 6
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 6
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 6
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 6
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 6
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 6
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 6
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 6
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 6
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 6
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 6
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 6
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 6
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 6
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 6
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 6
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 6
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 6
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 6
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 6
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 6
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 6
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 6
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 6
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 6
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 6
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 6
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 6
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 6
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 6
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 6
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 6
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 6
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 6
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 6
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 6
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 6
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 6
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 6
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 6
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 6
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 6
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 6
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 6
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 6
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 6
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 6
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 6
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 6
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 6
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 6
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 6
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 6
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 6
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 6
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 6
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 6
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 6
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 6
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 6
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 6
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 6
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 6
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 6
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 6
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 6
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 6
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 6
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 6
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 6
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 6
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 6
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 6
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 6
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 6
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 6
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 6
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 6
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 6
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 6
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 6
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 6
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 6
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 6
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 6
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 6
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 6
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 6
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 6
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 6
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 6
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 6
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 6
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 6
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 6
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 6
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 6
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 6
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 6
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 6
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 6
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 6
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 6
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 6
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 6
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 6
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 6
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 6
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 6
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 6
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 6
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 6
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 6
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 6
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 6
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 6
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 6
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 6
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 6
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 6
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 6
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 6
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 6
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 6
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 6
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 6
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 6
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 6
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 6
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 6
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 6
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 6
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 6
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 6
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 6
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 6
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 6
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 6
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 6
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 6
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 6
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 6
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 6
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 6
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 6
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 6
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 6
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 6
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 6
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 6
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 6
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 6
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 6
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 6
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 6
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 6
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 6
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 6
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 6
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 6
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 6
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 6
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 6
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 6
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 6
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 6
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 6
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 6
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 6
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 6
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 6
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 6
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 6
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 6
    epoch=7 step=290 lr=0.00045 loss=0.1552 step/sec=0.420 | ETA 00:07:32
    epoch=7 step=300 lr=0.00043 loss=0.1433 step/sec=3.143 | ETA 00:00:57
    epoch=7 step=310 lr=0.00041 loss=0.1410 step/sec=3.128 | ETA 00:00:54
    epoch=7 step=320 lr=0.00039 loss=0.1359 step/sec=3.134 | ETA 00:00:51
    epoch=7 step=330 lr=0.00037 loss=0.1395 step/sec=3.124 | ETA 00:00:48
    epoch=8 step=340 lr=0.00035 loss=0.1356 step/sec=3.031 | ETA 00:00:46
    epoch=8 step=350 lr=0.00033 loss=0.1425 step/sec=3.127 | ETA 00:00:41
    epoch=8 step=360 lr=0.00031 loss=0.1335 step/sec=3.120 | ETA 00:00:38
    epoch=8 step=370 lr=0.00029 loss=0.1367 step/sec=3.122 | ETA 00:00:35
    epoch=8 step=380 lr=0.00026 loss=0.1408 step/sec=3.121 | ETA 00:00:32
    Save model checkpoint to ./saved_model/unet_optic/8
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/8
    [EVAL]step=1 loss=0.21147 acc=0.9189 IoU=0.6510 step/sec=3.72 | ETA 00:00:12
    [EVAL]step=2 loss=0.19398 acc=0.9222 IoU=0.6645 step/sec=7.58 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9222 IoU=0.6645
    [EVAL]Category IoU: [0.9177 0.4112]
    [EVAL]Category Acc: [0.9221 0.9230]
    [EVAL]Kappa:0.5462
    Save best model ./saved_model/unet_optic/8 to ./saved_model/unet_optic/best_model, mIoU = 0.6645
    load test model: ./saved_model/unet_optic/8
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 8
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 8
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 8
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 8
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 8
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 8
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 8
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 8
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 8
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 8
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 8
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 8
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 8
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 8
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 8
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 8
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 8
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 8
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 8
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 8
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 8
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 8
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 8
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 8
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 8
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 8
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 8
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 8
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 8
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 8
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 8
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 8
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 8
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 8
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 8
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 8
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 8
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 8
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 8
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 8
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 8
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 8
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 8
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 8
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 8
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 8
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 8
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 8
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 8
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 8
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 8
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 8
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 8
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 8
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 8
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 8
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 8
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 8
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 8
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 8
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 8
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 8
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 8
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 8
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 8
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 8
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 8
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 8
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 8
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 8
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 8
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 8
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 8
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 8
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 8
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 8
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 8
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 8
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 8
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 8
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 8
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 8
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 8
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 8
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 8
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 8
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 8
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 8
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 8
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 8
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 8
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 8
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 8
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 8
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 8
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 8
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 8
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 8
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 8
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 8
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 8
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 8
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 8
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 8
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 8
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 8
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 8
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 8
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 8
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 8
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 8
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 8
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 8
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 8
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 8
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 8
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 8
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 8
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 8
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 8
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 8
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 8
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 8
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 8
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 8
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 8
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 8
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 8
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 8
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 8
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 8
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 8
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 8
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 8
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 8
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 8
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 8
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 8
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 8
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 8
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 8
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 8
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 8
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 8
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 8
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 8
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 8
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 8
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 8
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 8
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 8
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 8
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 8
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 8
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 8
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 8
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 8
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 8
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 8
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 8
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 8
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 8
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 8
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 8
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 8
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 8
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 8
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 8
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 8
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 8
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 8
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 8
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 8
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 8
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 8
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 8
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 8
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 8
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 8
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 8
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 8
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 8
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 8
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 8
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 8
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 8
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 8
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 8
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 8
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 8
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 8
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 8
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 8
    epoch=9 step=390 lr=0.00024 loss=0.1289 step/sec=0.413 | ETA 00:03:37
    epoch=9 step=400 lr=0.00022 loss=0.1270 step/sec=3.129 | ETA 00:00:25
    epoch=9 step=410 lr=0.00020 loss=0.1377 step/sec=3.136 | ETA 00:00:22
    epoch=9 step=420 lr=0.00018 loss=0.1367 step/sec=3.129 | ETA 00:00:19
    epoch=9 step=430 lr=0.00016 loss=0.1386 step/sec=3.124 | ETA 00:00:16
    epoch=10 step=440 lr=0.00013 loss=0.1313 step/sec=3.036 | ETA 00:00:13
    epoch=10 step=450 lr=0.00011 loss=0.1272 step/sec=3.124 | ETA 00:00:09
    epoch=10 step=460 lr=0.00009 loss=0.1356 step/sec=3.121 | ETA 00:00:06
    epoch=10 step=470 lr=0.00006 loss=0.1392 step/sec=3.120 | ETA 00:00:03
    epoch=10 step=480 lr=0.00004 loss=0.1265 step/sec=3.118 | ETA 00:00:00
    Save model checkpoint to ./saved_model/unet_optic/10
    Evaluation start
    #Device count: 1
    load test model: ./saved_model/unet_optic/10
    [EVAL]step=1 loss=0.19019 acc=0.9270 IoU=0.6864 step/sec=3.89 | ETA 00:00:12
    [EVAL]step=2 loss=0.17055 acc=0.9310 IoU=0.7025 step/sec=7.17 | ETA 00:00:06
    [EVAL]#image=7 acc=0.9310 IoU=0.7025
    [EVAL]Category IoU: [0.9263 0.4788]
    [EVAL]Category Acc: [0.9311 0.9294]
    [EVAL]Kappa:0.6131
    Save best model ./saved_model/unet_optic/10 to ./saved_model/unet_optic/best_model, mIoU = 0.7025
    load test model: ./saved_model/unet_optic/10
    #1 visualize image path: visual/93.png
    VisualDL visualization epoch 10
    #2 visualize image path: visual/50.png
    VisualDL visualization epoch 10
    #3 visualize image path: visual/60.png
    VisualDL visualization epoch 10
    #4 visualize image path: visual/135.png
    VisualDL visualization epoch 10
    #5 visualize image path: visual/178.png
    VisualDL visualization epoch 10
    #6 visualize image path: visual/22.png
    VisualDL visualization epoch 10
    #7 visualize image path: visual/75.png
    VisualDL visualization epoch 10
    #8 visualize image path: visual/68.png
    VisualDL visualization epoch 10
    #9 visualize image path: visual/144.png
    VisualDL visualization epoch 10
    #10 visualize image path: visual/185.png
    VisualDL visualization epoch 10
    #11 visualize image path: visual/63.png
    VisualDL visualization epoch 10
    #12 visualize image path: visual/150.png
    VisualDL visualization epoch 10
    #13 visualize image path: visual/100.png
    VisualDL visualization epoch 10
    #14 visualize image path: visual/171.png
    VisualDL visualization epoch 10
    #15 visualize image path: visual/43.png
    VisualDL visualization epoch 10
    #16 visualize image path: visual/131.png
    VisualDL visualization epoch 10
    #17 visualize image path: visual/123.png
    VisualDL visualization epoch 10
    #18 visualize image path: visual/192.png
    VisualDL visualization epoch 10
    #19 visualize image path: visual/70.png
    VisualDL visualization epoch 10
    #20 visualize image path: visual/12.png
    VisualDL visualization epoch 10
    #21 visualize image path: visual/161.png
    VisualDL visualization epoch 10
    #22 visualize image path: visual/77.png
    VisualDL visualization epoch 10
    #23 visualize image path: visual/23.png
    VisualDL visualization epoch 10
    #24 visualize image path: visual/101.png
    VisualDL visualization epoch 10
    #25 visualize image path: visual/159.png
    VisualDL visualization epoch 10
    #26 visualize image path: visual/108.png
    VisualDL visualization epoch 10
    #27 visualize image path: visual/85.png
    VisualDL visualization epoch 10
    #28 visualize image path: visual/7.png
    VisualDL visualization epoch 10
    #29 visualize image path: visual/139.png
    VisualDL visualization epoch 10
    #30 visualize image path: visual/84.png
    VisualDL visualization epoch 10
    #31 visualize image path: visual/95.png
    VisualDL visualization epoch 10
    #32 visualize image path: visual/187.png
    VisualDL visualization epoch 10
    #33 visualize image path: visual/128.png
    VisualDL visualization epoch 10
    #34 visualize image path: visual/37.png
    VisualDL visualization epoch 10
    #35 visualize image path: visual/2.png
    VisualDL visualization epoch 10
    #36 visualize image path: visual/152.png
    VisualDL visualization epoch 10
    #37 visualize image path: visual/80.png
    VisualDL visualization epoch 10
    #38 visualize image path: visual/173.png
    VisualDL visualization epoch 10
    #39 visualize image path: visual/168.png
    VisualDL visualization epoch 10
    #40 visualize image path: visual/129.png
    VisualDL visualization epoch 10
    #41 visualize image path: visual/109.png
    VisualDL visualization epoch 10
    #42 visualize image path: visual/62.png
    VisualDL visualization epoch 10
    #43 visualize image path: visual/46.png
    VisualDL visualization epoch 10
    #44 visualize image path: visual/58.png
    VisualDL visualization epoch 10
    #45 visualize image path: visual/165.png
    VisualDL visualization epoch 10
    #46 visualize image path: visual/179.png
    VisualDL visualization epoch 10
    #47 visualize image path: visual/117.png
    VisualDL visualization epoch 10
    #48 visualize image path: visual/113.png
    VisualDL visualization epoch 10
    #49 visualize image path: visual/54.png
    VisualDL visualization epoch 10
    #50 visualize image path: visual/158.png
    VisualDL visualization epoch 10
    #51 visualize image path: visual/130.png
    VisualDL visualization epoch 10
    #52 visualize image path: visual/76.png
    VisualDL visualization epoch 10
    #53 visualize image path: visual/176.png
    VisualDL visualization epoch 10
    #54 visualize image path: visual/132.png
    VisualDL visualization epoch 10
    #55 visualize image path: visual/160.png
    VisualDL visualization epoch 10
    #56 visualize image path: visual/141.png
    VisualDL visualization epoch 10
    #57 visualize image path: visual/29.png
    VisualDL visualization epoch 10
    #58 visualize image path: visual/191.png
    VisualDL visualization epoch 10
    #59 visualize image path: visual/69.png
    VisualDL visualization epoch 10
    #60 visualize image path: visual/89.png
    VisualDL visualization epoch 10
    #61 visualize image path: visual/44.png
    VisualDL visualization epoch 10
    #62 visualize image path: visual/184.png
    VisualDL visualization epoch 10
    #63 visualize image path: visual/162.png
    VisualDL visualization epoch 10
    #64 visualize image path: visual/120.png
    VisualDL visualization epoch 10
    #65 visualize image path: visual/188.png
    VisualDL visualization epoch 10
    #66 visualize image path: visual/33.png
    VisualDL visualization epoch 10
    #67 visualize image path: visual/186.png
    VisualDL visualization epoch 10
    #68 visualize image path: visual/183.png
    VisualDL visualization epoch 10
    #69 visualize image path: visual/19.png
    VisualDL visualization epoch 10
    #70 visualize image path: visual/174.png
    VisualDL visualization epoch 10
    #71 visualize image path: visual/86.png
    VisualDL visualization epoch 10
    #72 visualize image path: visual/61.png
    VisualDL visualization epoch 10
    #73 visualize image path: visual/11.png
    VisualDL visualization epoch 10
    #74 visualize image path: visual/145.png
    VisualDL visualization epoch 10
    #75 visualize image path: visual/24.png
    VisualDL visualization epoch 10
    #76 visualize image path: visual/82.png
    VisualDL visualization epoch 10
    #77 visualize image path: visual/122.png
    VisualDL visualization epoch 10
    #78 visualize image path: visual/65.png
    VisualDL visualization epoch 10
    #79 visualize image path: visual/94.png
    VisualDL visualization epoch 10
    #80 visualize image path: visual/57.png
    VisualDL visualization epoch 10
    #81 visualize image path: visual/134.png
    VisualDL visualization epoch 10
    #82 visualize image path: visual/142.png
    VisualDL visualization epoch 10
    #83 visualize image path: visual/4.png
    VisualDL visualization epoch 10
    #84 visualize image path: visual/137.png
    VisualDL visualization epoch 10
    #85 visualize image path: visual/143.png
    VisualDL visualization epoch 10
    #86 visualize image path: visual/72.png
    VisualDL visualization epoch 10
    #87 visualize image path: visual/41.png
    VisualDL visualization epoch 10
    #88 visualize image path: visual/52.png
    VisualDL visualization epoch 10
    #89 visualize image path: visual/0.png
    VisualDL visualization epoch 10
    #90 visualize image path: visual/35.png
    VisualDL visualization epoch 10
    #91 visualize image path: visual/83.png
    VisualDL visualization epoch 10
    #92 visualize image path: visual/13.png
    VisualDL visualization epoch 10
    #93 visualize image path: visual/175.png
    VisualDL visualization epoch 10
    #94 visualize image path: visual/81.png
    VisualDL visualization epoch 10
    #95 visualize image path: visual/197.png
    VisualDL visualization epoch 10
    #96 visualize image path: visual/166.png
    VisualDL visualization epoch 10
    #97 visualize image path: visual/118.png
    VisualDL visualization epoch 10
    #98 visualize image path: visual/47.png
    VisualDL visualization epoch 10
    #99 visualize image path: visual/172.png
    VisualDL visualization epoch 10
    #100 visualize image path: visual/115.png
    VisualDL visualization epoch 10
    #101 visualize image path: visual/71.png
    VisualDL visualization epoch 10
    #102 visualize image path: visual/74.png
    VisualDL visualization epoch 10
    #103 visualize image path: visual/163.png
    VisualDL visualization epoch 10
    #104 visualize image path: visual/79.png
    VisualDL visualization epoch 10
    #105 visualize image path: visual/177.png
    VisualDL visualization epoch 10
    #106 visualize image path: visual/146.png
    VisualDL visualization epoch 10
    #107 visualize image path: visual/170.png
    VisualDL visualization epoch 10
    #108 visualize image path: visual/157.png
    VisualDL visualization epoch 10
    #109 visualize image path: visual/99.png
    VisualDL visualization epoch 10
    #110 visualize image path: visual/148.png
    VisualDL visualization epoch 10
    #111 visualize image path: visual/196.png
    VisualDL visualization epoch 10
    #112 visualize image path: visual/126.png
    VisualDL visualization epoch 10
    #113 visualize image path: visual/34.png
    VisualDL visualization epoch 10
    #114 visualize image path: visual/136.png
    VisualDL visualization epoch 10
    #115 visualize image path: visual/149.png
    VisualDL visualization epoch 10
    #116 visualize image path: visual/51.png
    VisualDL visualization epoch 10
    #117 visualize image path: visual/112.png
    VisualDL visualization epoch 10
    #118 visualize image path: visual/193.png
    VisualDL visualization epoch 10
    #119 visualize image path: visual/138.png
    VisualDL visualization epoch 10
    #120 visualize image path: visual/190.png
    VisualDL visualization epoch 10
    #121 visualize image path: visual/3.png
    VisualDL visualization epoch 10
    #122 visualize image path: visual/147.png
    VisualDL visualization epoch 10
    #123 visualize image path: visual/56.png
    VisualDL visualization epoch 10
    #124 visualize image path: visual/155.png
    VisualDL visualization epoch 10
    #125 visualize image path: visual/199.png
    VisualDL visualization epoch 10
    #126 visualize image path: visual/140.png
    VisualDL visualization epoch 10
    #127 visualize image path: visual/73.png
    VisualDL visualization epoch 10
    #128 visualize image path: visual/180.png
    VisualDL visualization epoch 10
    #129 visualize image path: visual/9.png
    VisualDL visualization epoch 10
    #130 visualize image path: visual/31.png
    VisualDL visualization epoch 10
    #131 visualize image path: visual/32.png
    VisualDL visualization epoch 10
    #132 visualize image path: visual/151.png
    VisualDL visualization epoch 10
    #133 visualize image path: visual/96.png
    VisualDL visualization epoch 10
    #134 visualize image path: visual/125.png
    VisualDL visualization epoch 10
    #135 visualize image path: visual/42.png
    VisualDL visualization epoch 10
    #136 visualize image path: visual/1.png
    VisualDL visualization epoch 10
    #137 visualize image path: visual/20.png
    VisualDL visualization epoch 10
    #138 visualize image path: visual/91.png
    VisualDL visualization epoch 10
    #139 visualize image path: visual/116.png
    VisualDL visualization epoch 10
    #140 visualize image path: visual/45.png
    VisualDL visualization epoch 10
    #141 visualize image path: visual/88.png
    VisualDL visualization epoch 10
    #142 visualize image path: visual/104.png
    VisualDL visualization epoch 10
    #143 visualize image path: visual/28.png
    VisualDL visualization epoch 10
    #144 visualize image path: visual/87.png
    VisualDL visualization epoch 10
    #145 visualize image path: visual/25.png
    VisualDL visualization epoch 10
    #146 visualize image path: visual/21.png
    VisualDL visualization epoch 10
    #147 visualize image path: visual/49.png
    VisualDL visualization epoch 10
    #148 visualize image path: visual/164.png
    VisualDL visualization epoch 10
    #149 visualize image path: visual/182.png
    VisualDL visualization epoch 10
    #150 visualize image path: visual/16.png
    VisualDL visualization epoch 10
    #151 visualize image path: visual/40.png
    VisualDL visualization epoch 10
    #152 visualize image path: visual/194.png
    VisualDL visualization epoch 10
    #153 visualize image path: visual/26.png
    VisualDL visualization epoch 10
    #154 visualize image path: visual/64.png
    VisualDL visualization epoch 10
    #155 visualize image path: visual/66.png
    VisualDL visualization epoch 10
    #156 visualize image path: visual/105.png
    VisualDL visualization epoch 10
    #157 visualize image path: visual/55.png
    VisualDL visualization epoch 10
    #158 visualize image path: visual/36.png
    VisualDL visualization epoch 10
    #159 visualize image path: visual/17.png
    VisualDL visualization epoch 10
    #160 visualize image path: visual/30.png
    VisualDL visualization epoch 10
    #161 visualize image path: visual/181.png
    VisualDL visualization epoch 10
    #162 visualize image path: visual/127.png
    VisualDL visualization epoch 10
    #163 visualize image path: visual/133.png
    VisualDL visualization epoch 10
    #164 visualize image path: visual/90.png
    VisualDL visualization epoch 10
    #165 visualize image path: visual/48.png
    VisualDL visualization epoch 10
    #166 visualize image path: visual/53.png
    VisualDL visualization epoch 10
    #167 visualize image path: visual/10.png
    VisualDL visualization epoch 10
    #168 visualize image path: visual/111.png
    VisualDL visualization epoch 10
    #169 visualize image path: visual/5.png
    VisualDL visualization epoch 10
    #170 visualize image path: visual/18.png
    VisualDL visualization epoch 10
    #171 visualize image path: visual/189.png
    VisualDL visualization epoch 10
    #172 visualize image path: visual/167.png
    VisualDL visualization epoch 10
    #173 visualize image path: visual/106.png
    VisualDL visualization epoch 10
    #174 visualize image path: visual/39.png
    VisualDL visualization epoch 10
    #175 visualize image path: visual/6.png
    VisualDL visualization epoch 10
    #176 visualize image path: visual/114.png
    VisualDL visualization epoch 10
    #177 visualize image path: visual/103.png
    VisualDL visualization epoch 10
    #178 visualize image path: visual/67.png
    VisualDL visualization epoch 10
    #179 visualize image path: visual/121.png
    VisualDL visualization epoch 10
    #180 visualize image path: visual/98.png
    VisualDL visualization epoch 10
    #181 visualize image path: visual/195.png
    VisualDL visualization epoch 10
    #182 visualize image path: visual/107.png
    VisualDL visualization epoch 10
    #183 visualize image path: visual/119.png
    VisualDL visualization epoch 10
    #184 visualize image path: visual/156.png
    VisualDL visualization epoch 10
    #185 visualize image path: visual/154.png
    VisualDL visualization epoch 10
    #186 visualize image path: visual/38.png
    VisualDL visualization epoch 10
    #187 visualize image path: visual/8.png
    VisualDL visualization epoch 10
    #188 visualize image path: visual/27.png
    VisualDL visualization epoch 10
    #189 visualize image path: visual/102.png
    VisualDL visualization epoch 10
    #190 visualize image path: visual/15.png
    VisualDL visualization epoch 10
    #191 visualize image path: visual/124.png
    VisualDL visualization epoch 10
    #192 visualize image path: visual/92.png
    VisualDL visualization epoch 10
    #193 visualize image path: visual/97.png
    VisualDL visualization epoch 10
    Save model checkpoint to ./saved_model/unet_optic/final
    

从上面这段代码我们可以看到评估的结果：
```
[EVAL]step=1 loss=0.19019 acc=0.9270 IoU=0.6864 step/sec=3.89 | ETA 00:00:12
[EVAL]step=2 loss=0.17055 acc=0.9310 IoU=0.7025 step/sec=7.17 | ETA 00:00:06
[EVAL]#image=7 acc=0.9310 IoU=0.7025
[EVAL]Category IoU: [0.9263 0.4788]
[EVAL]Category Acc: [0.9311 0.9294]
[EVAL]Kappa:0.6131
```
可以在生成的visual文件夹中看到图片
![](visual/1.png)
同时我们可以进行预测部署,如下。


```python

!pip install -r PaddleSeg/deploy/python/requirements.txt
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting python-gflags
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/df/ec/e31302d355bcb9d207d9b858adc1ecc4a6d8c855730c8ba4ddbdd3f8eb8d/python-gflags-3.1.2.tar.gz (52 kB)
         |████████████████████████████████| 52 kB 2.4 MB/s             
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/deploy/python/requirements.txt (line 2)) (5.1.2)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/deploy/python/requirements.txt (line 3)) (1.19.5)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleSeg/deploy/python/requirements.txt (line 4)) (4.1.1.26)
    Collecting futures
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/05/80/f41cca0ea1ff69bce7e7a7d76182b47bb4e1a494380a532af3e8ee70b9ec/futures-3.1.1-py3-none-any.whl (2.8 kB)
    Building wheels for collected packages: python-gflags
      Building wheel for python-gflags (setup.py) ... [?25ldone
    [?25h  Created wheel for python-gflags: filename=python_gflags-3.1.2-py3-none-any.whl size=57366 sha256=e7b1f59d2ea4f273660514f6487e821ead51152831461bb891f456d19b6850c5
      Stored in directory: /home/aistudio/.cache/pip/wheels/96/55/58/a38b0322d9a29dfdb1e20bf0658c2fa8a4c5f1d30655f32297
    Successfully built python-gflags
    Installing collected packages: python-gflags, futures
    Successfully installed futures-3.1.1 python-gflags-3.1.2
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m
    


```python
!python PaddleSeg/deploy/python/infer.py --conf=freeze_model/deploy.yaml --input_dir=work/test
```

    W0226 21:45:29.926196  2323 analysis_predictor.cc:1350] Deprecated. Please use CreatePredictor instead.
    W0226 21:45:31.160086  2323 analysis_predictor.cc:795] The one-time configuration of analysis predictor failed, which may be due to native predictor called first and its configurations taken effect.
    [1m[35m--- Running analysis [ir_graph_build_pass][0m
    [1m[35m--- Running analysis [ir_graph_clean_pass][0m
    [1m[35m--- Running analysis [ir_analysis_pass][0m
    [32m--- Running IR pass [is_test_pass][0m
    [32m--- Running IR pass [simplify_with_basic_ops_pass][0m
    [32m--- Running IR pass [conv_affine_channel_fuse_pass][0m
    [32m--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass][0m
    [32m--- Running IR pass [conv_bn_fuse_pass][0m
    I0226 21:45:31.253378  2323 fuse_pass_base.cc:57] ---  detected 18 subgraphs
    [32m--- Running IR pass [conv_eltwiseadd_bn_fuse_pass][0m
    [32m--- Running IR pass [embedding_eltwise_layernorm_fuse_pass][0m
    [32m--- Running IR pass [multihead_matmul_fuse_pass_v2][0m
    [32m--- Running IR pass [squeeze2_matmul_fuse_pass][0m
    [32m--- Running IR pass [reshape2_matmul_fuse_pass][0m
    [32m--- Running IR pass [flatten2_matmul_fuse_pass][0m
    [32m--- Running IR pass [map_matmul_v2_to_mul_pass][0m
    [32m--- Running IR pass [map_matmul_v2_to_matmul_pass][0m
    [32m--- Running IR pass [map_matmul_to_mul_pass][0m
    [32m--- Running IR pass [fc_fuse_pass][0m
    [32m--- Running IR pass [fc_elementwise_layernorm_fuse_pass][0m
    [32m--- Running IR pass [conv_elementwise_add_act_fuse_pass][0m
    [32m--- Running IR pass [conv_elementwise_add2_act_fuse_pass][0m
    [32m--- Running IR pass [conv_elementwise_add_fuse_pass][0m
    [32m--- Running IR pass [transpose_flatten_concat_fuse_pass][0m
    [32m--- Running IR pass [runtime_context_cache_pass][0m
    [1m[35m--- Running analysis [ir_params_sync_among_devices_pass][0m
    I0226 21:45:31.263988  2323 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
    [1m[35m--- Running analysis [adjust_cudnn_workspace_size_pass][0m
    [1m[35m--- Running analysis [inference_op_replace_pass][0m
    [1m[35m--- Running analysis [memory_optimize_pass][0m
    I0226 21:45:31.303104  2323 memory_optimize_pass.cc:216] Cluster name : image  size: 3959520
    I0226 21:45:31.303133  2323 memory_optimize_pass.cc:216] Cluster name : bilinear_interp_3.tmp_0  size: 84469760
    I0226 21:45:31.303138  2323 memory_optimize_pass.cc:216] Cluster name : bilinear_interp_0.tmp_0  size: 10465280
    I0226 21:45:31.303150  2323 memory_optimize_pass.cc:216] Cluster name : relu_5.tmp_0  size: 21080064
    I0226 21:45:31.303159  2323 memory_optimize_pass.cc:216] Cluster name : concat_3.tmp_0  size: 168939520
    I0226 21:45:31.303164  2323 memory_optimize_pass.cc:216] Cluster name : relu_1.tmp_0  size: 84469760
    I0226 21:45:31.303174  2323 memory_optimize_pass.cc:216] Cluster name : relu_3.tmp_0  size: 42160128
    [1m[35m--- Running analysis [ir_graph_to_program_pass][0m
    I0226 21:45:31.314167  2323 analysis_predictor.cc:714] ======= optimize end =======
    W0226 21:45:31.359719  2323 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0226 21:45:31.364166  2323 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    save result of [work/test/9.jpg] done.
    images_num=[1],preprocessing_time=[0.043113],infer_time=[1.920070],postprocessing_time=[0.490708],total_runtime=[2.454639]
    

原图：

![](work/test/9.jpg)

生成的图片如下

![](work/test/9_jpg_result.png)

## 六、总结与升华
关于我的机器学习之路，很大程度上说就是我的机器视觉学习之路。从这三个例子，虽说管中窥豹，但是，我们可以看到，机器学习可以浓缩为一部机器视觉的发展历史。通过程序不断调试，我们能够更加深入地理解代码工作原理，同时，也可以加深算法数学原理的理解，也更能理会到NIN结构、Matting算法的应用。这次参加AI创想的训练营，收获颇丰。

## 七、个人总结
大家好，我是Kewei Chen，常用笔名irrational。

华中科技大学 2020级 启明学院 本科特优生

中国自动化学会会员 CAA Fellow

全国大学生数学竞赛(CMC)一等奖

华中科技大学节能减排秋季交流赛决赛一等奖

爱好 ：机器学习、算法

博客地址：https://blog.csdn.net/weixin_54227557?type=blog



## 提交链接
aistudio链接：https://aistudio.baidu.com/aistudio/projectdetail/3529339

github链接：https://github.com/Ethan-Chen-plus/paddle-machine-vision

gitee链接：https://gitee.com/qmckw/paddle-machine-vision/tree/master/


```python

```
