# fast-neural-style-tensorflow

原作者：[hzy46](https://github.com/hzy46/fast-neural-style-tensorflow)

翻译：[Rehtt](https://github.com/rehtt)

使用Tensorflow实现 [基于感知损失函数的实时风格转换和超分辨率重建](https://arxiv.org/abs/1603.08155)

代码基于 [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/slim) 和 [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style).

## 例子:

|                             配置                             |                             类型                             |                             例子                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [wave.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_wave.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/wave.jpg) |
| [cubist.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/cubist.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_cubist.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/cubist.jpg) |
| [denoised_starry.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/denoised_starry.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_denoised_starry.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/denoised_starry.jpg) |
| [mosaic.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/mosaic.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_mosaic.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/mosaic.jpg) |
| [scream.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/scream.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_scream.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/scream.jpg) |
| [feathers.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/feathers.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_feathers.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/feathers.jpg) |
| [udnie.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/udnie.yml) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_udnie.jpg) | ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/udnie.jpg) |

## 使用环境:

- Python 2.7.x
- <b>Tensorflow >= 1.0</b>

<b>注意：此代码还支持Tensorflow == 0.11。如果是您的版本，请使用commit 5309a2a（git reset --hard 5309a2a）</b>

确保安装了 pyyaml:

```
pip install pyyaml
```

## 使用预训练的模型:

你可以从 [这](https://pan.baidu.com/s/1i4GTS4d) 下载7中预训练的模型

例：使用 "wave.ckpt-done"模型, 运行:

```
python eval.py --model_file <wave.ckpt-done模型的路径> --image_file img/test.jpg
```

结果保存到 generated/res.jpg

例：视频风格化:
需要安装opencv:

```
pip install opencv-python
```

```
python video.py --model_file <wave.ckpt-done模型的路径> --video_file video/test.mp4 --out_video_file generated/test.mp4
```



## 训练模型:

训练新的模型, 首先需要从Tensorflow Slim下载 [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) 。 提取出 vgg_16.ckpt文件。复制到 pretrained/ 目录下:

```
cd <this repo>
mkdir pretrained
cp <your path to vgg_16.ckpt>  pretrained/
```

下载 [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). 将它解压，图片资源在名为"train2014"的目录下。 建立一个软连接:

```
cd <this repo>
ln -s <your path to the folder "train2014"> train2014
```

训练"wave"模型:

```
python train.py -c conf/wave.yml
```

(可选) 使用 tensorboard:

```
tensorboard --logdir models/wave/
```

模型将会保存到 "models/wave/".

查看[配置文件](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml)了解详情
