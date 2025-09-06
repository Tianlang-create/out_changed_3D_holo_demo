# **面向真实场景的实时智能3D全息摄影**

宋贤林、董佳庆、刘明浩、孙泽豪、张子邦、熊江浩、李子龙、刘璇、刘iegen*     
面向真实场景的实时智能3D全息摄影       
《光学快报》第32卷，第14期，24540-24552页（2024年）      
https://doi.org/10.1364/OE.529107    

https://github.com/djq-2000/123/assets/56143723/b8c3cbd7-5bac-45f7-ad20-8a40451dd00d

## 入门指南

此代码运行环境为Python 3.8.17、Pytorch 2.0.1和TensorRT 8.6.0

**- [./src/](./src/)**
- [train.py](./src/train.py)：模型的训练代码
- [NET1.py](./src/NET1.py)：模型1的网络结构
- [dataLoader.py](./src/dataLoader.py)：模型的数据加载器
- [rtholo.py](./src/rtholo.py)：实时全息摄影的代码
- [predict_rgbd_multiprocess.py](./src/predict_rgbd_multiprocess.py)：模型的测试代码
- [trt.py](./src/trt.py)：TensorRT类的代码
- [getBlaze.py](./src/getBlaze.py)：用于生成闪耀光栅的代码
- [GCD_ctrl.py](./src/GCD_ctrl.py)：用于控制电动线性平台的代码
- [depthcamera_ctrl.py](./src/depthcamera_ctrl.py)：用于控制深度相机Realsense D435的代码
- [gxipy](./src/gxipy)：大恒相机的SDK

**- [./trt/](./trt/)**
- [trt_create_v1.py](./trt/trt_create_v1.py)：用于生成TRT模型的代码
- [trt_inference_v1.py](./trt/trt_inference_v1.py)：用于测试TRT模型的代码

## 训练
```
python ./src/train.py --p_loss --l2_loss --num_epochs 60 --data_path <你的训练集地址>
```

## 测试
```
python predict_rgbd_multiprocess.py
```

## 检查点
我们提供了预训练的检查点。预训练模型位于 - [**./src/checkpoints/CNN_1024_30/53.pth**](./src/checkpoints/CNN_1024_30/53.pth)

## 致谢

感谢**[tensor_holography](https://github.com/liangs111/tensor_holography/tree/main)**、**[HoloEncoder](https://github.com/THUHoloLab/Holo-encoder)**、**[HoloEncoder-Pytorch-Version](https://github.com/flyingwolfz/holoencoder-python-version)** 和**[Self-Holo](https://github.com/SXHyeah/Self-Holo)** 的开源。这些工作对我们的研究非常有帮助。


## 其他相关项目
  * 基于分数生成模型的无透镜成像  
[<font size=5>**[论文]**</font>](https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract)  [<font size=5>**[代码]**</font>](https://github.com/yqx7150/LSGM)

  * 基于扩散模型的多相位FZA无透镜成像  
[<font size=5>**[论文]**</font>](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-12-20595&id=531211)  [<font size=5>**[代码]**</font>](https://github.com/yqx7150/MLDM)

  * 基于生成扩散模型的散射介质成像  
[<font size=5>**[论文]**</font>](https://doi.org/10.1063/5.0180176)  [<font size=5>**[代码]**</font>](https://github.com/yqx7150/ISDM)

  * 基于扩散模型的傅里叶单像素成像在极低采样率下的高分辨率迭代重建  
[<font size=5>**[论文]**</font>](https://doi.org/10.1364/OE.510692)  [<font size=5>**[代码]**</font>](https://github.com/yqx7150/FSPI-DM)

  * 双域均值回复扩散模型增强的时间压缩相干衍射成像  
[<font size=5>**[论文]**</font>](https://doi.org/10.1364/OE.517567)  [<font size=5>**[代码]**</font>](https://github.com/yqx7150/DMDTC)