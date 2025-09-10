以下为项目中所有 Python 文件的功能概述（按路径排序）：

### 根目录
- **demo_ablation.py**：运行消融实验示例脚本，演示不同损失项对结果的影响。
- **draw_train_loss.py**：读取训练日志并绘制损失曲线，可快速可视化模型收敛情况。
- **test_ablation.py**：调用 `src/ablation_study.py`，自动化地对比并验证各消融实验结果。

### src 目录
- **CNN.py**：核心卷积神经网络模型定义，包含正向传播与权重初始化逻辑。
- **CNN_PP.py**：`CNN.py` 的改进/后处理版本，引入额外层或预处理以提高性能。
- **GCD_calibrate.py**：针对 GPU 结构光设备（GCD）进行标定流程，实现相机—投影机空间对齐。
- **GCD_ctrl.py**：GPU 结构光设备（GCD）的控制脚本，负责硬件指令下发及状态监测。
- **NET1.py**：另一种网络架构实现，用于对比或替代 `CNN.py`。
- **ablation_study.py**：统一管理消融实验流程：加载模型、禁用指定损失项、记录指标并输出可视化。
- **alft.py**：实现自适应光场调谐算法 `AdaptiveLightFieldTuner`，通过迭代优化提升全息质量。
- **dataLoader.py**：自定义 PyTorch `Dataset` 与 `DataLoader`，负责图像与深度数据的批量加载及预处理。
- **depthcamera_ctrl.py**：深度相机硬件控制脚本，封装采集与曝光调节接口。
- **focal_frequency_loss/focal_frequency_loss.py**：引入 FFLoss（焦域频率损失）实现，用于生成对比学习或超分场景。
- **getBlaze.py**：生成或提取激光条纹（Blaze）模板，用于光束整形校正。
- **gxipy/** 模块：国产工业相机 SDK 的 Python 封装，包括 `gxiapi.py`、`dxwrapper.py` 等文件，提供相机枚举、触发与参数设置。
- **intelligent_holography/__init__.py**：包入口，延迟导入核心类 `rtholo`、`AdaptiveLightFieldTuner`，并暴露 `_misc` 中的演示 API。
- **intelligent_holography/_misc.py**：无功能负载的示例模块，包含 `noop`、`Placeholder`、`Fibonacci` 等演示函数/类。
- **perceptualloss.py**：实现基于 VGG 特征的感知损失 `PerceptualLoss`，用于提高输出图像主观质量。
- **predict_rgbd_multiprocess.py**：多进程推理脚本，批量处理 RGB-D 数据并生成全息图。
- **propagation_ASM.py**：基于角谱法（ASM）的光场传播工具函数，实现复振幅的频域转移。
- **pytorch_msssim/ssim.py**：第三方 `MS-SSIM` 库的本地副本，用于计算多尺度结构相似度指标。
- **rtholo.py**：实时全息重建核心类 `RtHolo`，封装网络推理、传播及后处理流程。
- **time_test.py**：性能基准测试脚本，统计关键模块耗时。
- **train.py**：主训练脚本，负责参数解析、模型/损失/优化器构建、训练与保存。
- **trt.py**：TensorRT 推理加速脚本，将 PyTorch 模型转换并部署到 TensorRT 引擎。
- **utils.py**：常用工具函数集合（日志、图像处理、张量操作等）。

### src/checkpoints
- **CNN_*/CNN_test** 等子目录下未包含 .py 文件，仅存放模型权重。

### trt 目录（一致性推理工具）
- **trt_create_v1.py**：将训练好的 PyTorch 模型转换为 TensorRT 引擎并保存。
- **trt_inference_v1.py**：加载 TensorRT 引擎并进行推理测试，输出速度与准确率指标。