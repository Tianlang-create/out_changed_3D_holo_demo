# AI-CGH 系统使用指南

## 项目概述
AI驱动的计算全息（AI-CGH）系统，包含深度学习模型训练、实时推理和光学重现功能。

## 环境要求
- Python 3.9+
- PyTorch 1.10+
- CUDA 11.0+ (GPU加速)
- Mermaid支持 (用于流程图渲染)

## 安装依赖
```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy matplotlib
pip install tensorboard scikit-image
```

## 数据集准备
1. 下载MIT-4K数据集到 `mit-4k/train/` 目录
2. 确保目录结构：
```
mit-4k/train/
├── depth/      # 深度图
└── img_color/  # RGB彩色图像
```

## 模型训练流程

### 1. 数据预处理
```bash
# 检查数据集完整性
python src/utils.py --check-dataset
```

### 2. 开始训练
```bash
# 基础训练配置
python src/train.py \
    --batch_size 8 \
    --epochs 200 \
    --lr 0.0001 \
    --base_channel 32 \
    --save_dir save/CNN_test
```

### 3. 监控训练过程
```bash
# 启动TensorBoard监控
tensorboard --logdir src/runs/CNN_test
```

### 4. 训练参数说明
- `--batch_size`: 批次大小 (默认: 8)
- `--epochs`: 训练轮次 (默认: 200)
- `--lr`: 学习率 (默认: 0.0001)
- `--base_channel`: 基础通道数 (默认: 32)
- `--save_dir`: 模型保存目录

## 实时推理流程

### 1. 模型转换与优化
```bash
# 转换为TensorRT格式 (可选)
python src/trt.py \
    --model_path save/CNN_test/best_model.pth \
    --output_path trt/model.engine
```

### 2. 启动实时推理
```bash
# 使用多进程推理
python src/predict_rgbd_multiprocess.py \
    --model_path save/CNN_test/best_model.pth \
    --output_dir save/CNN_test/out_amp
```

### 3. 性能测试
```bash
# 推理速度测试
python src/time_test.py \
    --model_path save/CNN_test/best_model.pth \
    --iterations 1000
```

## 硬件控制流程

### 1. SLM控制
```bash
# 初始化SLM设备
python src/GCD_ctrl.py --init

# 加载相位图到SLM
python src/GCD_ctrl.py --load-phase phase_image.png
```

### 2. CCD相机控制
```bash
# 初始化CCD相机
python src/gxipy/__init__.py --init-camera

# 开始图像采集
python src/depthcamera_ctrl.py --start-capture
```

### 3. 全系统自动化控制
```bash
# 启动完整AI-CGH流水线
python src/rtholo.py \
    --model_path save/CNN_test/best_model.pth \
    --slm_device 0 \
    --camera_device 0
```

## 验证与测试

### 1. 模型验证
```bash
# 在测试集上验证模型性能
python src/CNN_test.py \
    --model_path save/CNN_test/best_model.pth \
    --test_data mit-4k/test/
```

### 2. 光学重现验证
```bash
# 生成全息图并验证光学质量
python src/propagation_ASM.py \
    --input_dir save/CNN_test/out_amp \
    --output_dir save/CNN_test/reconstructed
```

## 错误处理与调试

### 1. 日志查看
```bash
# 查看训练日志
tail -f log/CNN_test.log
```

### 2. 常见问题解决
```bash
# 检查GPU可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查模型兼容性
python src/utils.py --check-model-compatibility
```

## 高级功能

### 1. 混合精度训练
```bash
python src/train.py --amp --batch_size 16
```

### 2. 分布式训练
```bash
python src/train.py --distributed --gpus 4
```

### 3. 自定义损失函数
```bash
python src/train.py --loss-config config/custom_loss.json
```

## 文件说明

### 核心代码文件
- `src/train.py`: 主训练脚本
- `src/CNN.py`: CNN网络定义
- `src/NET1.py`: NET1网络定义
- `src/dataLoader.py`: 数据加载器
- `src/propagation_ASM.py`: 角谱传播算法

### 硬件控制
- `src/GCD_ctrl.py`: SLM控制接口
- `src/gxipy/`: CCD相机控制库
- `src/depthcamera_ctrl.py`: 深度相机控制

### 工具脚本
- `src/utils.py`: 通用工具函数
- `src/time_test.py`: 性能测试
- `src/trt.py`: TensorRT转换

## 注意事项

1. **硬件要求**: 需要NVIDIA GPU支持CUDA
2. **内存需求**: 训练时需要至少8GB GPU内存
3. **实时性**: 确保SLM刷新率与相机采集速率匹配
4. **数据格式**: 输入图像需要为sRGB线性空间

## 故障排除

如果遇到问题，请检查：
1. CUDA驱动版本是否匹配
2. 模型文件路径是否正确
3. 硬件设备是否正常连接
4. 依赖包版本是否兼容

## 支持与反馈

如有问题请查看日志文件或联系开发团队。