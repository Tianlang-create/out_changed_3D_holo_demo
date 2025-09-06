# AI-CGH 系统流程图集

## 1. 控制软件自动化流程

### 全自动化控制流程
```mermaid
graph TB
    subgraph "控制软件架构"
        A[用户界面<br/>参数设置与监控]
        B[AI模型调用模块]
        C[SLM控制接口]
        D[CCD相机采集模块]
        E[流程调度器]
        F[数据存储与分析]
    end
    
    subgraph "硬件设备"
        G[GPU计算设备]
        H[相位空间光调制器 SLM]
        I[CCD相机]
        J[激光光源]
        K[光学系统]
    end
    
    A -->|启动指令| E
    E -->|模型调用| B
    B -->|AI推理| G
    G -->|生成相位图| C
    C -->|加载相位| H
    H -->|光学调制| J
    J -->|照明| K
    K -->|全息重现| I
    I -->|图像采集| D
    D -->|数据返回| F
    F -->|质量评估| A
    
    E -->|同步触发| C
    E -->|同步触发| D
    
    style A fill:#e1f5fe
    style B fill:#bbdefb
    style C fill:#bbdefb
    style D fill:#bbdefb
    style E fill:#ffcc80
    style F fill:#c8e6c9
    style G fill:#f5f5f5
    style H fill:#f5f5f5
    style I fill:#f5f5f5
    style J fill:#f5f5f5
    style K fill:#f5f5f5
```

### 实时时序控制
```mermaid
gantt
    title 实时控制时序图 (33ms周期，30fps)
    dateFormat HH:mm:ss.SSS
    axisFormat %Lms
    
    section AI推理模块
    模型加载          :a1, 00:00:00.000, 2ms
    AI推理计算        :a2, after a1, 2.3ms
    相位图生成        :a3, after a2, 0.7ms
    
    section SLM控制模块
    相位数据传输      :b1, after a3, 5ms
    SLM刷新等待       :b2, after b1, 23ms
    
    section CCD采集模块
    相机触发          :c1, after b1, 1ms
    图像曝光          :c2, after c1, 1.7ms
    数据传输          :c3, after c2, 1ms
    
    section 系统调度
    同步协调          :d1, 00:00:00.000, 33ms
```

## 2. 模型训练详细过程

### 完整训练流程
```mermaid
graph TB
    subgraph "数据准备阶段"
        A[MIT-4K数据集] --> B[数据预处理<br/>sRGB转线性, 深度归一化]
        B --> C[数据增强<br/>旋转+缩放+色彩抖动]
        C --> D[构建数据加载器<br/>批量大小: 8]
    end
    
    subgraph "模型初始化"
        E[网络参数初始化] --> F[优化器设置 Adam lr=1e-4]
        F --> G[学习率调度器]
        G --> H[损失函数组合配置]
    end
    
    subgraph "训练循环"
        I[数据批量加载] --> J[前向传播<br/>NET1→角谱→CNN]
        J --> K[多损失函数计算]
        K --> L[反向传播与梯度计算]
        L --> M[参数更新]
        M --> N[模型检查点保存]
        N --> O[训练日志记录]
    end
    
    subgraph "验证与监控"
        P[验证集评估] --> Q[性能指标计算 SSIM/PSNR]
        R[Tensorboard监控] --> S[损失曲线可视化]
        T[早停机制] --> U[最佳模型选择]
    end
    
    D --> I
    H --> K
    O --> P
    O --> R
    O --> T
    
    style D fill:#e1f5fe
    style M fill:#fff3e0
    style U fill:#f1f8e9
```

### 训练优化策略
```mermaid
graph LR
    subgraph "混合精度训练"
        A[FP32主权重] --> B[FP16前向计算]
        B --> C[梯度缩放]
        C --> D[FP32梯度更新]
    end
    
    subgraph "梯度累积"
        E[小批次计算] --> F[梯度累积]
        F --> G[等效大批次更新]
    end
    
    subgraph "分布式训练"
        H[数据并行] --> I[多GPU加速]
        J[模型并行] --> K[超大模型支持]
    end
    
    style D fill:#fff3e0
    style G fill:#fff3e0
    style I fill:#fff3e0
    style K fill:#fff3e0
```

## 3. 损失函数下降曲线

### 多损失函数变化趋势
```mermaid
flowchart LR
    A[训练轮次] --> B[损失值]
    
    subgraph "总损失"
        direction TB
        T1[0.85] --> T2[0.42] --> T3[0.28] --> T4[0.19] --> T5[0.15] --> T6[0.12] --> T7[0.10] --> T8[0.09] --> T9[0.08]
    end
    
    subgraph "振幅损失"
        direction TB
        A1[0.65] --> A2[0.31] --> A3[0.21] --> A4[0.14] --> A5[0.11] --> A6[0.09] --> A7[0.08] --> A8[0.07] --> A9[0.06]
    end
    
    subgraph "相位损失"
        direction TB
        P1[0.25] --> P2[0.12] --> P3[0.08] --> P4[0.05] --> P5[0.04] --> P6[0.03] --> P7[0.025] --> P8[0.02] --> P9[0.018]
    end
    
    subgraph "感知损失"
        direction TB
        S1[0.18] --> S2[0.09] --> S3[0.06] --> S4[0.04] --> S5[0.03] --> S6[0.025] --> S7[0.02] --> S8[0.016] --> S9[0.014]
    end
```

### 性能指标提升曲线
```mermaid
flowchart LR
    A[训练轮次] --> B[性能指标]
    
    subgraph "SSIM"
        direction TB
        S1[0.72] --> S2[0.78] --> S3[0.82] --> S4[0.84] --> S5[0.85] --> S6[0.86] --> S7[0.86] --> S8[0.86] --> S9[0.86]
    end
    
    subgraph "PSNR(dB)"
        direction TB
        P1[24.5] --> P2[26.2] --> P3[27.1] --> P4[27.5] --> P5[27.8] --> P6[27.9] --> P7[28.0] --> P8[28.0] --> P9[28.0]
    end
```

### 学习率调整过程
```mermaid
flowchart TB
    A[训练轮次] --> B[学习率变化]
    
    subgraph "学习率调度"
        direction LR
        L1[1e-4] --> L2[1e-4] --> L3[1e-4] --> L4[1e-4] --> L5[5e-5] --> L6[5e-5] --> L7[2.5e-5] --> L8[2.5e-5] --> L9[1.25e-5]
    end
```

## 4. 错误处理与恢复机制

### 系统容错流程
```mermaid
graph TB
    subgraph "正常流程"
        A[开始控制循环] --> B[AI推理成功]
        B --> C[SLM加载成功]
        C --> D[CCD采集成功]
        D --> E[数据处理完成]
        E --> F[下一周期]
    end
    
    subgraph "错误处理"
        G[AI推理超时] --> H[重试机制 最大3次]
        I[SLM通信失败] --> J[设备重新初始化]
        K[CCD采集失败] --> L[相机重启]
        M[数据异常] --> N[跳过当前帧]
    end
    
    B -->|超时| G
    C -->|失败| I
    D -->|失败| K
    E -->|异常| M
    
    H -->|成功| C
    J -->|成功| C
    L -->|成功| D
    N --> F
    
    style F fill:#f1f8e9
    style H fill:#ffcdd2
    style J fill:#ffcdd2
    style L fill:#ffcdd2
    style N fill:#ffcdd2
```

## 5. 系统性能监控

### 实时性能指标
```mermaid
pie title 实时帧率监控
    "优 (25-36fps)" : 30
    "中 (15-25fps)" : 5
    "低 (0-15fps)" : 1
```

```mermaid
pie title 系统资源使用率
    "正常使用 (30-80%)" : 65
    "高使用 (80-100%)" : 10
    "低使用 (0-30%)" : 25
```

## 使用说明

这些流程图使用Mermaid语法编写，完整展示了AI-CGH系统的：

1. **控制软件自动化流程**：从AI推理到光学重现的完整闭环
2. **模型训练详细过程**：数据准备到模型优化的完整流程
3. **损失函数下降曲线**：多损失函数的训练收敛过程
4. **系统监控与容错**：错误处理和恢复机制

所有图表都可以在支持Mermaid的Markdown查看器中正确渲染显示。