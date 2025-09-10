# 训练流程示意图（Mermaid）

```mermaid
graph TD
    A[开始] --> B[加载配置与参数]
    B --> C[加载训练数据]
    C --> D[构建模型]
    D --> E[前向推理]
    E --> F[计算损失<br/>(L1/L2/Perceptual/FFL/MS-SSIM)]
    F --> G[反向传播与优化]
    G --> H[记录日志并可视化]
    H --> I{达到终止条件?}
    I -- 否 --> C
    I -- 是 --> J[保存权重]
    J --> K[结束]
```