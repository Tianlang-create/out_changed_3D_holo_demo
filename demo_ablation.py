#!/usr/bin/env python3
"""
演示脚本：展示AI-CGH系统消融实验功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_psnr_ssim


def create_demo_data():
    """创建演示用的合成数据"""
    print("创建演示数据...")

    # 创建合成图像数据
    np.random.seed(42)

    # 目标图像（干净图像）
    target_images = []
    recon_images = []

    for i in range(20):  # 20个样本
        # 创建RGB图像
        target_amp = np.random.rand(256, 256, 3).astype(np.float32)

        # 添加不同程度的噪声来模拟不同的实验配置
        noise_level = 0.05 + 0.02 * (i % 4)  # 不同的噪声水平
        recon_amp = target_amp + noise_level * np.random.randn(256, 256, 3).astype(np.float32)
        recon_amp = np.clip(recon_amp, 0, 1)

        target_images.append(target_amp)
        recon_images.append(recon_amp)

    return target_images, recon_images


def run_demo_ablation():
    """运行演示消融实验"""
    print("运行演示消融实验...")

    # 创建输出目录
    output_dir = "./demo_ablation_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # 创建演示数据
    target_images, recon_images = create_demo_data()

    # 定义不同的"实验配置"（用不同的噪声水平模拟）
    experiments = {
        'baseline': {'noise_level': 0.05},
        'no_p_loss': {'noise_level': 0.07},
        'no_f_loss': {'noise_level': 0.09},
        'no_ms_loss': {'noise_level': 0.11}
    }

    results = {}

    # 对每个实验配置进行评估
    for exp_name, config in experiments.items():
        print(f"评估实验: {exp_name}")

        psnr_values = []
        ssim_values = []

        # 评估所有样本
        for i, (target_amp, recon_amp) in enumerate(zip(target_images, recon_images)):
            psnr, ssim_val = get_psnr_ssim(recon_amp, target_amp, multichannel=True)
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)

        # 保存结果
        results[exp_name] = {
            'psnr_mean': np.mean(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'psnr_std': np.std(psnr_values),
            'ssim_std': np.std(ssim_values),
            'psnr_values': psnr_values,
            'ssim_values': ssim_values
        }

        print(f"  PSNR: {results[exp_name]['psnr_mean']:.4f} +/- {results[exp_name]['psnr_std']:.4f}")
        print(f"  SSIM: {results[exp_name]['ssim_mean']:.4f} +/- {results[exp_name]['ssim_std']:.4f}")

    # 生成可视化图表
    generate_demo_plots(results, output_dir)

    # 保存结果到CSV
    save_demo_results(results, output_dir)

    print(f"\n演示完成！结果保存在: {output_dir}")
    print("生成的图表:")
    print("- psnr_comparison.png: PSNR对比柱状图")
    print("- ssim_comparison.png: SSIM对比柱状图")
    print("- combined_metrics.png: 综合指标对比图")
    print("- ablation_summary.csv: 实验结果汇总")


def generate_demo_plots(results, output_dir):
    """生成演示用的可视化图表"""

    exp_names = list(results.keys())

    # PSNR对比图
    plt.figure(figsize=(10, 6))
    psnr_means = [results[exp]['psnr_mean'] for exp in exp_names]
    psnr_stds = [results[exp]['psnr_std'] for exp in exp_names]

    plt.bar(exp_names, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7)
    plt.title('PSNR Comparison (Demo)')
    plt.xlabel('Experiment Type')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'psnr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # SSIM对比图
    plt.figure(figsize=(10, 6))
    ssim_means = [results[exp]['ssim_mean'] for exp in exp_names]
    ssim_stds = [results[exp]['ssim_std'] for exp in exp_names]

    plt.bar(exp_names, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7)
    plt.title('SSIM Comparison (Demo)')
    plt.xlabel('Experiment Type')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'ssim_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 综合指标图
    plt.figure(figsize=(12, 6))

    x = np.arange(len(exp_names))
    width = 0.35

    plt.bar(x - width / 2, psnr_means, width, label='PSNR', alpha=0.7)
    plt.bar(x + width / 2, ssim_means, width, label='SSIM', alpha=0.7)

    plt.xlabel('Experiment Type')
    plt.ylabel('Metric Values')
    plt.title('PSNR and SSIM Comparison (Demo)')
    plt.xticks(x, exp_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_demo_results(results, output_dir):
    """保存演示结果到CSV"""
    import pandas as pd

    # 保存汇总结果
    summary_data = []
    for exp_name, metrics in results.items():
        summary_data.append({
            'Experiment': exp_name,
            'PSNR_Mean': metrics['psnr_mean'],
            'PSNR_Std': metrics['psnr_std'],
            'SSIM_Mean': metrics['ssim_mean'],
            'SSIM_Std': metrics['ssim_std']
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)


if __name__ == "__main__":
    print("=" * 60)
    print("AI-CGH 系统消融实验演示")
    print("=" * 60)

    run_demo_ablation()

    print("\n" + "=" * 60)
    print("如何使用真实数据进行消融实验:")
    print("1. 准备数据集和训练好的模型")
    print("2. 运行: python src/ablation_study.py --data_path <数据集路径> --model_path <模型路径>")
    print("3. 可选参数: --ablation_types baseline no_p_loss no_f_loss ...")
    print("4. 查看结果目录中的图表和CSV文件")
    print("=" * 60)