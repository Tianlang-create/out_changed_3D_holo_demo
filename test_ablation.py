#!/usr/bin/env python3
"""
Simple test script for ablation study functionality
This script tests the basic components without requiring full dataset
"""

import os
import numpy as np
import torch
from src.utils import get_psnr_ssim

def test_metrics_calculation():
    """Test PSNR and SSIM calculation with synthetic data"""
    print("Testing PSNR and SSIM calculation...")
    
    # Create synthetic test images (smaller size for SSIM compatibility)
    np.random.seed(42)
    
    # Target image (clean)
    target_amp = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Reconstructed image (with some noise)
    recon_amp = target_amp + 0.1 * np.random.randn(256, 256, 3).astype(np.float32)
    recon_amp = np.clip(recon_amp, 0, 1)
    
    # Calculate metrics
    psnr_val, ssim_val = get_psnr_ssim(recon_amp, target_amp, multichannel=True)
    
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    
    # Verify metrics are reasonable
    assert psnr_val > 10, f"PSNR too low: {psnr_val}"
    assert ssim_val > 0.5, f"SSIM too low: {ssim_val}"
    
    print("? Metrics calculation test passed!")
    return True

def test_tensor_conversion():
    """Test tensor conversion and device handling"""
    print("\nTesting tensor conversion...")
    
    # Test numpy to tensor conversion
    numpy_array = np.random.rand(1, 3, 256, 256).astype(np.float32)
    tensor = torch.from_numpy(numpy_array)
    
    # Test device placement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor device: {tensor.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    assert tensor.shape == (1, 3, 256, 256), "Tensor shape mismatch"
    print("? Tensor conversion test passed!")
    return True

def test_directory_creation():
    """Test output directory creation"""
    print("\nTesting directory creation...")
    
    test_dir = "./test_output"
    
    # Clean up if exists
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # Create directories
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'plots'), exist_ok=True)
    
    # Verify directories exist
    assert os.path.exists(test_dir), "Main directory not created"
    assert os.path.exists(os.path.join(test_dir, 'samples')), "Samples directory not created"
    assert os.path.exists(os.path.join(test_dir, 'plots')), "Plots directory not created"
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
    print("? Directory creation test passed!")
    return True

def test_ablation_configs():
    """Test ablation experiment configurations"""
    print("\nTesting ablation configurations...")
    
    # Define test configurations
    experiments = {
        'baseline': {
            'p_loss': True, 'p_loss_weight': 1.0,
            'f_loss': True, 'f_loss_weight': 1.0,
            'ms_loss': True, 'ms_loss_weight': 1.0,
            'l1_loss': True, 'l1_loss_weight': 1.0,
            'l2_loss': True, 'l2_loss_weight': 1.0
        },
        'no_p_loss': {
            'p_loss': False, 'p_loss_weight': 0.0,
            'f_loss': True, 'f_loss_weight': 1.0,
            'ms_loss': True, 'ms_loss_weight': 1.0,
            'l1_loss': True, 'l1_loss_weight': 1.0,
            'l2_loss': True, 'l2_loss_weight': 1.0
        }
    }
    
    # Verify configurations
    assert 'baseline' in experiments, "Baseline config missing"
    assert 'no_p_loss' in experiments, "Ablation config missing"
    
    # Verify parameter types
    for exp_name, config in experiments.items():
        assert isinstance(config['p_loss'], bool), f"p_loss should be bool in {exp_name}"
        assert isinstance(config['p_loss_weight'], float), f"p_loss_weight should be float in {exp_name}"
        assert 0 <= config['p_loss_weight'] <= 1, f"p_loss_weight out of range in {exp_name}"
    
    print("? Ablation configurations test passed!")
    return True

def main():
    """Run all tests"""
    print("Running ablation study functionality tests...")
    print("=" * 60)
    
    try:
        test_metrics_calculation()
        test_tensor_conversion()
        test_directory_creation()
        test_ablation_configs()
        
        print("\n" + "=" * 60)
        print("? All tests passed! The ablation study framework is ready.")
        print("\nNext steps:")
        print("1. Prepare your dataset and model checkpoint")
        print("2. Run: python src/ablation_study.py --data_path <dataset_path> --model_path <model_path>")
        print("3. Check results in the output directory")
        
        return True
        
    except Exception as e:
        print(f"\n? Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)