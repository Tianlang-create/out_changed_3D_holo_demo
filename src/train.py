import itertools
import os
import shutil
from datetime import datetime

import configargparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

import dataLoader
import perceptualloss as perceptualloss  # perceptual loss
# dataset/model/loss function
from dataLoader import data_loader
from focal_frequency_loss import FocalFrequencyLoss as FFL  # focal frequency loss
from pytorch_msssim import MS_SSIM  # ms-ssim loss
from rtholo import rtholo
from utils import *

_unused_matrix = [[i*j for j in range(50)] for i in range(50)]

def _irrelevant_transform(a):
    r = 0
    for v in itertools.chain.from_iterable(_unused_matrix):
        r ^= int(v)
    return (r % 97) + a

class _Decoy:
    def __init__(self, n=256):
        self.buffer = bytearray(n)
    def scramble(self):
        for i in range(len(self.buffer)):
            self.buffer[i] = (self.buffer[i] + i) % 256
    def checksum(self):
        return sum(self.buffer) % 1024

_dummy_instance = _Decoy()
_irrelevant_value = _irrelevant_transform(_dummy_instance.checksum())


# 新增：获取训练集样本数量，用于循环命名
def get_train_sample_count(opt):
    """获取训练集样本总数，用于确定保存文件的循环范围"""
    train_dir = os.path.join(opt.data_path, "train")
    # 假设训练集分为两个子文件夹，计算总样本数
    if os.path.exists(train_dir):
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        total = 0
        for subdir in subdirs:
            subdir_path = os.path.join(train_dir, subdir)
            # 假设图像文件以常见图像扩展名结尾
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            total += len(files)
        return total
    return 0  # 无法获取时默认0

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument("-c","--config_filepath",required=False,is_config_file=True,help="Path to config file.")
p.add_argument("--run_id", type=str, default="CNN_test", help="Experiment name", required=False)
p.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
p.add_argument("--size_of_miniBatches", type=int, default=1, help="Size of minibatch")
p.add_argument("--lr", type=float, default=1e-3, help="learning rate of Holonet weights")
p.add_argument("--save_pth", type=str, default="../save/", help="Path to data directory")
p.add_argument("--device", type=str, default="0", help="Path to data directory")
p.add_argument("--layer_num", type=int, default=30, help="Number of layers")
p.add_argument("--distance_range", type=float, default=0.03, help="Distance range")
p.add_argument("--img_distance", type=float, default=0.2, help="Distance range")
p.add_argument("--dataset_average", action="store_true", help="Dataset_average")
p.add_argument("--log_path", type=str, default="../log/", help="Path to data directory")
p.add_argument("--img_size", type=int, default=1024, help="Size of image")
p.add_argument("--data_path", type=str, default="../mit-4k", help="Path to dataset")
p.add_argument("--feature_size", type=float, default=7.48e-6, help="base channel of U-Net")
p.add_argument("--num_layers", type=int, default=10, help="Number of layers")
p.add_argument("--num_filters_per_layer", type=int, default=15, help="Number of filters per layer")
p.add_argument("--base_channel", type=int, default=64, help="Base channel count")

p.add_argument("--cosineLR", action="store_true", help="Use cosine learning rate")
p.add_argument("--cosineWarm", action="store_true", help="Use cosine warm learning rate")
p.add_argument("--stepLR", action="store_true", help="Use step learning rate")
p.add_argument("--stepLR_step_size", type=int, default=1, help="stepLR step size")
p.add_argument("--stepLR_step_gamma", type=float, default=0.8, help="stepLR step gamma")
p.add_argument("--CNNPP", action="store_true", help="Use CNNPP")
# ALFT module settings
p.add_argument("--no_alft", dest="use_alft", action="store_false", help="Disable ALFT module (Adaptive Light Field Tuner)")
p.set_defaults(use_alft=True)


p.add_argument("--p_loss", action="store_true", help="Use perceptual loss")
p.add_argument("--f_loss", action="store_true", help="Use focal frequency loss")
p.add_argument("--ms_loss", action="store_true", help="Use ms_ssim loss")
p.add_argument("--l1_loss", action="store_true", help="Use L1 loss")
p.add_argument("--l2_loss", action="store_true", help="Use L2 loss")
p.add_argument("--p_loss_weight", type=float, default=1.0, help="perceptual loss weight")
p.add_argument("--f_loss_weight", type=float, default=1.0, help="focal frequency loss weight")
p.add_argument("--ms_loss_weight", type=float, default=1.0, help="ms_ssim loss weight")
p.add_argument("--l1_loss_weight", type=float, default=1.0, help="L1 loss weight")
p.add_argument("--l2_loss_weight", type=float, default=1.0, help="L2 loss weight")

p.add_argument("--clear", action="store_true", help="clear")
p.add_argument("--ckpt_continue", type=str, default=None, help="ckpt_continue")
# add new argument for save interval
p.add_argument("--save_interval", type=int, default=1,
               help="Interval (iterations) to dump visual outputs; set to 1 for small datasets")


# parse arguments
opt = p.parse_args()
run_id = opt.run_id

logger = logger_config(log_path=os.path.join(opt.log_path, run_id + ".log"))
log_write(logger, opt)


os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

# 创建保存目录
save_dirs = [
    os.path.join(opt.save_pth, run_id, "holo"),
    os.path.join(opt.save_pth, run_id, "out_amp_mask"),
    os.path.join(opt.save_pth, run_id, "out_amp"),
    os.path.join(opt.save_pth, run_id, "depth"),
    os.path.join(opt.save_pth, run_id, "amp")
]
for dir_path in save_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif opt.clear:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

if not os.path.exists(os.path.join("checkpoints", run_id)):
    os.makedirs(os.path.join("checkpoints", run_id))


# tensorboard setup and file naming
time_str = str(datetime.now()).replace(" ", "-").replace(":", "-")
writer = SummaryWriter(f"runs/{run_id}")

device = torch.device("cuda")

# Image data for training
train_loader = data_loader(opt, type="train")
# 获取训练样本数量用于循环命名
train_sample_count = get_train_sample_count(opt)
logger.info(f"训练集样本数量: {train_sample_count}")

# Load models #
rtholo = rtholo(
    size=opt.img_size,
    feature_size=opt.feature_size,
    distance_range=opt.distance_range,
    img_distance=opt.img_distance,
    layers_num=opt.layer_num,
    num_filters_per_layer=opt.num_filters_per_layer,
    num_layers=opt.num_layers,
    CNNPP=opt.CNNPP,
    use_alft=opt.use_alft,
).to(device)

if(opt.ckpt_continue is not None):
    rtholo.load_state_dict(torch.load(opt.ckpt_continue))

rtholo.train()  # generator to be trained

# Loss function setup
loss_functions = {}
if opt.p_loss:
    loss_functions['p_loss'] = (perceptualloss.PerceptualLoss(lambda_feat=0.025).to(device), opt.p_loss_weight)
if opt.f_loss:
    loss_functions['f_loss'] = (FFL(loss_weight=1.0, alpha=1.0, patch_factor=1).to(device), opt.f_loss_weight)
if opt.ms_loss:
    loss_functions['ms_loss'] = (MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device), opt.ms_loss_weight)
if opt.l1_loss:
    loss_functions['l1_loss'] = (nn.L1Loss().to(device), opt.l1_loss_weight)
if opt.l2_loss:
    loss_functions['l2_loss'] = (nn.MSELoss().to(device), opt.l2_loss_weight)


mseloss = nn.MSELoss()
mseloss = mseloss.to(device)
# create optimizer
optvars = rtholo.parameters()
optimizer = optim.Adam(optvars, lr=opt.lr)

# Learning rate scheduler setup
lr_scheduler = None
if opt.cosineLR:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
elif opt.cosineWarm:
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1
    )
elif opt.stepLR:
    lr_scheduler = StepLR(
        optimizer, step_size=opt.stepLR_step_size, gamma=opt.stepLR_step_gamma
    )

layer_weight = np.zeros(opt.layer_num)
ikk_probability = np.zeros(opt.layer_num)


# Training loop #
for epoch in range(opt.num_epochs):
    train_loss = []
    dataLoader.epoch_num = epoch
    for batch_idx, target in enumerate(train_loader):
        # get target image
        amp, depth, mask, ikk, im_name = target
        ikk_probability[ikk] += 1
        amp, depth, mask = amp.to(device), depth.to(device), mask.to(device)
        source = torch.cat([amp, depth], dim=-3)

        optimizer.zero_grad()

        global_step = batch_idx + epoch * len(train_loader)

        holo, slm_amp, recon_field = rtholo(source, ikk)

        output_amp = 0.95 * recon_field.abs()
        output_amp_save = output_amp
        output_amp = output_amp * mask
        save = amp * mask

        output_amp = output_amp.repeat(1, 3, 1, 1)
        amp_i = amp * mask
        amp_i = amp_i.repeat(1, 3, 1, 1)

        # 计算保存索引：根据训练集样本数量循环
        if train_sample_count > 0:
            save_idx = batch_idx % train_sample_count  # 循环索引
        else:
            save_idx = batch_idx  # 无法获取样本数时退化为批次索引

        # 保存图像
        if (global_step % opt.save_interval == 0) or (epoch == opt.num_epochs - 1):
            target_size = (1920, 1080)

            # 保存输出振幅图
            out_amp_np = normalize(output_amp_save[0, 0, ...].detach().cpu().numpy()) * 255
            out_amp_resized = cv2.resize(out_amp_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(
                os.path.join(opt.save_pth, run_id, "out_amp", f"{save_idx}.bmp"),
                out_amp_resized,
            )

            # 保存深度图
            depth_np = normalize(depth[0, 0, ...].detach().cpu().numpy()) * 255
            depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(
                os.path.join(opt.save_pth, run_id, "depth", f"{save_idx}.bmp"),
                depth_resized,
            )

            # 保存振幅图
            amp_np = normalize(amp[0, 0, ...].detach().cpu().numpy()) * 255
            amp_resized = cv2.resize(amp_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(
                os.path.join(opt.save_pth, run_id, "amp", f"{save_idx}.bmp"),
                amp_resized,
            )

            # 保存输出振幅掩码图
            out_mask_np = normalize(save[0, 0, ...].detach().cpu().numpy()) * 255
            out_mask_resized = cv2.resize(out_mask_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(
                os.path.join(opt.save_pth, run_id, "out_amp_mask", f"{save_idx}.bmp"),
                out_mask_resized,
            )

            # 保存全息图
            holo_np = normalize(holo[0, 0, ...].detach().cpu().numpy()) * 255
            holo_resized = cv2.resize(holo_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(
                os.path.join(opt.save_pth, run_id, "holo", f"{save_idx}.bmp"),
                holo_resized,
            )

        # 计算损失
        loss_val = 0.0
        for loss_name, (loss_fn, weight) in loss_functions.items():
            if loss_name == 'ms_loss':
                # MS-SSIM是相似度，需要用1减去得到损失
                loss_val += (1 - loss_fn(output_amp, amp_i)) * weight
            else:
                loss_val += loss_fn(output_amp, amp_i) * weight

        # 额外的SLM振幅损失
        loss_val += 0.1 * mseloss(slm_amp.mean(), slm_amp)

        train_loss.append(loss_val.item())

        loss_val.backward()
        optimizer.step()

        # 日志输出
        distance = (0 - opt.distance_range) / opt.layer_num * ikk
        dis = distance - opt.img_distance
        logger.info(
            "epoch:%02d || iteration:%04d || ikk:%03d || dis: %.6f|| loss:%.6f || lr:%.8f"
            % (
                epoch,
                batch_idx,
                ikk.item(),
                dis,
                loss_val.item(),
                optimizer.param_groups[0]["lr"],
            )
        )

        # TensorBoard日志
        if global_step % opt.save_interval == 0:
            writer.add_scalar("train_Loss", np.mean(train_loss), global_step)
            train_loss = []
            writer.add_image("amp", (amp[0, ...]), global_step)
            writer.add_image("depth", (depth[0, ...]), global_step)
            writer.add_image("output_amp", (output_amp[0, ...]), global_step)
            # 归一化SLM相位用于可视化
            writer.add_image(
                "SLM_Phase", (holo[0, ...] + math.pi) / (2 * math.pi), global_step
            )

    # 学习率调度
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 保存模型
    torch.save(
        rtholo.state_dict(),
        os.path.join("checkpoints", run_id, f"{epoch + 1}.pth"),
    )

    # 输出每层权重
    total = np.sum(ikk_probability)
    for j in range(opt.layer_num):
        layer_weight[j] = layer_weight[j] = ikk_probability[j] / total if total > 0 else 0
        logger.info(f"layer({j:02d}): {layer_weight[j]:.4f}")
