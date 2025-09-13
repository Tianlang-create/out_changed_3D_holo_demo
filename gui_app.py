# -*- coding: utf-8 -*-
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rtholo import rtholo
from NET1 import NET1
from CNN import CNN
from alft import AdaptiveLightFieldTuner
from propagation_ASM import propagation_ASM
from utils import rect_to_polar, polar_to_rect
from inference_dataset import im2float, resize_keep_aspect  # Import these functions
from depth_estimator import load_midas, predict_depth  # Import MiDaS functions

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, \
    QMessageBox, QCheckBox, QLineEdit, QSizePolicy, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt


class HolographyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能 3D 全息图生成器")
        self.setGeometry(100, 100, 1400, 900)  # 调整窗口大小

        self.current_rgb_image = None
        self.current_depth_image = None
        self.model_size = 1024

        self.midas_model = None
        self.midas_transform = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)  # 设置主布局间距

        # 左侧面板：输入和控制
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(8)  # 设置左侧面板间距
        left_panel_layout.setAlignment(Qt.AlignTop)  # 顶部对齐

        # 标题
        title_label = QLabel("操作面板")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        left_panel_layout.addWidget(title_label)

        # 加载图片按钮
        self.load_image_button = QPushButton("加载图片")
        self.load_image_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; font-size: 16px;")
        self.load_image_button.clicked.connect(self.load_image)
        left_panel_layout.addWidget(self.load_image_button)

        # RGB图片显示
        self.rgb_image_label = QLabel("原始 RGB 图像")
        self.rgb_image_label.setAlignment(Qt.AlignCenter)
        self.rgb_image_label.setMinimumSize(300, 300)  # 最小尺寸
        self.rgb_image_label.setStyleSheet("border: 2px solid #ddd; background-color: #f0f0f0; border-radius: 5px;")
        self.rgb_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        left_panel_layout.addWidget(self.rgb_image_label)

        # 转换为深度图按钮
        self.convert_depth_button = QPushButton("转换为深度图")
        self.convert_depth_button.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 10px; border-radius: 5px; font-size: 16px;")
        self.convert_depth_button.clicked.connect(self.convert_to_depth)
        left_panel_layout.addWidget(self.convert_depth_button)

        # 生成全息图按钮
        self.generate_holo_button = QPushButton("生成全息图")
        self.generate_holo_button.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 10px; border-radius: 5px; font-size: 16px;")
        self.generate_holo_button.clicked.connect(self.generate_hologram)
        left_panel_layout.addWidget(self.generate_holo_button)

        # 批量生成控件
        self.batch_checkbox = QCheckBox("启用批量生成 out_amp")
        self.batch_checkbox.setStyleSheet("font-size: 14px; margin-top: 10px;")
        left_panel_layout.addWidget(self.batch_checkbox)

        batch_controls_layout = QHBoxLayout()
        batch_controls_layout.setSpacing(5)
        batch_controls_layout.addWidget(QLabel("起始距离:"))
        self.start_distance_input = QLineEdit("0.1")
        self.start_distance_input.setValidator(QDoubleValidator())
        self.start_distance_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        batch_controls_layout.addWidget(self.start_distance_input)

        batch_controls_layout.addWidget(QLabel("结束距离:"))
        self.end_distance_input = QLineEdit("0.3")
        self.end_distance_input.setValidator(QDoubleValidator())
        self.end_distance_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        batch_controls_layout.addWidget(self.end_distance_input)

        batch_controls_layout.addWidget(QLabel("步长:"))
        self.step_distance_input = QLineEdit("0.01")
        self.step_distance_input.setValidator(QDoubleValidator())
        self.step_distance_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        batch_controls_layout.addWidget(self.step_distance_input)

        left_panel_layout.addLayout(batch_controls_layout)

        self.batch_generate_button = QPushButton("批量生成 out_amp")
        self.batch_generate_button.setStyleSheet(
            "background-color: #FF5722; color: white; padding: 10px; border-radius: 5px; font-size: 16px;")
        self.batch_generate_button.clicked.connect(self.batch_generate_out_amp)
        left_panel_layout.addWidget(self.batch_generate_button)

        self.status_label = QLabel("状态: 准备就绪")
        self.status_label.setStyleSheet(
            "font-size: 14px; color: #333; margin-top: 10px; padding: 5px; border: 1px solid #eee; background-color: #e0e0e0; border-radius: 3px;")
        left_panel_layout.addWidget(self.status_label)

        left_panel_layout.addStretch(1)
        main_layout.addLayout(left_panel_layout)

        # 右侧面板：输出图像
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(8)
        right_panel_layout.setAlignment(Qt.AlignTop)

        # 标题
        output_title_label = QLabel("结果展示")
        output_title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        output_title_label.setAlignment(Qt.AlignCenter)
        right_panel_layout.addWidget(output_title_label)

        # 图像显示区域
        image_display_layout = QGridLayout()
        image_display_layout.setSpacing(10)

        self.depth_image_label = QLabel("转换后的深度图像")
        self.depth_image_label.setAlignment(Qt.AlignCenter)
        self.depth_image_label.setMinimumSize(300, 300)
        self.depth_image_label.setStyleSheet("border: 2px solid #ddd; background-color: #f0f0f0; border-radius: 5px;")
        self.depth_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_display_layout.addWidget(self.depth_image_label, 0, 0)

        self.holo_image_label = QLabel("生成的全息图")
        self.holo_image_label.setAlignment(Qt.AlignCenter)
        self.holo_image_label.setMinimumSize(300, 300)
        self.holo_image_label.setStyleSheet("border: 2px solid #ddd; background-color: #f0f0f0; border-radius: 5px;")
        self.holo_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_display_layout.addWidget(self.holo_image_label, 0, 1)

        self.out_amp_image_label = QLabel("输出振幅图像")
        self.out_amp_image_label.setAlignment(Qt.AlignCenter)
        self.out_amp_image_label.setMinimumSize(300, 300)
        self.out_amp_image_label.setStyleSheet("border: 2px solid #ddd; background-color: #f0f0f0; border-radius: 5px;")
        self.out_amp_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_display_layout.addWidget(self.out_amp_image_label, 1, 0, 1, 2)  # 跨两列显示

        right_panel_layout.addLayout(image_display_layout)
        right_panel_layout.addStretch(1)
        main_layout.addLayout(right_panel_layout)

        self.setLayout(main_layout)

        self.model = None
        self.load_model()
        self.load_midas_model()

    def load_midas_model(self):
        try:
            self.status_label.setText("正在加载 MiDaS 模型...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.midas_model, self.midas_transform = load_midas(device=device)
            self.status_label.setText("MiDaS 模型加载成功。")
            print("MiDaS model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "MiDaS Model Load Error", f"Failed to load MiDaS model: {e}")
            self.status_label.setText(f"MiDaS 模型加载失败: {e}")
            print(f"Failed to load MiDaS model: {e}")

    def load_model(self):
        try:
            self.model = rtholo(size=self.model_size, mode='test', use_alft=False)  # Use self.model_size
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'src', 'checkpoints', 'CNN_1024_30',
                                           '53.pth')  # Adjust path as needed
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU first
                self.model.load_state_dict(checkpoint)
                self.model.eval()  # Set model to evaluation mode
                if torch.cuda.is_available():
                    self.model.cuda()  # Move model to GPU if available
                self.status_label.setText("模型加载成功。")
                print("Model loaded successfully.")
            else:
                QMessageBox.warning(self, "Model Load Error", f"Model checkpoint not found at {checkpoint_path}")
                self.status_label.setText(f"模型加载失败: 检查点未找到 {checkpoint_path}")
                print(f"Model checkpoint not found at {checkpoint_path}")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model: {e}")
            self.status_label.setText(f"模型加载失败: {e}")
            print(f"Failed to load model: {e}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "PNG 图片 (*.png);;所有文件 (*)")
        if file_path:
            try:
                # Load image using OpenCV
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("无法加载图片，请检查文件路径和格式。")

                # Convert OpenCV image (BGR) to QImage (RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Display image in QLabel
                pixmap = QPixmap.fromImage(q_img)
                self.rgb_image_label.setPixmap(
                    pixmap.scaled(self.rgb_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.rgb_image_label.setAlignment(Qt.AlignCenter)
                self.current_rgb_image = img_rgb  # Store the RGB image for further processing
                self.status_label.setText(f"图片加载成功: {file_path}")
            except Exception as e:
                self.status_label.setText(f"图片加载失败: {e}")

    def save_image(self, image_data, folder_name, file_name):
        save_dir = os.path.join(os.path.dirname(__file__), "GUI_USER_SAVE", folder_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, image_data)
        print(f"Image saved to {save_path}")

    def convert_to_depth(self):
        if self.current_rgb_image is not None:
            if self.midas_model is None or self.midas_transform is None:
                self.status_label.setText("MiDaS 模型未加载，请等待或检查加载错误。")
                return
            try:
                self.status_label.setText("正在使用 MiDaS 模型生成深度图...")
                device = next(self.midas_model.parameters()).device  # Get the device of the MiDaS model
                depth_map = predict_depth(self.midas_model, self.midas_transform, self.current_rgb_image, device=device)

                # Normalize depth map to 0-255 for display
                normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Convert numpy array to QImage for display
                h, w = normalized_depth.shape
                bytes_per_line = w
                q_img = QImage(normalized_depth.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(q_img)
                self.depth_image_label.setPixmap(
                    pixmap.scaled(self.depth_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.depth_image_label.setAlignment(Qt.AlignCenter)
                self.status_label.setText("深度图生成成功。")
                self.current_depth_image = normalized_depth  # Store the generated depth image
                self.save_image(normalized_depth, "depth", "depth_map.png")
            except Exception as e:
                self.status_label.setText(f"深度图生成失败: {e}")
        else:
            self.status_label.setText("请先加载一张 RGB 图片。")

    def generate_hologram(self):
        if self.model is None:
            self.status_label.setText("模型未加载，请检查模型路径。")
            return

        if self.current_rgb_image is None or self.current_depth_image is None:
            self.status_label.setText("请先加载 RGB 图片并转换为 Depth 图片。")
            return

        try:
            # Preprocess RGB image for amplitude input (convert to grayscale and apply transformations)
            # This part mimics the amplitude processing in inference_dataset.py
            amp_img_gray = cv2.cvtColor(self.current_rgb_image, cv2.COLOR_RGB2GRAY)
            amp_img_gray = amp_img_gray[..., np.newaxis]  # (H, W, 1)
            im = im2float(amp_img_gray, dtype=np.float32)
            low_val = im <= 0.04045
            im[low_val] = 25 / 323 * im[low_val]
            im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11) / 211) ** (12 / 5)
            amp = np.sqrt(im)
            amp = np.transpose(amp, (2, 0, 1))  # (C, H, W)
            amp = resize_keep_aspect(amp, [self.model_size, self.model_size])
            amp = np.reshape(amp, (1, 1, self.model_size, self.model_size))  # (1, C, H, W) for batch, C=1 for grayscale

            # Preprocess simulated depth image
            # This part mimics the depth processing in inference_dataset.py
            depth = self.current_depth_image[..., np.newaxis]  # (H, W, 1)
            depth = im2float(depth, dtype=np.float32)
            depth = np.transpose(depth, (2, 0, 1))  # (C, H, W)
            depth = resize_keep_aspect(depth, [self.model_size, self.model_size])
            depth = np.reshape(depth,
                               (1, 1, self.model_size, self.model_size))  # (1, C, H, W) for batch, C=1 for grayscale
            depth = 1 - depth  # Invert depth

            # Convert to torch tensors and move to device
            device = next(self.model.parameters()).device  # Get the device of the model
            amp_t = torch.from_numpy(amp).to(device)
            depth_t = torch.from_numpy(depth).to(device)

            # Concatenate amplitude and depth to form the source input for rtholo
            source_t = torch.cat((amp_t, depth_t), 1)  # (1, 2, H, W)

            # Perform inference
            with torch.no_grad():
                holo, slm_amp, out_amp = self.model(source_t, ikk=None)  # ikk is for training, not needed for inference

            # Process holo for display
            # Assuming holo is a complex tensor, we take its amplitude for visualization
            holo_display = holo.abs().squeeze().cpu().numpy()
            holo_display = cv2.normalize(holo_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = holo_display.shape
            q_holo_img = QImage(holo_display.data, w, h, w, QImage.Format_Grayscale8)
            pixmap_holo = QPixmap.fromImage(q_holo_img)
            self.holo_image_label.setPixmap(
                pixmap_holo.scaled(self.holo_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.holo_image_label.setAlignment(Qt.AlignCenter)
            self.save_image(holo_display, "holo", "hologram.png")

            # Process out_amp for display
            out_amp_display = out_amp.abs().squeeze().cpu().numpy()
            # Apply a threshold to remove background stray light
            threshold = 0.05 * out_amp_display.max()  # Set threshold as 5% of max intensity
            out_amp_display[out_amp_display < threshold] = 0
            out_amp_display = cv2.normalize(out_amp_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Display the image in the GUI
            h, w = out_amp_display.shape
            q_out_amp_img = QImage(out_amp_display.data, w, h, w, QImage.Format_Grayscale8)
            pixmap_out_amp = QPixmap.fromImage(q_out_amp_img)
            self.out_amp_image_label.setPixmap(
                pixmap_out_amp.scaled(self.out_amp_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.out_amp_image_label.setAlignment(Qt.AlignCenter)

            self.status_label.setText("全息图生成成功。")

        except Exception as e:
            self.status_label.setText(f"全息图生成失败: {e}")

    def batch_generate_out_amp(self):
        if self.model is None:
            self.status_label.setText("模型未加载，请检查模型路径。")
            return

        if self.current_rgb_image is None or self.current_depth_image is None:
            self.status_label.setText("请先加载 RGB 图片并转换为 Depth 图片。")
            return

        try:
            start_dist = float(self.start_distance_input.text())
            end_dist = float(self.end_distance_input.text())
            step_dist = float(self.step_distance_input.text())

            if start_dist >= end_dist:
                QMessageBox.warning(self, "输入错误", "起始距离必须小于结束距离。")
                return

            self.status_label.setText("正在批量生成 out_amp 图像并动态展示...")

            # Preprocess RGB image for amplitude input
            amp_img_gray = cv2.cvtColor(self.current_rgb_image, cv2.COLOR_RGB2GRAY)
            amp_img_gray = amp_img_gray[..., np.newaxis]
            im = im2float(amp_img_gray, dtype=np.float32)
            low_val = im <= 0.04045
            im[low_val] = 25 / 323 * im[low_val]
            im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11) / 211) ** (12 / 5)
            amp = np.sqrt(im)
            amp = np.transpose(amp, (2, 0, 1))
            amp = resize_keep_aspect(amp, [self.model_size, self.model_size])
            amp = np.reshape(amp, (1, 1, self.model_size, self.model_size))

            # Preprocess simulated depth image
            depth = self.current_depth_image[..., np.newaxis]
            depth = im2float(depth, dtype=np.float32)
            depth = np.transpose(depth, (2, 0, 1))
            depth = resize_keep_aspect(depth, [self.model_size, self.model_size])
            depth = np.reshape(depth, (1, 1, self.model_size, self.model_size))
            depth = 1 - depth

            device = next(self.model.parameters()).device
            amp_t = torch.from_numpy(amp).to(device)
            depth_t = torch.from_numpy(depth).to(device)
            source_t = torch.cat((amp_t, depth_t), 1)

            distances = np.arange(start_dist, end_dist + step_dist, step_dist)
            if len(distances) == 0:
                QMessageBox.warning(self, "生成警告", "没有生成任何距离，请检查输入参数。")
                self.status_label.setText("批量 out_amp 动态展示完成。")
                return

            for i, dist in enumerate(distances):
                self.status_label.setText(f"正在生成距离 {dist:.2f} 的 out_amp 图像 ({i + 1}/{len(distances)})...")
                QApplication.processEvents()  # Allow GUI to update

                with torch.no_grad():
                    # Pass z_distance to the model's forward method
                    _, _, out_amp = self.model(source_t, ikk=None)

                # Process out_amp for display
                out_amp_display = out_amp.abs().squeeze().cpu().numpy()
                # Apply a threshold to remove background stray light
                threshold = 0.05 * out_amp_display.max()
                out_amp_display[out_amp_display < threshold] = 0
                out_amp_display = cv2.normalize(out_amp_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Display the image in the GUI
                h, w = out_amp_display.shape
                q_out_amp_img = QImage(out_amp_display.data, w, h, w, QImage.Format_Grayscale8)
                pixmap_out_amp = QPixmap.fromImage(q_out_amp_img)
                self.out_amp_image_label.setPixmap(
                    pixmap_out_amp.scaled(self.out_amp_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.out_amp_image_label.setAlignment(Qt.AlignCenter)

                # Small delay to make the dynamic display visible
                QApplication.processEvents()  # Process events to update GUI
                import time
                time.sleep(0.1)  # Adjust delay as needed

            self.status_label.setText("批量 out_amp 动态展示完成。")
            QMessageBox.information(self, "批量生成", "所有 out_amp 图像已成功动态展示。")

        except Exception as e:
            self.status_label.setText(f"批量 out_amp 动态展示失败: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Optional: for a modern look
    window = HolographyApp()
    window.show()
    sys.exit(app.exec_())