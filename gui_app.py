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
    QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class HolographyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent 3D Holography GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.current_rgb_image = None
        self.current_depth_image = None
        self.model_size = 1024  # Initialize model_size here

        self.midas_model = None
        self.midas_transform = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left panel for input and controls
        left_panel_layout = QVBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        left_panel_layout.addWidget(self.load_image_button)

        self.rgb_image_label = QLabel("Original RGB Image")
        self.rgb_image_label.setAlignment(Qt.AlignCenter)
        self.rgb_image_label.setFixedSize(400, 400)  # Placeholder size
        self.rgb_image_label.setStyleSheet("border: 1px solid gray;")
        left_panel_layout.addWidget(self.rgb_image_label)

        self.convert_depth_button = QPushButton("Convert to Depth")
        self.convert_depth_button.clicked.connect(self.convert_to_depth)
        left_panel_layout.addWidget(self.convert_depth_button)

        self.generate_holo_button = QPushButton("Generate Hologram")
        self.generate_holo_button.clicked.connect(self.generate_hologram)
        left_panel_layout.addWidget(self.generate_holo_button)

        self.status_label = QLabel("Status: Ready")  # Add a status label
        left_panel_layout.addWidget(self.status_label)

        left_panel_layout.addStretch(1)
        main_layout.addLayout(left_panel_layout)

        # Right panel for output images
        right_panel_layout = QVBoxLayout()

        self.depth_image_label = QLabel("Converted Depth Image")
        self.depth_image_label.setAlignment(Qt.AlignCenter)
        self.depth_image_label.setFixedSize(400, 400)  # Placeholder size
        self.depth_image_label.setStyleSheet("border: 1px solid gray;")
        right_panel_layout.addWidget(self.depth_image_label)

        self.holo_image_label = QLabel("Generated Hologram")
        self.holo_image_label.setAlignment(Qt.AlignCenter)
        self.holo_image_label.setFixedSize(400, 400)  # Placeholder size
        self.holo_image_label.setStyleSheet("border: 1px solid gray;")
        right_panel_layout.addWidget(self.holo_image_label)

        self.out_amp_image_label = QLabel("Output Amplitude Image")
        self.out_amp_image_label.setAlignment(Qt.AlignCenter)
        self.out_amp_image_label.setFixedSize(400, 400)
        self.out_amp_image_label.setStyleSheet("border: 1px solid gray;")
        right_panel_layout.addWidget(self.out_amp_image_label)

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
            h, w = out_amp_display.shape
            q_out_amp_img = QImage(out_amp_display.data, w, h, w, QImage.Format_Grayscale8)
            pixmap_out_amp = QPixmap.fromImage(q_out_amp_img)
            self.out_amp_image_label.setPixmap(
                pixmap_out_amp.scaled(self.out_amp_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.out_amp_image_label.setAlignment(Qt.AlignCenter)
            self.save_image(out_amp_display, "out_amp", "output_amplitude.png")

            self.status_label.setText("全息图和输出振幅生成成功。")

        except Exception as e:
            self.status_label.setText(f"全息图生成失败: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Optional: for a modern look
    window = HolographyApp()
    window.show()
    sys.exit(app.exec_())