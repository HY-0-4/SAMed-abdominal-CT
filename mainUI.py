import sys
import os
import re
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QSlider, QPushButton, QGroupBox, QGridLayout,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QComboBox, QFileDialog, QStatusBar, QFrame, QSizePolicy,
    QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPalette, QIcon
import threading


# 器官名称与颜色映射
ORGAN_NAMES = {
    0: '背景',
    1: '脾脏',
    2: '右肾',
    3: '左肾',
    4: '胆囊',
    5: '肝脏',
    6: '胃',
    7: '主动脉',
    8: '胰腺',
}

ORGAN_COLORS = {
    0: (0, 0, 0, 0),         # 背景透明
    1: (255, 0, 0, 180),     # 脾脏 - 红
    2: (0, 255, 0, 180),     # 右肾 - 绿
    3: (0, 200, 255, 180),   # 左肾 - 青
    4: (255, 255, 0, 180),   # 胆囊 - 黄
    5: (255, 128, 0, 180),   # 肝脏 - 橙
    6: (200, 0, 255, 180),   # 胃 - 紫
    7: (0, 0, 255, 180),     # 主动脉 - 蓝
    8: (255, 0, 255, 180),   # 胰腺 - 品红
}

PREDICTIONS_DIR = './output/predictions'
LOG_FILE = './output/test_log/log.txt'


class NiiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('腹部多器官 CT 图像分割系统')
        self.setMinimumSize(1400, 850)

        # 数据
        self.img_data = None
        self.pred_data = None
        self.gt_data = None
        self.current_slice = 0
        self.total_slices = 0
        self.current_case = None

        # 窗宽窗位（图像已归一化到0~1）
        self.window_width = 1.0
        self.window_center = 0.5

        # 叠加透明度
        self.overlay_alpha = 0.5

        # TTA 相关
        self.tta_model = None  # 进程内加载的模型
        self.pred_tta_data = None  # TTA 预测数据（简单均值融合）
        self.pred_tta_uw_data = None  # TTA 预测数据（不确定性加权融合）
        self.entropy_data = None  # 每切片熵图数据 (H, W, N_slices)
        self.current_class_overlap_metrics = {}

        self._init_ui()
        self._apply_style()
        self._load_case_list()
        self._load_log_metrics()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ========== 左侧面板 ==========
        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)

        # 病例列表
        case_group = QGroupBox('病例列表')
        case_layout = QVBoxLayout(case_group)
        self.case_list = QListWidget()
        self.case_list.currentItemChanged.connect(self._on_case_selected)
        case_layout.addWidget(self.case_list)

        # 打开其他文件夹
        self.btn_open_folder = QPushButton('打开其他文件夹')
        self.btn_open_folder.clicked.connect(self._open_folder)
        case_layout.addWidget(self.btn_open_folder)
        left_panel.addWidget(case_group)

        # 模型推理
        infer_group = QGroupBox('模型推理')
        infer_layout = QVBoxLayout(infer_group)
        # TTA 进程内推理按钮
        self.btn_run_tta = QPushButton('TTA 推理')
        self.btn_run_tta.setToolTip('在 UI 进程中直接加载模型并执行 TTA 推理，支持 CPU/GPU')
        self.btn_run_tta.clicked.connect(self._run_tta_inprocess)
        infer_layout.addWidget(self.btn_run_tta)

        # TTA 融合模式选择
        fusion_layout = QHBoxLayout()
        fusion_label = QLabel('融合模式:')
        self.fusion_combo = QComboBox()
        self.fusion_combo.addItems(['简单均值', '不确定性加权'])
        self.fusion_combo.setToolTip('简单均值: 各增强概率直接平均\n不确定性加权: 以像素熵倒数为权重融合')
        fusion_layout.addWidget(fusion_label)
        fusion_layout.addWidget(self.fusion_combo)
        infer_layout.addLayout(fusion_layout)

        # Epsilon 控件
        eps_layout = QHBoxLayout()
        eps_label = QLabel('ε 值:')
        self.eps_spinbox = QDoubleSpinBox()
        self.eps_spinbox.setDecimals(10)
        self.eps_spinbox.setRange(1e-12, 1e-2)
        self.eps_spinbox.setValue(1e-8)
        self.eps_spinbox.setSingleStep(1e-9)
        self.eps_spinbox.setToolTip('不确定性加权融合的防除零平滑项')
        eps_layout.addWidget(eps_label)
        eps_layout.addWidget(self.eps_spinbox)
        infer_layout.addLayout(eps_layout)

        self.inference_progress = QProgressBar()
        self.inference_progress.setRange(0, 0)  # 不确定进度
        self.inference_progress.setVisible(False)
        infer_layout.addWidget(self.inference_progress)
        self.inference_status_label = QLabel('')
        self.inference_status_label.setWordWrap(True)
        infer_layout.addWidget(self.inference_status_label)
        left_panel.addWidget(infer_group)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(200)

        # ========== 中央面板 ==========
        center_panel = QVBoxLayout()
        center_panel.setSpacing(6)

        # 显示模式选择
        mode_layout = QHBoxLayout()
        mode_label = QLabel('显示模式:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['三视图并排', '叠加显示（预测）', '叠加显示（GT）', '叠加显示（TTA预测）', '对比：预测 vs TTA',
                                   '叠加显示（UW预测）', '熵热力图', '熵图叠加CT'])
        self.mode_combo.currentIndexChanged.connect(self._update_display)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()

        # 截图按钮
        self.btn_screenshot = QPushButton('截图保存')
        self.btn_screenshot.clicked.connect(self._save_screenshot)
        mode_layout.addWidget(self.btn_screenshot)
        center_panel.addLayout(mode_layout)

        # 图像显示区域
        self.image_container = QWidget()
        self.image_layout = QHBoxLayout(self.image_container)
        self.image_layout.setSpacing(6)

        self.label_img = QLabel('原始 CT')
        self.label_pred = QLabel('预测结果')
        self.label_gt = QLabel('GT 标注')
        for lbl in [self.label_img, self.label_pred, self.label_gt]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(280, 280)
            lbl.setStyleSheet('background-color: #1a1a2e; border: 2px solid #c0392b; border-radius: 4px;')
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_layout.addWidget(self.label_img)
        self.image_layout.addWidget(self.label_pred)
        self.image_layout.addWidget(self.label_gt)

        # 熵值色标条
        self.colorbar_label = QLabel()
        self.colorbar_label.setFixedWidth(80)
        self.colorbar_label.setMinimumHeight(200)
        self.colorbar_label.setVisible(False)
        self.image_layout.addWidget(self.colorbar_label)

        center_panel.addWidget(self.image_container, stretch=1)

        # 图像标题
        title_layout = QHBoxLayout()
        for txt in ['原始 CT 图像', '模型预测结果', 'Ground Truth']:
            t = QLabel(txt)
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet('font-weight: bold; color: #c0392b; font-size: 13px;')
            title_layout.addWidget(t)
        self.title_labels = title_layout
        center_panel.addLayout(title_layout)

        # 切片滑动条
        slice_layout = QHBoxLayout()
        slice_label = QLabel('切片:')
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_info = QLabel('0 / 0')
        self.slice_info.setMinimumWidth(80)
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_slider, stretch=1)
        slice_layout.addWidget(self.slice_info)
        center_panel.addLayout(slice_layout)

        # 窗宽窗位
        ww_layout = QHBoxLayout()
        ww_label = QLabel('窗宽:')
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setMinimum(1)
        self.ww_slider.setMaximum(100)
        self.ww_slider.setValue(int(self.window_width * 100))
        self.ww_slider.valueChanged.connect(self._on_ww_changed)
        self.ww_value = QLabel(f'{self.window_width:.2f}')
        self.ww_value.setMinimumWidth(50)

        wl_label = QLabel('窗位:')
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setMinimum(0)
        self.wl_slider.setMaximum(100)
        self.wl_slider.setValue(int(self.window_center * 100))
        self.wl_slider.valueChanged.connect(self._on_wl_changed)
        self.wl_value = QLabel(f'{self.window_center:.2f}')
        self.wl_value.setMinimumWidth(50)

        ww_layout.addWidget(ww_label)
        ww_layout.addWidget(self.ww_slider, stretch=1)
        ww_layout.addWidget(self.ww_value)
        ww_layout.addSpacing(20)
        ww_layout.addWidget(wl_label)
        ww_layout.addWidget(self.wl_slider, stretch=1)
        ww_layout.addWidget(self.wl_value)
        center_panel.addLayout(ww_layout)

        # 叠加透明度
        alpha_layout = QHBoxLayout()
        alpha_label = QLabel('叠加透明度:')
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.overlay_alpha * 100))
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.alpha_value = QLabel(f'{self.overlay_alpha:.2f}')
        self.alpha_value.setMinimumWidth(50)
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(self.alpha_slider, stretch=1)
        alpha_layout.addWidget(self.alpha_value)
        center_panel.addLayout(alpha_layout)

        center_widget = QWidget()
        center_widget.setLayout(center_panel)

        # ========== 右侧面板 ==========
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        # 器官图例
        legend_group = QGroupBox('器官图例')
        legend_layout = QVBoxLayout(legend_group)
        for idx in range(1, 9):
            row = QHBoxLayout()
            color_block = QLabel()
            color_block.setFixedSize(20, 20)
            r, g, b, a = ORGAN_COLORS[idx]
            color_block.setStyleSheet(f'background-color: rgba({r},{g},{b},{a}); border: 1px solid #888; border-radius: 3px;')
            name_label = QLabel(f'{idx}. {ORGAN_NAMES[idx]}')
            row.addWidget(color_block)
            row.addWidget(name_label)
            row.addStretch()
            legend_layout.addLayout(row)
        right_panel.addWidget(legend_group)

        # 元信息
        info_group = QGroupBox('图像信息')
        info_layout = QVBoxLayout(info_group)
        self.info_size = QLabel('尺寸: -')
        self.info_spacing = QLabel('体素间距: -')
        self.info_slices = QLabel('切片数: -')
        self.info_case = QLabel('当前病例: -')
        for w in [self.info_case, self.info_size, self.info_spacing, self.info_slices]:
            w.setWordWrap(True)
            info_layout.addWidget(w)
        right_panel.addWidget(info_group)

        # 评估指标
        metric_group = QGroupBox('评估指标')
        metric_layout = QVBoxLayout(metric_group)

        # 当前病例指标
        self.case_metric_label = QLabel('当前病例:')
        self.case_metric_label.setStyleSheet('font-weight: bold;')
        metric_layout.addWidget(self.case_metric_label)
        self.case_dice_label = QLabel('Dice: -')
        self.case_iou_label = QLabel('IoU: -')
        self.case_hd95_label = QLabel('HD95: -')
        self.case_iou_label.setWordWrap(True)
        metric_layout.addWidget(self.case_dice_label)
        metric_layout.addWidget(self.case_iou_label)
        metric_layout.addWidget(self.case_hd95_label)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('color: #c0392b;')
        metric_layout.addWidget(line)

        # 各类别指标表格
        self.metric_table = QTableWidget()
        self.metric_table.setColumnCount(4)
        self.metric_table.setHorizontalHeaderLabels(['器官', 'Dice', 'IoU', 'HD95'])
        self.metric_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metric_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metric_table.setAlternatingRowColors(True)
        self.metric_table.verticalHeader().setVisible(False)
        metric_layout.addWidget(self.metric_table)

        # 总体指标
        self.overall_label = QLabel('总体: -')
        self.overall_label.setWordWrap(True)
        self.overall_label.setStyleSheet('font-weight: bold; color: #c0392b;')
        metric_layout.addWidget(self.overall_label)
        right_panel.addWidget(metric_group, stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(280)

        # ========== 组装 ==========
        main_layout.addWidget(left_widget)
        main_layout.addWidget(center_widget, stretch=1)
        main_layout.addWidget(right_widget)

        # 状态栏
        self.statusBar().showMessage('就绪')

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fdf2f2;
            }
            QWidget {
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
                font-size: 12px;
                color: #2c2c2c;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #c0392b;
                border: 2px solid #e74c3c;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                background-color: #fdf2f2;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #e8b4b4;
                border-radius: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #f5dede;
            }
            QListWidget::item:selected {
                background-color: #e74c3c;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #fadbd8;
            }
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #d5a6a6;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e8b4b4;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e74c3c;
                border: 2px solid #c0392b;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #e74c3c;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e8b4b4;
                border-radius: 4px;
                gridline-color: #f5dede;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #e8b4b4;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QProgressBar {
                border: 1px solid #e8b4b4;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #e74c3c;
                color: white;
                font-size: 12px;
            }
        """)

    # ========== 数据加载 ==========

    def _load_case_list(self, folder=None):
        if folder is None:
            folder = PREDICTIONS_DIR
        self.predictions_dir = folder
        self.case_list.clear()
        if not os.path.exists(folder):
            return
        files = os.listdir(folder)
        cases = sorted(set(
            f.replace('_pred_tta_uw.nii.gz', '').replace('_pred_tta.nii.gz', '').replace('_pred.nii.gz', '').replace('_img.nii.gz', '').replace('_gt.nii.gz', '').replace('_entropy.nii.gz', '')
            for f in files if f.endswith('.nii.gz')
        ))
        for c in cases:
            self.case_list.addItem(c)
        if cases:
            self.statusBar().showMessage(f'已加载 {len(cases)} 个病例')

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择预测结果文件夹', '.')
        if folder:
            self._load_case_list(folder)

    def _on_case_selected(self, current, previous):
        if current is None:
            return
        case_name = current.text()
        self.current_case = case_name
        self._load_nii_data(case_name)

    def _load_nii_data(self, case_name):
        folder = self.predictions_dir
        img_path = os.path.join(folder, f'{case_name}_img.nii.gz')
        pred_path = os.path.join(folder, f'{case_name}_pred.nii.gz')
        gt_path = os.path.join(folder, f'{case_name}_gt.nii.gz')
        pred_tta_path = os.path.join(folder, f'{case_name}_pred_tta.nii.gz')
        pred_tta_uw_path = os.path.join(folder, f'{case_name}_pred_tta_uw.nii.gz')

        try:
            img_nii = nib.load(img_path)
            pred_nii = nib.load(pred_path)
            gt_nii = nib.load(gt_path)

            self.img_data = img_nii.get_fdata()
            self.pred_data = pred_nii.get_fdata()
            self.gt_data = gt_nii.get_fdata()

            # 尝试加载 TTA 预测
            if os.path.exists(pred_tta_path):
                self.pred_tta_data = nib.load(pred_tta_path).get_fdata()
            else:
                self.pred_tta_data = None

            # 尝试加载 UW TTA 预测
            if os.path.exists(pred_tta_uw_path):
                self.pred_tta_uw_data = nib.load(pred_tta_uw_path).get_fdata()
            else:
                self.pred_tta_uw_data = None

            # 熵数据：尝试从磁盘加载
            entropy_path = os.path.join(folder, f'{case_name}_entropy.nii.gz')
            if os.path.exists(entropy_path):
                self.entropy_data = nib.load(entropy_path).get_fdata()
            else:
                self.entropy_data = None

            # 切片维度为第3维 (axis=2)
            self.total_slices = self.img_data.shape[2]
            self.slice_slider.setMaximum(self.total_slices - 1)
            self.slice_slider.setValue(self.total_slices // 2)

            # 更新元信息
            spacing = img_nii.header.get_zooms()
            shape = self.img_data.shape
            self.info_case.setText(f'当前病例: {case_name}')
            self.info_size.setText(f'尺寸: {shape[0]}×{shape[1]}×{shape[2]}')
            self.info_spacing.setText(f'体素间距: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f}')
            self.info_slices.setText(f'切片数: {self.total_slices}')

            # 更新当前病例指标
            self._update_case_overlap_metrics(case_name)
            self._update_case_metric(case_name)
            self._refresh_metric_table()

            self._update_display()
            self.statusBar().showMessage(f'已加载病例: {case_name}')

        except Exception as e:
            QMessageBox.warning(self, '加载失败', f'无法加载病例 {case_name}:\n{str(e)}')

    # ========== 图像渲染 ==========

    def _apply_window(self, img_slice):
        """应用窗宽窗位"""
        vmin = self.window_center - self.window_width / 2
        vmax = self.window_center + self.window_width / 2
        img = np.clip(img_slice, vmin, vmax)
        img = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        return img

    def _slice_to_qpixmap(self, img_slice, size=None):
        """将灰度切片转为 QPixmap"""
        img = self._apply_window(img_slice)
        h, w = img.shape
        qimg = QImage(img.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        if size:
            pix = pix.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    def _mask_to_color(self, mask_slice):
        """将分割 mask 转为 RGBA 彩色图"""
        h, w = mask_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for label_val, color in ORGAN_COLORS.items():
            if label_val == 0:
                continue
            region = mask_slice == label_val
            rgba[region] = color
        return rgba

    def _overlay_mask_on_ct(self, img_slice, mask_slice):
        """将分割 mask 叠加到 CT 图上"""
        ct_gray = self._apply_window(img_slice)
        h, w = ct_gray.shape
        # 创建 RGB CT 底图
        rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float64)
        # 彩色 mask
        mask_rgba = self._mask_to_color(mask_slice).astype(np.float64)
        mask_rgb = mask_rgba[:, :, :3]
        mask_a = mask_rgba[:, :, 3] / 255.0 * self.overlay_alpha
        # 混合
        for c in range(3):
            rgb[:, :, c] = rgb[:, :, c] * (1 - mask_a) + mask_rgb[:, :, c] * mask_a
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _mask_to_qpixmap(self, mask_slice):
        """将分割 mask 转为彩色 QPixmap"""
        rgba = self._mask_to_color(mask_slice)
        h, w = mask_slice.shape
        # 黑色背景 + 彩色 mask
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask_a = rgba[:, :, 3].astype(np.float64) / 255.0
        for c in range(3):
            rgb[:, :, c] = (rgba[:, :, c].astype(np.float64) * mask_a).astype(np.uint8)
        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _entropy_to_qpixmap(self, entropy_slice, size=None):
        """将熵图转为伪彩色热力图 QPixmap（蓝→绿→黄→红）"""
        h, w = entropy_slice.shape
        # 归一化到 [0, 1]
        emin, emax = entropy_slice.min(), entropy_slice.max()
        if emax - emin > 1e-10:
            norm = (entropy_slice - emin) / (emax - emin)
        else:
            norm = np.zeros_like(entropy_slice)

        # 手动实现 jet-like 色图：蓝→青→绿→黄→红
        r = np.clip(1.5 - np.abs(norm * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(norm * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(norm * 4 - 1), 0, 1)

        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if size:
            pix = pix.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    def _overlay_entropy_on_ct(self, img_slice, entropy_slice, size=None):
        """将熵热力图半透明叠加到 CT 灰度图上"""
        ct_gray = self._apply_window(img_slice)
        h, w = ct_gray.shape
        # CT 底图 RGB
        ct_rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float64)

        # 熵归一化
        emin, emax = entropy_slice.min(), entropy_slice.max()
        if emax - emin > 1e-10:
            norm = (entropy_slice - emin) / (emax - emin)
        else:
            norm = np.zeros_like(entropy_slice)

        # jet-like 色图
        r = np.clip(1.5 - np.abs(norm * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(norm * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(norm * 4 - 1), 0, 1)
        heat_rgb = np.stack([r, g, b], axis=-1) * 255.0

        # 用熵值本身作为叠加强度：低熵区域几乎透明，高熵区域更明显
        alpha = (norm * self.overlay_alpha)[:, :, np.newaxis]
        blended = ct_rgb * (1 - alpha) + heat_rgb * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        qimg = QImage(blended.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if size:
            pix = pix.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    def _draw_colorbar(self, emin, emax):
        """绘制垂直色标条，标注熵值范围"""
        bar_w = 80
        bar_h = max(self.colorbar_label.height(), 200)
        gradient_w = 20
        margin_top = 20
        margin_bottom = 20
        gradient_h = bar_h - margin_top - margin_bottom

        pixmap = QPixmap(bar_w, bar_h)
        pixmap.fill(QColor(26, 26, 46))  # 背景色与图像区域一致

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制渐变色条（从上到下：高熵红色→低熵蓝色）
        for i in range(gradient_h):
            norm = 1.0 - i / max(gradient_h - 1, 1)  # 顶部=1（高熵），底部=0（低熵）
            r = max(0.0, min(1.0, 1.5 - abs(norm * 4 - 3)))
            g = max(0.0, min(1.0, 1.5 - abs(norm * 4 - 2)))
            b = max(0.0, min(1.0, 1.5 - abs(norm * 4 - 1)))
            painter.setPen(QColor(int(r * 255), int(g * 255), int(b * 255)))
            painter.drawLine(10, margin_top + i, 10 + gradient_w, margin_top + i)

        # 绘制边框
        painter.setPen(QColor(200, 200, 200))
        painter.drawRect(10, margin_top, gradient_w, gradient_h)

        # 标注数值
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(10 + gradient_w + 4, margin_top + 10, f'{emax:.3f}')
        mid_val = (emin + emax) / 2
        painter.drawText(10 + gradient_w + 4, margin_top + gradient_h // 2 + 4, f'{mid_val:.3f}')
        painter.drawText(10 + gradient_w + 4, margin_top + gradient_h - 2, f'{emin:.3f}')

        # 标题
        painter.drawText(5, margin_top - 5, '熵值')

        painter.end()
        self.colorbar_label.setPixmap(pixmap)

    def _update_display(self):
        if self.img_data is None:
            return

        idx = self.current_slice
        img_slice = self.img_data[:, :, idx]
        pred_slice = self.pred_data[:, :, idx]
        gt_slice = self.gt_data[:, :, idx]

        mode = self.mode_combo.currentIndex()

        # 色标条：仅在熵图模式（6、7）时显示
        if mode in (6, 7) and self.entropy_data is not None:
            entropy_slice = self.entropy_data[:, :, idx]
            self.colorbar_label.setVisible(True)
            self._draw_colorbar(float(entropy_slice.min()), float(entropy_slice.max()))
        else:
            self.colorbar_label.setVisible(False)

        if mode == 0:
            # 三视图并排
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(True)

            # 获取显示区域大小
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_pred = self._overlay_mask_on_ct(img_slice, pred_slice)
            pix_pred = pix_pred.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix_gt = self._overlay_mask_on_ct(img_slice, gt_slice)
            pix_gt = pix_gt.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_pred)
            self.label_gt.setPixmap(pix_gt)

        elif mode == 1:
            # 叠加显示（预测）
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(False)

            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_overlay = self._overlay_mask_on_ct(img_slice, pred_slice)
            pix_overlay = pix_overlay.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_overlay)

        elif mode == 2:
            # 叠加显示（GT）
            self.label_img.setVisible(True)
            self.label_pred.setVisible(False)
            self.label_gt.setVisible(True)

            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_overlay = self._overlay_mask_on_ct(img_slice, gt_slice)
            pix_overlay = pix_overlay.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_gt.setPixmap(pix_overlay)

        elif mode == 3:
            # 叠加显示（TTA预测）
            if self.pred_tta_data is None:
                self.statusBar().showMessage('当前病例没有 TTA 预测结果')
                return
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(False)

            tta_slice = self.pred_tta_data[:, :, idx]
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_overlay = self._overlay_mask_on_ct(img_slice, tta_slice)
            pix_overlay = pix_overlay.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_overlay)

        elif mode == 4:
            # 对比：预测 vs TTA
            if self.pred_tta_data is None:
                self.statusBar().showMessage('当前病例没有 TTA 预测结果')
                return
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(True)

            tta_slice = self.pred_tta_data[:, :, idx]
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_pred = self._overlay_mask_on_ct(img_slice, pred_slice)
            pix_pred = pix_pred.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix_tta = self._overlay_mask_on_ct(img_slice, tta_slice)
            pix_tta = pix_tta.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_pred)
            self.label_gt.setPixmap(pix_tta)

        elif mode == 5:
            # 叠加显示（UW预测）
            if self.pred_tta_uw_data is None:
                self.statusBar().showMessage('当前病例没有不确定性加权 TTA 预测结果')
                return
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(False)

            uw_slice = self.pred_tta_uw_data[:, :, idx]
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_overlay = self._overlay_mask_on_ct(img_slice, uw_slice)
            pix_overlay = pix_overlay.scaled(sz[0], sz[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_overlay)

        elif mode == 6:
            # 熵热力图
            if self.entropy_data is None:
                self.statusBar().showMessage('当前病例没有熵图数据，请先运行不确定性加权 TTA 推理')
                return
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(False)

            entropy_slice = self.entropy_data[:, :, idx]
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_entropy = self._entropy_to_qpixmap(entropy_slice, sz)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_entropy)

        elif mode == 7:
            # 熵图叠加CT
            if self.entropy_data is None:
                self.statusBar().showMessage('当前病例没有熵图数据，请先运行 TTA 推理')
                return
            self.label_img.setVisible(True)
            self.label_pred.setVisible(True)
            self.label_gt.setVisible(False)

            entropy_slice = self.entropy_data[:, :, idx]
            lbl_size = self.label_img.size()
            sz = (lbl_size.width() - 4, lbl_size.height() - 4)

            pix_img = self._slice_to_qpixmap(img_slice, sz)
            pix_overlay = self._overlay_entropy_on_ct(img_slice, entropy_slice, sz)

            self.label_img.setPixmap(pix_img)
            self.label_pred.setPixmap(pix_overlay)

    # ========== 控件事件 ==========

    def _on_slice_changed(self, value):
        self.current_slice = value
        self.slice_info.setText(f'{value + 1} / {self.total_slices}')
        self._update_display()

    def _on_ww_changed(self, value):
        self.window_width = value / 100.0
        self.ww_value.setText(f'{self.window_width:.2f}')
        self._update_display()

    def _on_wl_changed(self, value):
        self.window_center = value / 100.0
        self.wl_value.setText(f'{self.window_center:.2f}')
        self._update_display()

    def _on_alpha_changed(self, value):
        self.overlay_alpha = value / 100.0
        self.alpha_value.setText(f'{self.overlay_alpha:.2f}')
        self._update_display()

    # ========== 评估指标 ==========

    @staticmethod
    def _dice_to_iou(dice):
        if dice is None or not np.isfinite(dice):
            return np.nan
        denominator = 2.0 - dice
        if denominator <= 0:
            return np.nan
        return dice / denominator

    @staticmethod
    def _format_metric(value, precision=4):
        if value is None or not np.isfinite(value):
            return '无数据'
        return f'{value:.{precision}f}'

    def _compute_overlap_metrics(self, pred, gt):
        """计算多类别分割的 Dice 和 IoU，忽略背景类别 0。"""
        if pred is None or gt is None or pred.shape != gt.shape:
            return np.nan, np.nan, {}

        pred_labels = np.rint(pred).astype(np.int16)
        gt_labels = np.rint(gt).astype(np.int16)
        class_metrics = {}

        for cls_id in range(1, max(ORGAN_NAMES.keys()) + 1):
            pred_region = pred_labels == cls_id
            gt_region = gt_labels == cls_id

            intersection = np.logical_and(pred_region, gt_region).sum(dtype=np.float64)
            pred_sum = pred_region.sum(dtype=np.float64)
            gt_sum = gt_region.sum(dtype=np.float64)
            union = pred_sum + gt_sum - intersection

            if union == 0:
                dice = np.nan
                iou = np.nan
            else:
                dice = 2.0 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else np.nan
                iou = intersection / union

            class_metrics[cls_id] = {'dice': dice, 'iou': iou}

        valid_dice = [m['dice'] for m in class_metrics.values() if np.isfinite(m['dice'])]
        valid_iou = [m['iou'] for m in class_metrics.values() if np.isfinite(m['iou'])]
        mean_dice = float(np.mean(valid_dice)) if valid_dice else np.nan
        mean_iou = float(np.mean(valid_iou)) if valid_iou else np.nan
        return mean_dice, mean_iou, class_metrics

    def _update_case_overlap_metrics(self, case_name):
        mean_dice, mean_iou, class_metrics = self._compute_overlap_metrics(self.pred_data, self.gt_data)
        self.current_class_overlap_metrics = class_metrics

        if case_name:
            case_metric = self.case_metrics.setdefault(case_name, {})
            if np.isfinite(mean_dice):
                case_metric['dice'] = mean_dice
            if np.isfinite(mean_iou):
                case_metric['iou'] = mean_iou

            if self.pred_tta_data is not None:
                _, tta_iou, _ = self._compute_overlap_metrics(self.pred_tta_data, self.gt_data)
                if np.isfinite(tta_iou):
                    case_metric['tta_iou'] = tta_iou

            if self.pred_tta_uw_data is not None:
                _, tta_uw_iou, _ = self._compute_overlap_metrics(self.pred_tta_uw_data, self.gt_data)
                if np.isfinite(tta_uw_iou):
                    case_metric['tta_uw_iou'] = tta_uw_iou

    def _refresh_metric_table(self):
        class_ids = sorted(set(self.class_metrics.keys()) | set(self.current_class_overlap_metrics.keys()))
        self.metric_table.setRowCount(len(class_ids))
        for row, cls_id in enumerate(class_ids):
            data = self.class_metrics.get(cls_id, {})
            current_metrics = self.current_class_overlap_metrics.get(cls_id, {})
            organ_name = ORGAN_NAMES.get(cls_id, data.get('name', f'Class {cls_id}'))
            dice = current_metrics.get('dice', data.get('dice', np.nan))
            iou = current_metrics.get('iou', data.get('iou', np.nan))
            hd95 = data.get('hd95', np.nan)

            self.metric_table.setItem(row, 0, QTableWidgetItem(organ_name))
            self.metric_table.setItem(row, 1, QTableWidgetItem(self._format_metric(dice)))
            self.metric_table.setItem(row, 2, QTableWidgetItem(self._format_metric(iou)))
            self.metric_table.setItem(row, 3, QTableWidgetItem(self._format_metric(hd95, precision=2)))

    def _load_log_metrics(self):
        self.case_metrics = {}
        self.class_metrics = {}
        self.overall_metrics = None
        self.current_class_overlap_metrics = {}

        if not os.path.exists(LOG_FILE):
            return

        try:
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return

        # 解析各病例指标（取最后一次运行的结果）
        case_pattern = re.compile(r'idx \d+ case (\w+) mean_dice ([\d.]+) mean_hd95 ([\d.]+)')
        class_pattern = re.compile(r'Mean class (\d+) name (\w[\w\s]*?) mean_dice ([\d.]+) mean_hd95 ([\d.]+)')
        overall_pattern = re.compile(r'Testing performance.*mean_dice : ([\d.]+) mean_hd95 : ([\d.]+)')

        for line in lines:
            m = case_pattern.search(line)
            if m:
                case_name = m.group(1)
                dice = float(m.group(2))
                hd95 = float(m.group(3))
                self.case_metrics[case_name] = {'dice': dice, 'iou': self._dice_to_iou(dice), 'hd95': hd95}

            m = class_pattern.search(line)
            if m:
                cls_id = int(m.group(1))
                cls_name = m.group(2).strip()
                dice = float(m.group(3))
                hd95 = float(m.group(4))
                self.class_metrics[cls_id] = {'name': cls_name, 'dice': dice, 'iou': self._dice_to_iou(dice), 'hd95': hd95}

            m = overall_pattern.search(line)
            if m:
                self.overall_metrics = {
                    'dice': float(m.group(1)),
                    'iou': self._dice_to_iou(float(m.group(1))),
                    'hd95': float(m.group(2))
                }

        # 填充各类别表格
        self._refresh_metric_table()

        if self.overall_metrics:
            self.overall_label.setText(
                f"总体  Dice: {self._format_metric(self.overall_metrics.get('dice'))}\n"
                f"IoU: {self._format_metric(self.overall_metrics.get('iou'))}  "
                f"HD95: {self._format_metric(self.overall_metrics.get('hd95'), precision=2)}"
            )

    def _update_case_metric(self, case_name):
        if case_name in self.case_metrics:
            data = self.case_metrics[case_name]
            self.case_metric_label.setText(f'当前病例: {case_name}')
            self.case_dice_label.setText(f'Dice: {self._format_metric(data.get("dice"))}')

            iou_text = f'IoU: {self._format_metric(data.get("iou"))}'
            tta_parts = []
            if 'tta_iou' in data:
                tta_parts.append(f'TTA {self._format_metric(data.get("tta_iou"))}')
            if 'tta_uw_iou' in data:
                tta_parts.append(f'UW {self._format_metric(data.get("tta_uw_iou"))}')
            if tta_parts:
                iou_text = f'{iou_text}  ({" / ".join(tta_parts)})'
            self.case_iou_label.setText(iou_text)

            self.case_hd95_label.setText(f'HD95: {self._format_metric(data.get("hd95"), precision=2)}')
        else:
            self.case_metric_label.setText(f'当前病例: {case_name}')
            self.case_dice_label.setText('Dice: 无数据')
            self.case_iou_label.setText('IoU: 无数据')
            self.case_hd95_label.setText('HD95: 无数据')

    # ========== 进程内 TTA 推理 ==========

    def _load_model_inprocess(self):
        """在 UI 进程中加载 SAMed 模型，支持 GPU/CPU 自动检测"""
        if self.tta_model is not None:
            return self.tta_model

        try:
            import torch
            from segment_anything import sam_model_registry
            from importlib import import_module

            # 自动检测设备
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
                self.inference_status_label.setText(f'使用 GPU: {device_name}')
            else:
                device = torch.device('cpu')
                self.inference_status_label.setText('使用 CPU（无 GPU 可用）')

            QApplication.processEvents()

            # 模型参数（与 test.py 一致）
            img_size = 512
            num_classes = 8
            vit_name = 'vit_b'
            rank = 4
            ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'sam_vit_b_01ec64.pth')
            lora_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'epoch_159.pth')

            sam, img_embedding_size = sam_model_registry[vit_name](
                image_size=img_size, num_classes=num_classes,
                checkpoint=ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]
            )

            pkg = import_module('sam_lora_image_encoder')
            net = pkg.LoRA_Sam(sam, rank).to(device)
            net.load_lora_parameters(lora_ckpt)
            net.eval()

            self.tta_model = net
            self.inference_status_label.setText('模型加载完成')
            return net

        except Exception as e:
            QMessageBox.critical(self, '模型加载失败', f'无法加载模型：\n{str(e)}')
            return None

    def _run_tta_inprocess(self):
        """在 UI 进程中对当前病例执行 TTA 推理"""
        if self.img_data is None:
            QMessageBox.information(self, '提示', '请先选择一个病例。')
            return

        fusion_mode = 'uncertainty' if self.fusion_combo.currentIndex() == 1 else 'mean'
        epsilon = self.eps_spinbox.value()
        fusion_label = self.fusion_combo.currentText()

        reply = QMessageBox.question(
            self, '确认',
            f'确定要对当前病例 {self.current_case} 执行 TTA 推理吗？\n'
            f'融合模式: {fusion_label}\n'
            f'ε = {epsilon:.2e}\n'
            '这将在 UI 进程中加载模型并推理，可能需要几分钟。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.btn_run_tta.setEnabled(False)
        self.inference_progress.setVisible(True)
        self.statusBar().showMessage('正在加载模型...')
        self.inference_status_label.setText('正在加载模型...')
        QApplication.processEvents()

        # 在后台线程执行推理
        def tta_worker():
            try:
                import torch
                from scipy.ndimage import zoom
                from einops import repeat
                import torch.nn.functional as F
                import SimpleITK as sitk

                net = self._load_model_inprocess()
                if net is None:
                    return

                device = next(net.parameters()).device
                img_data = self.img_data  # shape: (H, W, N_slices)
                num_slices = img_data.shape[2]
                prediction = np.zeros(img_data.shape, dtype=np.float32)
                entropy_maps = np.zeros(img_data.shape, dtype=np.float32)

                input_size = [224, 224]
                patch_size = [512, 512]

                for ind in range(num_slices):
                    slice_np = img_data[:, :, ind]
                    x, y = slice_np.shape[0], slice_np.shape[1]

                    augmentations = [
                        ('original', None),
                        ('hflip', None),
                        ('scale_0.9', 0.9),
                        ('scale_1.1', 1.1),
                    ]

                    all_probs = []
                    for aug_name, aug_param in augmentations:
                        if aug_name == 'original':
                            aug_slice = slice_np.copy()
                        elif aug_name == 'hflip':
                            aug_slice = np.flip(slice_np, axis=1).copy()
                        elif aug_name.startswith('scale'):
                            aug_slice = zoom(slice_np, aug_param, order=3)
                        else:
                            aug_slice = slice_np.copy()

                        # 预处理
                        ax, ay = aug_slice.shape[0], aug_slice.shape[1]
                        if ax != input_size[0] or ay != input_size[1]:
                            aug_slice = zoom(aug_slice, (input_size[0] / ax, input_size[1] / ay), order=3)
                        nx, ny = aug_slice.shape[0], aug_slice.shape[1]
                        if nx != patch_size[0] or ny != patch_size[1]:
                            aug_slice = zoom(aug_slice, (patch_size[0] / nx, patch_size[1] / ny), order=3)

                        inputs = torch.from_numpy(aug_slice).unsqueeze(0).unsqueeze(0).float().to(device)
                        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)

                        with torch.no_grad():
                            outputs = net(inputs, True, patch_size[0])
                            probs = torch.softmax(outputs['masks'], dim=1)

                        # 逆变换
                        if aug_name == 'hflip':
                            probs = torch.flip(probs, dims=[3])
                        elif aug_name.startswith('scale'):
                            probs = F.interpolate(probs, size=(patch_size[0], patch_size[1]),
                                                  mode='bilinear', align_corners=False)

                        all_probs.append(probs)

                    if fusion_mode == 'uncertainty':
                        # 不确定性加权融合
                        weighted_sum = torch.zeros_like(all_probs[0])
                        weight_sum = torch.zeros(1, 1, all_probs[0].shape[2], all_probs[0].shape[3], device=device)

                        for probs in all_probs:
                            entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1, keepdim=True)
                            w = 1.0 / (entropy + epsilon)
                            weighted_sum += w * probs
                            weight_sum += w

                        prob_fused = weighted_sum / weight_sum
                    else:
                        # 简单均值融合
                        prob_fused = torch.stack(all_probs, dim=0).mean(dim=0)

                    # 无论融合模式，都计算融合后概率的熵图
                    entropy_map = -torch.sum(prob_fused * torch.log(prob_fused + epsilon), dim=1).squeeze(0)
                    entropy_map = entropy_map.cpu().detach().numpy()

                    out = torch.argmax(prob_fused, dim=1).squeeze(0).cpu().detach().numpy()
                    out_h, out_w = out.shape
                    if x != out_h or y != out_w:
                        pred = zoom(out, (x / out_h, y / out_w), order=0)
                    else:
                        pred = out
                    prediction[:, :, ind] = pred

                    if x != entropy_map.shape[0] or y != entropy_map.shape[1]:
                        entropy_map = zoom(entropy_map, (x / entropy_map.shape[0], y / entropy_map.shape[1]), order=1)
                    entropy_maps[:, :, ind] = entropy_map

                    # 更新进度
                    progress_pct = int((ind + 1) / num_slices * 100)
                    self._tta_progress_msg = f'TTA 推理中 ({fusion_label}): {ind + 1}/{num_slices} ({progress_pct}%)'

                # 保存结果
                save_dir = self.predictions_dir
                os.makedirs(save_dir, exist_ok=True)
                prd_itk = sitk.GetImageFromArray(prediction.transpose(2, 1, 0).astype(np.float32))
                prd_itk.SetSpacing((1, 1, 1))
                if fusion_mode == 'uncertainty':
                    save_path = os.path.join(save_dir, f'{self.current_case}_pred_tta_uw.nii.gz')
                else:
                    save_path = os.path.join(save_dir, f'{self.current_case}_pred_tta.nii.gz')
                sitk.WriteImage(prd_itk, save_path)

                # 保存熵图到磁盘
                entropy_save_path = os.path.join(save_dir, f'{self.current_case}_entropy.nii.gz')
                ent_itk = sitk.GetImageFromArray(entropy_maps.transpose(2, 1, 0).astype(np.float32))
                ent_itk.SetSpacing((1, 1, 1))
                sitk.WriteImage(ent_itk, entropy_save_path)

                self._tta_entropy_data = entropy_maps
                self._tta_result = ('success', save_path)
            except Exception as e:
                self._tta_result = ('error', str(e))

        self._tta_result = None
        self._tta_progress_msg = 'TTA 推理中...'
        self._tta_entropy_data = None
        thread = threading.Thread(target=tta_worker, daemon=True)
        thread.start()

        # 用 QTimer 轮询线程状态
        def check_tta_done():
            self.inference_status_label.setText(self._tta_progress_msg)
            if self._tta_result is not None:
                self.tta_timer.stop()
                self.btn_run_tta.setEnabled(True)
                self.inference_progress.setVisible(False)

                status, info = self._tta_result
                if status == 'success':
                    self.inference_status_label.setText('TTA 推理完成！')
                    self.statusBar().showMessage(f'TTA 结果已保存: {info}')
                    # 重新加载当前病例以加载 TTA 预测和熵图
                    if self.current_case:
                        self._load_nii_data(self.current_case)
                    QMessageBox.information(self, '完成',
                                            f'TTA 推理完成！\n融合模式: {fusion_label}\n结果已保存至：\n{info}')
                else:
                    self.inference_status_label.setText('TTA 推理失败')
                    self.statusBar().showMessage('TTA 推理失败')
                    QMessageBox.warning(self, '失败', f'TTA 推理失败：\n{info}')

        self.tta_timer = QTimer(self)
        self.tta_timer.timeout.connect(check_tta_done)
        self.tta_timer.start(500)

    # ========== 截图 ==========

    def _save_screenshot(self):
        if self.img_data is None:
            QMessageBox.information(self, '提示', '请先加载病例数据。')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '保存截图', f'{self.current_case}_slice{self.current_slice}.png',
            'PNG 图片 (*.png);;JPEG 图片 (*.jpg)'
        )
        if path:
            self.image_container.grab().save(path)
            self.statusBar().showMessage(f'截图已保存: {path}')

    # ========== 窗口事件 ==========

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))
    window = NiiViewer()
    window.show()
    sys.exit(app.exec_())
