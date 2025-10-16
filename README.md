# 🚁 Catch Anything V3：The multi-strategy UAV target tracking system assisted by Kalman filtering

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
> 🎯 *面向飞行视频中的动态目标追踪，适用于多类无人机系统*

Catch Anything V3 是一个融合多策略的目标追踪系统，专为处理无人机拍摄视频中的复杂跟踪任务而设计。系统通过光流法、模板匹配、卡尔曼滤波辅助、颜色建模与自适应机制结合，在目标遮挡、快速移动、遮挡或消失等情况下依然实现高鲁棒性跟踪。


---
## 🙌 支持一下

- 如果这个项目对你有帮助，欢迎 ⭐️ Star！
---


## 📢 Latest Updates

- **Oct-10-25**： **Catch Anything V3 is released.**。
- **Aug-8-25**: Catch Anything V2 **[Code]** has been uploaded.- **[Test video download link](https://hkustgz-my.sharepoint.com/:v:/g/personal/jwen341_connect_hkust-gz_edu_cn/EXq3_f4CBn5JqND1mnNP1gQBDNmhY5jhLA4gOI3i644RMg?e=TKVQs1)**
 🎉🎉
- **Aug-7-25**: [Catch Anything V2 [Demo]](https://www.bilibili.com/video/BV1J1tNzNEag/) has been uploaded. 🎬🎬
- **Aug-2-25**: [Catch Anything V1 [Project]](https://github.com/Jeffry-wen/Drone-Tracking-with-Optical-Flow-and-Color-Histogram) is released. 🔥🔥


---

## 🚀 V3 相比 V2 的核心升级

1. **卡尔曼滤波辅助（显示更稳、失联可维持）**

* 使用 `KalmanBoxTracker`（6维状态：`[cx, cy, vx, vy, w, h]`，4维测量：`[cx, cy, w, h]`），按实际帧率自适应 `dt`。
* **不改变原有检测/决策逻辑**（光流/模板/颜色仍决定“有效框”），Kalman 只用于**平滑显示**和**短暂失联维持**，减少抖动与闪跳。

2. **颜色相似性 + 直方图回投的融合找回更强**

* 以模板中心区域统计 **Lab 空间**(a,b) 的**马氏距离**分布，结合 **HSV** 的 **H-S 直方图回投**，自适应加权融合成候选热力图；
* 通过候选框的 **颜色得分 + 直方图相似度 + 尺度先验 + 中心先验 + 形状紧致度** 进行综合评分筛选，显著提升复杂背景下的找回成功率。

3. **自校正与稳态更新更可靠**

* **面积稳定门限**：若连续 `stable_count_thresh` 帧（默认 50）内面积波动 < `area_stable_thresh`（默认 0.2），自动用扩大区域 `view_r`（默认 1.3）刷新模板，降低漂移风险；
* **周期性匹配校正**：每 `match_interval` 帧（默认 50）通过初始模板进行 NCC 匹配，得分达阈值即重置光流点与模板，抑制累积误差。

4. **ROI 自适应参数**

* 根据目标框面积自动调整 **LK 光流**角点质量、窗口与 **Canny** 阈值，小目标更“敏感”，大目标更“稳”。

5. **更高的帧率 **
* 输入分辨率可调 优化找回逻辑与检测区域


---

## 🧠 系统处理流程

1. **初始化**

   * 读取视频并可选缩放到 `input_size`；
   * 手工框选 ROI，提取 **初始模板**、**光流角点**、**HSV 直方图**；
   * 初始化 **Kalman**（可选开启）。

2. **逐帧跟踪主循环**

   1. 估计当前 FPS → 得到 `dt`；
   2. 计算 **HSV 回投**与灰度图；
   3. **LK 光流**跟踪角点 → 中值中心与 50 像素圈定过滤离群点 → 得到候选框；
   4. **面积稳定监测**与 **模板稳定更新**；
   5. **周期性模板匹配** 与 **尺度异常触发匹配**；
   6. 若光流点不足或匹配失败，则执行 **颜色相似性融合找回**（Lab 马氏距离 + HSV 回投 + 形态学 + 评分）；
   7. **Kalman 融合显示**：有有效测量则 `correct`，否则使用预测框平滑显示；
   8. 绘制最终（平滑）框、中心坐标、FPS、KF 状态，并可写入输出视频。

3. **交互**

   * `R`：人工复位（尝试模板匹配 / 颜色找回并刷新角点）；
   * `ESC`：退出。

---

## ✨ Highlights

* **多策略融合**：LK 光流 + 模板匹配（NCC）+ 颜色相似性（Lab 马氏 + HSV 回投）
* **模板自更新 & 周期校正**：稳定即更新，定期再矫正，抗漂移
* **Kalman 显示平滑**：不干预决策、仅用于显示与短时维持
* **ROI 自适应参数**：小目标更灵敏，大目标更稳健
* **高帧率兼容**：动态 `dt` + 轻量运算路径
* **视频导出**：`mp4v` 编码，支持复盘与标注数据生成

---

## 🔧 关键参数

| 参数名                        |                            默认值 | 说明                     |
| -------------------------- | -----------------------------: | ---------------------- |
| `VIDEO_PATH`               |                           路径示例 | 输入视频路径（或摄像头）           |
| `HSV_LOWER` / `HSV_UPPER`  | `(28,0,113)` / `(180,255,255)` | 初始颜色掩码区间（用于直方图/回投）     |
| `view_r`                   |                          `1.3` | 模板更新时扩大采样窗口比例          |
| `area_stable_thresh`       |                          `0.2` | 面积稳定阈值（相对变化率）          |
| `stable_count_thresh`      |                           `50` | 连续稳定帧数，达标即更新模板         |
| `match_interval`           |                           `50` | 周期性模板匹配间隔（帧）           |
| `area_thresh_range`        |                   `(0.5, 1.6)` | 面积异常触发匹配的相对区间          |
| `FEATURE_MAX_CORNERS`      |                          `300` | LK 角点上限                |
| `save_video` / `save_path` |    `True` / `output_video/...` | 是否保存与保存路径              |
| `input_size`               |                    `(960,540)` | 读入后统一缩放分辨率（`None` 则原始） |
| `USE_KALMAN`               |                         `True` | 是否启用卡尔曼辅助显示            |

> **说明**：Kalman 的过程噪声/测量噪声矩阵已在代码中给出适中的默认值；如轨迹发飘，可略降过程噪声或增大测量噪声；如响应偏慢，可相反调整。

---

## 🧠 算法要点

* **光流主线**：`cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`，对离群点做中位数过滤，使用包围盒更新位置与尺度。
* **卡尔曼滤波辅助**：有有效测量则 `correct`，否则使用预测框平滑显示；
* **模板匹配**：统一在灰度图进行，周期性触发或异常触发；优先在上次 bbox 周边的**局部窗口**搜索（`match_template_nearby`），仅在必要时全图。
* **颜色回捕**：

  * HSV 反投影（`cv2.calcBackProject`）配合 CamShift 微调（保留逻辑）。
  * **Lab 主色快速回捕**：无 SciPy 依赖，降采样 + 量化直方图（`dominant_color_lab_fast`）估计主色；对全图做快速阈值/轮廓筛选后按面积与上一尺度相近原则选取最佳候选（`recover_by_color_similarity_fast`）。
* **自适应**：根据 ROI 尺寸动态调整光流/角点参数（`adjust_parameters_by_roi_size`）；稳定帧增多→**延长匹配间隔**，异常→**缩短间隔**。



## 🖥️ 环境依赖与安装

* Python ≥ 3.8
* OpenCV ≥ 4.5
* Numpy ≥ 1.19，SciPy

```bash
pip install opencv-python numpy scipy
```

---

## 🚀 快速开始

```bash
python catch_anything_v3.py
```

1. 将 `VIDEO_PATH` 指向你的视频文件；
2. 运行后在弹窗中 **框选目标**；
3. 窗口中实时显示追踪结果、中心坐标与 FPS（左上角为 FPS / KF 状态，右上角为中心坐标与提示）；
4. `R` 人工复位；`ESC` 退出；
5. 若开启 `save_video=True`，结果会保存到 `save_path`。

---

## 📁 项目结构

```bash
.
├── catch_anything_v3.py        # 你提供的 V3 主程序
├── README.md                   # 本文档
└── output_video/               # 输出视频目录
```

---

## 🧩 TODO

* 将 `recover_target`（CamShift）融入找回投票流程以增强旋转/尺度鲁棒性；
* 多目标并行（ID 管理 + ReID / 颜色签名）；
* 可选深度特征模板（如轻量 SiamFC/SiamDW）；
* GUI 参数面板与可视化调参。

---

## 📜 许可与致谢

本项目感谢The shan的大力支持，代码仅供交流学习，请勿用于其他用途。📬 [欢迎交流联系](mailto:jwen341@connect.hkust-gz.edu.cn)。

[![Email Me](https://img.shields.io/badge/Email-jwen341%40connect.hkust--gz.edu.cn-blue)](mailto:jwen341@connect.hkust-gz.edu.cn)

---

