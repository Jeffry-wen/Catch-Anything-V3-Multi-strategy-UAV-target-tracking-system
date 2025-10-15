# 🚁 Catch Anything V3：The multi-strategy unmanned aerial vehicle target tracking system assisted by Kalman filtering

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

- **2025-10-15**： **Catch Anything V3 is released.**（卡尔曼辅助 + 多颜色空间找回 + 兼顾高帧率）。
- **Aug-8-31**: Catch Anything V2 **[Code]** has been uploaded.- **[Test video download link](https://hkustgz-my.sharepoint.com/:v:/g/personal/jwen341_connect_hkust-gz_edu_cn/EXq3_f4CBn5JqND1mnNP1gQBDNmhY5jhLA4gOI3i644RMg?e=TKVQs1)**
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



## 🧮 核心算法

本项目采用“**稀疏光流 + 局部模板匹配 + 颜色/直方图恢复 + 自适应模板**”的多源融合方案，实现对小目标在遮挡与外观变化场景下的稳健实时跟踪。

### 1) 稀疏光流（Pyramidal Lucas–Kanade, KLT）与角点筛选（Shi–Tomasi）

**亮度恒常 + 小位移线性化**：对帧间位移 \$ \mathbf{u}=\[u,v]^\top \$ 有

$$
I(x+u,y+v,t+1)\approx I(x,y,t)+\nabla I^\top \mathbf{u}+I_t .
$$

在窗口 \$\Omega\$ 内最小二乘：

$$
E(\mathbf{u})=\sum_{\mathbf{x}\in\Omega} w(\mathbf{x})\big(\nabla I(\mathbf{x})^\top \mathbf{u}+I_t(\mathbf{x})\big)^2 .
$$

得到正规方程（\$\mathbf{G}\$ 为结构张量）：

$$
\underbrace{\sum_{\Omega} w\,\nabla I\,\nabla I^\top}_{\mathbf{G}}\;\mathbf{u}
=
-\sum_{\Omega} w\,\nabla I\, I_t .
$$

**角点可跟踪性（Shi–Tomasi）**：

$$
R=\lambda_{\min}(\mathbf{G})>\tau .
$$

实现：`goodFeaturesToTrack` 获取角点，`calcOpticalFlowPyrLK` 金字塔解；以**中位数**去噪并用存活点的包围盒更新 \$(x,y,w,h)\$。

---

### 2) 局部归一化互相关模板匹配（NCC，灰度，邻域搜索）

对初始模板 \$T\$ 与当前帧灰度图 \$I\$ 计算（OpenCV `TM_CCOEFF_NORMED`）：

$$
R(x,y)=
\frac{\sum_{i,j}\big(T_{ij}-\bar T\big)\big(I_{x+i,y+j}-\overline{I}_{x,y}\big)}
{\sqrt{\sum_{i,j}\big(T_{ij}-\bar T\big)^2}\;
 \sqrt{\sum_{i,j}\big(I_{x+i,y+j}-\overline{I}_{x,y}\big)^2}}
\in[-1,1].
$$

只在上一帧包围盒 \$B=(x,y,w,h)\$ 的**自适应扩展窗口**内搜索：

$$
\alpha=\max\!\left(\alpha_0,\;1.2\cdot\min\!\left\{\frac{W_T}{w},\,\frac{H_T}{h}\right\}\right),
$$

若 \$\max R\ge s\_{\min}\$ 则接受匹配并矫正漂移。

---

### 3) HSV 直方图反投影 + CamShift 恢复

在初始 ROI 估计二维直方图 \$p(h,s)\$，对整帧进行**反投影**：

$$
b(x,y)=p\big(H(x,y),\,S(x,y)\big)\in[0,1].
$$

在 \$b\$ 上运行 **CamShift** 得到窗口 \$B'\$，并以面积一致性判据过滤：

$$
0.7\,A_{\text{prev}} \;<\; A' \;<\; 1.3\,A_{\text{prev}},\qquad A'=w'h'.
$$

---

### 4) Lab 主色 + 颜色相似性快速找回（无 SciPy）

模板降采样至 CIE-Lab 并量化 16 档：

$$
\tilde L=\Big\lfloor\frac{L}{16}\Big\rfloor,\quad
\tilde a=\Big\lfloor\frac{a}{16}\Big\rfloor,\quad
\tilde b=\Big\lfloor\frac{b}{16}\Big\rfloor,
$$

以直方图众数得主色 \$\mathbf{c}\_0=\[L\_0,a\_0,b\_0]^\top\$。

对当前帧计算 Lab 欧氏距离并按 10% 分位阈值得到掩膜：

$$
d(x,y)=\big\|\mathbf{c}(x,y)-\mathbf{c}_0\big\|_2,\qquad
\mathbb{M}(x,y)=\mathbf{1}\{d(x,y)<t\}.
$$

在中心 \$1/3\$ 视野内取各连通域外接矩形 \$(w\_i,h\_i)\$，按最小尺度差选择候选：

$$
\hat{i}=\arg\min_i \big|w_i h_i - A_{\text{prev}}\big|.
$$

---

### 5) 自适应模板更新与稳定性判据

以面积 \$A\_t=w\_t h\_t\$ 的相对变化衡量稳定：

$$
\frac{|A_t-A_{t-1}|}{\max(A_{t-1},1)}<\tau_{\text{area}}
\;\Rightarrow\; s\leftarrow s+1 .
$$

当 \$s\ge N\_{\text{stable}}\$ 时，以倍率 \$r\$ 扩展当前 ROI 更新模板；并依据稳定/异常**动态调整**模板匹配间隔 \$K\$（稳定↑，异常↓）。若

$$
A_t\notin\big[\alpha_{\min}A_{\text{ref}},\,\alpha_{\max}A_{\text{ref}}\big],
$$

则触发模板/颜色快速纠偏。

---

### 6) 多源融合优先级

1. **KLT 光流**成功（足够内点、残差可接受）→ 直接更新 \$B\$；
2. 周期或异常触发 **NCC 局部模板匹配** → 减漂移；
3. 光流/模板失败 → **Lab 主色相似性**快速找回；
4. 颜色弱或遮挡严重 → **反投影 + CamShift** 辅助恢复；
5. 达到稳定阈值 → **模板自更新** 并刷新角点集。

> 以上流程在**速度**（邻域搜索、量化统计）与**鲁棒性**（多模态证据、尺度/面积约束、外点抑制）之间取得平衡，适配无人机等小目标的实时跟踪。

---

如果你仍遇到不渲染的情况，请检查：

* `README.md` 中 `$$...$$` 公式前后是否各有**一行空行**；
* 公式是否被 4 个空格或 Tab **缩进**（那会被当作代码块）；
* 是否在表格单元格内（GFM 表格对公式的支持较差，建议表格外书写）。

---

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

