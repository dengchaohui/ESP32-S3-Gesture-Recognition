\# 🖐️ ESP32-S3 端侧手势识别系统 (TFLM + Web 大屏)



!\[ESP32-S3](https://img.shields.io/badge/ESP32--S3-Espressif-red)

!\[TFLM](https://img.shields.io/badge/TensorFlow\_Lite\_Micro-Edge\_AI-orange)

!\[C++](https://img.shields.io/badge/C++-17-blue)

!\[WebSocket](https://img.shields.io/badge/WebSocket-Real--time-green)



本项目是一个基于 \*\*ESP32-S3\*\* 和 \*\*MPU6050\*\* 六轴传感器的端侧手势识别系统。使用 1D-CNN 模型在本地完成推理，并内置 HTTP + WebSocket 服务器，实现浏览器端的实时可视化大屏，彻底告别枯燥的串口打印。



\## ✨ 核心特性 (Features)



\- 🧠 \*\*纯端侧 AI\*\*：利用 TensorFlow Lite Micro (TFLM) 部署 FP32 模型，内存占用仅 28KB。

\- 📈 \*\*动态窗口状态机\*\*：结合能量阈值与 ZCR（过零率），精准判定动作开始与结束，有效过滤“拿起放下”等误触。

\- 🌐 \*\*Web 可视化面板\*\*：ESP32 自建 WebSocket 服务器，手机/电脑浏览器直连，实时展示能量、概率与判定结果。

\- 🚀 \*\*大屏展示模式\*\*：识别到画圆/画叉时，网页端弹出巨大且带发光效果的 `○ 圆` 或 `✕ 叉`，极其适合演示与视频录制。



\## 🛠️ 技术栈 (Tech Stack)



\- \*\*硬件\*\*：ESP32-S3 (8MB Flash), MPU6050, I2C (400kHz)

\- \*\*软件框架\*\*：ESP-IDF v5.3.1

\- \*\*边缘计算\*\*：TensorFlow Lite Micro (TFLM)

\- \*\*网络通信\*\*：ESP HTTP Server / WebSocket

\- \*\*前端展示\*\*：原生 HTML5 / JS



\## 🚀 快速开始 (Quick Start)



\### 1. 硬件连接

| MPU6050 | ESP32-S3 |

| :--- | :--- |

| VCC | 3.3V |

| GND | GND |

| SCL | GPIO 9 |

| SDA | GPIO 8 |



\### 2. 编译与烧录

确保已配置 ESP-IDF 环境：

```bash

idf.py build

idf.py -p COM10 flash monitor

