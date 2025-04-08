# GridNet: Vision-based Mitigation of GPS Attacks for Aerial Vehicles

**[IJCNN 2025]**  

---

## 📌 Introduction

GridNet is a vision-based deep learning framework designed to **mitigate GPS jamming and spoofing attacks** against UAVs. Unlike traditional GNSS security methods, GridNet leverages **real-time aerial imagery** and deep neural networks to **localize aerial vehicles without relying on GPS signals**.

This repository provides **code, models, and datasets** used in our IJCNN 2025 paper. It aims to facilitate research on robust, mapless UAV navigation in GPS-denied environments.


---

## 🧠 System Architecture

GridNet consists of:
- A pre-trained **CycleGAN**: transforms real-time aerial images into satellite style.
- A **grid-based classifier**: locates UAVs on satellite grids.
- A **localization module**: extracts GPS-like coordinates from classification results.

---

## 📁 Repository Contents

```bash
GridNet/
├── IJCNN25_Gridnet.pdf      # 📄 Paper
├── models/                  # 🧠 Pre-trained DNN and CycleGAN
├── code/                    # 🧪 Inference and training scripts
├── data/                    # 📷 Sample satellite & aerial data (small-scale)
├── README.md                # 📘 Project documentation

```

---

## 🔬 Citation

If you use this work, please cite:

```bibtex
@inproceedings{GridNet2025,
  title     = {GridNet: Vision-based Mitigation of GPS Attacks for Aerial Vehicles},
  author    = {Nian Xue and Zhen Li and Xianbin Hong and Christina Pöpper},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025}
}
