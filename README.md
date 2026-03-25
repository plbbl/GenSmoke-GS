<div align="center">

# 🚀 GenSmoke-GS

### Reconstruction-Oriented Multi-Stage Framework for Smoke-Degraded Multi-View Images

🔥 **Results released (03-25). Code and models coming soon.**

</div>

---

## 📌 Overview

**GenSmoke-GS** is a reconstruction-oriented multi-stage pipeline designed to improve **3D reconstruction under smoke-degraded multi-view conditions**.


---

## 📦 Data & Results (03-25 Release)

> 📢 All results are released via **Baidu Netdisk**
> 🔑 Extraction code: **`plbb`**

---

### 📁 1. Smoke-Degraded Dataset

* 🔗 https://pan.baidu.com/s/1e1jp8-uuqqKyWFSoL_VyvA

---

### 🔧 2. Preliminary Restoration (UDPNet)

* 🔗 https://pan.baidu.com/s/1Ea5j3WNVK3vdZU8eVMRcAg

---

### 🌫️ 3. Dehazing (DCP)

* 🔗 https://pan.baidu.com/s/1IDokNAZgEUQw1S2c8iXbFw

---

### 🧠 4. MLLM Enhancement

* 🔗 https://pan.baidu.com/s/1M14Tw5RY42ovroslUz0PWA

---

### 🧱 5. Final Reconstruction Results

* 🔗 https://pan.baidu.com/s/1pW--LhgjuKOiCylqCLJcaQ

---

### 🔁 6. Multi-run Results (91 runs)

* 🔗 https://pan.baidu.com/s/1roCxrpJEd8pTqFOMCbMlyQ

---

## 🔄 Pipeline Summary

```text
Haze Images
   ↓
UDPNet Restoration
   ↓
DCP Dehazing
   ↓
MLLM Enhancement
   ↓
3DGS-MCMC Reconstruction (91 runs)
   ↓
Multi-run Averaging
   ↓
Final NVS Results
```

---

## 🚧 Release Plan

> ⚠️ This repository is under active release. Components will be released step-by-step.

### ✅ 2026-03-25 (Results)

* [x] Repository initialized
* [x] Release intermediate results
* [x] Release final results
* [ ] Add detailed documentation

---

### 🔜 2026-03-26+ (Code)

* [ ] Release core pipeline code
* [ ] Release preprocessing scripts
* [ ] Release reconstruction scripts
* [ ] Provide usage instructions

---

### 🔜 2026-03-27+ (Weights)

* [ ] Release model checkpoints
* [ ] Provide download links
* [ ] Add evaluation scripts

---




## 📊 Method Pipeline

> 🚧 Visualization coming soon

---

## 🚀 Quick Start

> ⏳ Code will be available after **2026-03-26**

---

## 📦 TODO

* [x] Initialize repository
* [x] Upload results (03-25)
* [ ] Upload code (03-26)
* [ ] Upload weights (03-27)
* [ ] Add documentation
* [ ] Add examples
* [ ] Add configs
* [ ] Add reproducibility guide

---

## 📌 Notes

* This project follows a **reconstruction-oriented design philosophy**
* MLLM enhancement is applied **independently per view with structure preservation**
* Multi-run reconstruction improves **robustness and stability**
* Final outputs are obtained via **averaging across multiple runs**

---

## 📬 Contact

If you have any questions, feel free to:

* Open an **Issue**
* Contact the authors

---
