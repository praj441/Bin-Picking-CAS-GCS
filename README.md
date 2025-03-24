# 🏗️ **Source Code for Our Work**  
**"Bin-picking with Category-Agnostic Segmentation for Unreliable Depth Scenarios"**  

---

## 📌 **Overview**  
This repository contains the source code for our research on **bin-picking using category-agnostic segmentation**, specifically designed for scenarios with unreliable depth information. Robotic table-top grasping (bin-picking) of unknown, heterogeneous objects is challenging due to clutter and unreliable depth data, especially with commodity-grade sensors for thin or transparent objects. To address this, we propose a depth-independent CNN that co-learns category-agnostic instance segmentation, instance-wise grasp-confidence scores (GCS), and monocular depth estimation from an RGB image. The predicted depth assists in collision detection and 3D grasp pose transformation. The GCS branch helps filter graspable objects, improving efficiency by reducing unnecessary grasp planning. A custom synthetic dataset is used for training, and experiments show that our method reliably picks various unknown objects, including non-opaque and thin objects, from cluttered bins.

---




### **CNN Design**
<img src="data/images/cnn_design.png" alt="CNN Design" width="900">

### **Grasp Planning Framework**
<img src="data/images/grasp_planning_framework.png" alt="Grasp Planning Framework" width="900">

---

## 🔍 **How to Access the Code?**  

Navigate to the respective folders for different modules of our work:  

📂 **`CNN_src/`** → Source code for **CNN design, training, and testing**.  
📂 **`simulation/`** → Source code for **simulation & synthetic dataset generation**.  
📂 **`grasp_planning/`** → Source code for **grasp-pose planning and execution**.  

---

## 🚀 **Getting Started**  
### **🔹 Clone the Repository**  
```bash
git clone https://github.com/praj441/Bin-Picking-CAS-GCS.git
cd Bin-Picking-CAS-GCS
