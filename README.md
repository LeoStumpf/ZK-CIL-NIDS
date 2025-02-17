# **ZK-CIL-NIDS**  
### *A Zero-Knowledge Class-Incremental Approach for Network Intrusion Detection*  

[![License](TBD)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![Status](https://img.shields.io/badge/status-active-success.svg)]()  

## 🚀 Overview  
**ZK-CIL-NIDS** is a **Zero-Knowledge Class-Incremental Learning (CIL) framework** for **Network Intrusion Detection Systems (NIDS)**. In previous work, a **ZK-CIL-NIDS** method was developed, which is used in this study but was not created as part of this paper. This repository and accompanying paper focus on **evaluating and comparing different CIL-NIDS approaches** from the literature, including our previously developed method, to determine their effectiveness in detecting unknown attack classes **without prior knowledge**. By leveraging **incremental learning and outlier scoring techniques**, we assess the performance of various methods in both **standard and zero-knowledge settings**, providing insights into their adaptability to **emerging cyber threats**.

## 🔥 Key Features  
✅ **Zero-Knowledge Learning** – No prior attack labels or predefined knowledge required  
✅ **Class-Incremental Learning (CIL)** – Detects and adapts to new attack classes dynamically  
✅ **Outlier Scoring** – Evaluates the best scoring method for identifying novel threats  
✅ **Scalability** – Designed to handle large-scale network traffic data  
✅ **Extensible Framework** – Can be integrated with various NIDS architectures  

## 🛠️ Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/ZK-CIL-NIDS.git
cd ZK-CIL-NIDS
```

### **2️⃣ Install Dependencies**
Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **📂 Project Structure**
```bash
📦 ZK-CIL-NIDS
├── 📂 data                # Datasets and preprocessing scripts
├── 📂 models              # Machine learning models for incremental learning
├── 📂 evaluation          # Outlier scoring and performance evaluation scripts
├── 📂 experiments         # Experiment scripts for different NIDS scenarios
├── 📜 README.md           # Project documentation
├── 📜 requirements.txt    # Python dependencies
└── 📜 main.py             # Entry point for training and evaluation
```


