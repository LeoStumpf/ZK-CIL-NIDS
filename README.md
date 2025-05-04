# ZK-CIL-NIDS

**ZK-CIL-NIDS: A Zero-Knowledge Class-Incremental Approach for Network Intrusion Detection**

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🧠 Project Overview

This repository contains the implementation and experiments from the research paper:

**ZK-CIL-NIDS: A Zero-Knowledge Class-Incremental Approach for Network Intrusion Detection**

The goal of this research is to investigate and evaluate a the best novelty detection strategy for **network intrusion detection systems (NIDS)** without requiring prior knowledge of unseen attack classes — a setting we call **Zero-Knowledge Class-Incremental Learning (ZK-CIL)**.

We aim to assess how novelty detection methods can be incrementally applied to evolving threat landscapes, where new types of attacks appear over time, using real-world intrusion datasets.

---

## 📊 Algorithms Compared

The following novelty and outlier detection algorithms are evaluated:

- `Random` (baseline)
- `OneClassForestwoBootstrap`
- `OneClassForest`
- `IsolationForest`
- `NeuronalNetworkLoss`
- `NeuronalNetwork`
- `OneClassSVN`
- `EnergyFlowClassifier`
- `DistanceLOF`
- `LocalOutlierFactor`

These models are tested under class-incremental scenarios to assess their ability to generalize across newly introduced attack classes.

---

## 🗂 Dataset and Splits

Experiments are based on various incremental configurations of training and testing data, simulating real-world deployment where attacks emerge over time. We split the data as follows:

### Day-Based Datasets

| Dataset Name | Train Set | Test Set |
|--------------|-----------|----------|
| TrainDay0_TestDay1234 | BENIGN | BENIGN, FTP-Patator, SSH-Patator, DoS, Heartbleed, Web, Infiltration, Botnet, Portscan, DDoS |
| TrainDay01_TestDay234 | BENIGN, FTP-Patator, SSH-Patator | BENIGN, DoS, Heartbleed, Web, Infiltration, Botnet, Portscan, DDoS |
| TrainDay012_TestDay34 | BENIGN, FTP-Patator, SSH-Patator, DoS, Heartbleed | BENIGN, Web, Infiltration, Botnet, Portscan, DDoS |
| TrainDay0123_TestDay4 | BENIGN, FTP-Patator, SSH-Patator, DoS, Heartbleed, Web, Infiltration | BENIGN, Botnet, Portscan, DDoS |

### Attack-Specific Datasets

| Dataset Name | Train Set | Test Set |
|--------------|-----------|----------|
| TrainDay0_DDoS | BENIGN | BENIGN, DDoS |
| TrainDay0_DoS | BENIGN | BENIGN, DoS |
| TrainDay0_FTPPatator | BENIGN | BENIGN, FTP-Patator |
| TrainDay0_Heartbleed | BENIGN | BENIGN, Heartbleed |
| TrainDay0_Infiltration | BENIGN | BENIGN, Infiltration |
| TrainDay0_SSHPatator | BENIGN | BENIGN, SSH-Patator |
| TrainDay0_Web | BENIGN | BENIGN, Web |

---

## 📁 Folder Structure

```bash
.
├── Test/                # Test definitions and experiment entry points
├── Plots/               # Visualization and postprocessing scripts
├── Implementation/      # Custom novelty detection algorithm implementations
├── datasets/            # Test and Trainings datasets (not uploaded to git)
├── Helper/              # Auxiliary utility functions and helpers
```

## 🛠️ Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/LeoStumpf/ZK-CIL-NIDS
cd ZK-CIL-NIDS
```

### **2️⃣ Install Dependencies**
Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 🚀 Running the Experiments
Navigate to the Test/ directory and run any of the defined scripts for specific dataset evaluations or to reproduce paper results.

```bash
cd Test/
python test1.py
```

## 📄 License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software with proper attribution.