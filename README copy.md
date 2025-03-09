# **MaintenaSense - Predictive Maintenance System**

## **Project Overview**
MaintenaSense is an **AI-powered predictive maintenance system** designed to detect early warning signs of machinery failure using **sensor data analysis**. The goal is to **reduce unplanned downtime, optimize maintenance scheduling, and lower costs** by identifying anomalies before breakdowns occur.

This system is ideal for **manufacturing plants, industrial facilities, and smart factories** looking to transition from **reactive to predictive maintenance**.

---

## **Repository Structure**
This repository contains all the necessary files for building, training, and evaluating the predictive maintenance models.

```
📂 MaintenaSense
├── 📂 data                  # Datasets used for training and testing
├── 📂 models                # Trained AI models (Gradient Boost, Random Forest, LSTM, Autoencoder)
├── 📂 notebooks             # Jupyter notebooks for data analysis, model training, and evaluation
├── 📂 scripts               # Python scripts for preprocessing, anomaly detection, and visualization
├── README.md                # Project description and setup guide
└── requirements.txt         # Dependencies required to run the project
```

---

## **Key Features**
**Real-time Anomaly Detection:** Detects unusual machine behavior using AI models.  
**Predictive Failure Alerts:** Provides early warnings before breakdowns occur.  
**Sensor-Based Insights:** Uses data like **temperature, vibration, and pressure** to predict failures.  
**Hybrid Model Approach:** Combines traditional machine learning (Gradient Boost, Random Forest) with deep learning (Autoencoders, LSTM) for better accuracy.  
**Technician Feedback Loop:** Allows human validation of alerts to refine AI predictions.  

---

## **Datasets Used**
Since real-world factory data was unavailable, we selected **public machine sensor datasets** from **Kaggle** that closely resemble industrial environments. The datasets used include:
- **Water Pump Dataset** (Primary dataset used for training)
- **AI4I Synthetic Dataset**
- **Car Engine Data**
- **Equipment Failure Dataset**
- **MFPSuD Dataset**
- **Predictive Maintenance Dataset**

These datasets contain **time-series sensor readings** labeled with failure events. 

**Key Challenge:** **Severe class imbalance**, as machine failures are rare compared to normal operation. 

---

## **Models Implemented**
Several models were tested to determine the best approach for **predictive maintenance:**

### **1. Machine Learning Models (Best for Classification)**
- **Gradient Boosting** (Best overall performer - 99.93% accuracy)
- **Random Forest** (99.92% accuracy, high recall)
- **Extra Trees Classifier**
- **Decision Trees** (Dropped due to low performance)

### **2. Deep Learning Models (Best for Anomaly Detection)**
- **Autoencoder** (Detects subtle anomalies but requires tuning)
- **LSTM-based Anomaly Detection** (Captures sequential sensor patterns but underperformed in direct classification)

**Final Conclusion:** 
- **Gradient Boosting and Random Forest are the most effective for failure classification.**
- **Autoencoders and LSTMs hold promise for improving anomaly detection in future versions.**

---

## **Evaluation Metrics**
The models were evaluated using the following metrics:
- **Accuracy:** Measures the overall correctness of predictions.
- **Precision:** Ensures fewer false alarms.
- **Recall:** Captures the actual failures that occurred.
- **F1 Score:** Balances precision and recall.

**Final Model Performance:**
| Model          | Accuracy  | Precision | Recall | F1 Score |
|---------------|-----------|------------|------------|------------|
| Gradient Boost | **99.93%** | **0.9935** | **0.9956** | **0.9946** |
| Random Forest  | **99.92%** | **0.9907** | **0.9974** | **0.9940** |
| Autoencoder    | 93.76% | 0.6688 | 0.1017 | 0.1767 |
| LSTM           | 92.51% | 0.0426 | 0.0064 | 0.0113 |

---

## **How to Use MaintenaSense**

### **1. Installation**
Clone this repository and install dependencies:
```
git clone https://github.com/Rodauv/MaintenaSense-Final-Project
cd MaintenaSense-Final-Project
pip install -r requirements.txt
```

### **2. Running the Model**
To test the predictive maintenance model:
```
python dashboard.py
```
This will **analyze the sensor data and generate failure predictions**.

### **3. Viewing Results**
- **Alerts and anomalies** will be displayed in the terminal.
- **A visualization dashboard** is available in the `notebooks/visualizations.ipynb` file.

---

## **Future Enhancements**
While MaintenaSense has demonstrated strong performance, there are areas for further development:

### **1. Expanding Real-World Testing**
- Deploy MaintenaSense in a real factory environment to collect **actual sensor data**.
- Gather technician feedback to improve system accuracy.

### **2. Improving Anomaly Detection**
- Refine **Autoencoder and LSTM models** to better capture early failure indicators.
- Implement **attention mechanisms** in LSTM models for improved predictions.

### **3. Reducing False Positives & Building AI Trust**
- Enhance **explainability in AI predictions** (e.g., adding trend visualizations for better decision-making).
- Develop an **adaptive learning system**, where AI adjusts its sensitivity based on technician feedback.

### **4. IoT & Cloud Integration**
- Future updates will focus on **integrating MaintenaSense with IoT platforms** for real-time monitoring.
- Enable **cloud-based predictive maintenance analytics**, allowing multiple factories to share failure patterns and improve accuracy collectively.

---

## **Contributors**
MaintenaSense was developed as part of a research-driven project to advance predictive maintenance solutions. Contributions to improve and expand the system are welcome!

---

## **License**
MaintenaSense is an open-source project under the MIT License. Feel free to use and improve upon the existing code for research and industrial applications.

For inquiries and collaborations, please contact: https://github.com/Rodauv

---

This page provides **a structured, easy-to-understand overview** of your project, making it ideal for a **README file** or repository documentation. Let me know if you need **any refinements!** 🚀