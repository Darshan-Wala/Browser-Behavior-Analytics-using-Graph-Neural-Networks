# Browser Behavior Analytics using Graph Neural Networks

A scalable browser behavior analytics platform that leverages **Graph Neural Networks (Graph Attention Networks)** to detect anomalous browser activity from large-scale telemetry data. The project focuses on transforming browser interaction logs into graph representations and performing graph-based behavioral analysis for intelligent threat detection.

---

## 🚀 Features

- Scalable data pipeline for collecting and preprocessing browser telemetry data
- Graph construction from browser sessions using NetworkX
- Graph Attention Network (GAT) implementation using PyTorch Geometric
- Large-scale behavioral analysis on browser interaction data
- Feature engineering and graph-based representation learning
- Model training, inference, and graph visualization
- Attention score visualization for model interpretability

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, PyTorch Geometric
- **Graph Processing:** NetworkX
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Graph Attention Networks (GAT)

---

## 📂 Project Structure

```
Browser-Behavior-Analytics-using-Graph-Neural-Networks/
│
├── data/                  # Dataset (excluded from repository)
├── models/                # Saved model checkpoints
├── outputs/               # Generated graph visualizations
├── src/
│   ├── graph_builder.py
│   ├── pyg_dataset.py
│   ├── gat_model.py
│   ├── train_gat.py
│   ├── inference.py
│   └── visualize_graph.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Workflow

1. Collect browser telemetry logs.
2. Preprocess and clean raw browser activity.
3. Construct session-level graphs using NetworkX.
4. Generate graph datasets for PyTorch Geometric.
5. Train a Graph Attention Network (GAT).
6. Perform inference on unseen browser sessions.
7. Visualize graph structure and attention weights.

---

## 📊 Output

The project generates:

- Browser interaction graphs
- Attention heatmaps
- Model predictions
- Graph visualizations

Example outputs are available in the **outputs/** directory.

---

## 📁 Dataset

The original browser telemetry dataset (**1M+ records**) is **not included** in this repository due to GitHub file size limitations.

The repository contains the complete implementation of:

- Data preprocessing
- Graph construction
- Feature engineering
- Graph Neural Network pipeline
- Model training
- Inference workflow

---

## 🎯 Future Improvements

- Distributed data processing using PySpark
- Real-time browser behavior monitoring
- Incremental graph updates
- Explainable AI for graph predictions
- Containerized deployment using Docker

---

## 👨‍💻 Author

**Darshan Wala**

GitHub: https://github.com/Darshan-Wala
