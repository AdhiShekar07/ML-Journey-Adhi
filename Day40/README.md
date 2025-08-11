# 📅 Day 40 – DVC & Data Versioning (From Basic to Advanced)

## 📌 Overview
In Machine Learning projects, datasets change over time — new data is added, old data is cleaned, and features evolve.  
**DVC (Data Version Control)** is a version control system for data and ML models, working alongside Git to track large files, datasets, and model artifacts efficiently.

This ensures:
- **Reproducibility**: Easily roll back to any previous dataset or model.
- **Collaboration**: Share the exact dataset version with teammates.
- **Scalability**: Handle large files without bloating Git repositories.

---

## 🧠 Simple Analogy
Think of **DVC** as **Google Docs Version History** — but instead of tracking text changes, it tracks entire datasets and models.  
Just like you can “go back” to any document version, with DVC you can go back to any dataset or model state.

---

## 🚀 Key Features
- 📂 **Data Versioning** – Track and manage datasets like source code.
- 🔄 **Model Versioning** – Save different model versions with metadata.
- 🌐 **Remote Storage Support** – AWS S3, Google Drive, Azure, SSH, etc.
- 🤝 **Integration with Git** – Works seamlessly alongside Git for code.
- ⚡ **Lightweight Metadata** – Stores file hashes instead of heavy data in Git.

---

## 🛠 How DVC Works
1. **Initialize DVC in your repo**
   ```bash
   dvc init
   ```
2. **Add a dataset to DVC**
   ```bash
   dvc add data/dataset.csv
   git add data/dataset.csv.dvc .gitignore
   git commit -m "Add dataset to DVC"
   ```
3. **Configure remote storage**
   ```bash
   dvc remote add -d myremote s3://mybucket/path
   ```
4. **Push data to remote**
   ```bash
   dvc push
   ```
5. **Pull data from remote**
   ```bash
   dvc pull
   ```

---

## 🧩 Advanced Concepts
- **Pipelines**: Automate data → preprocess → train → evaluate → deploy steps.
- **Metrics & Plots**: Track model performance across versions.
- **Experiment Tracking**: Compare multiple training runs.
- **Data Lineage**: See how data flows from raw files to final model.

---

## 📊 Example Workflow
```bash
# Track data
dvc add data/raw/images.zip
git add data/raw/images.zip.dvc

# Commit changes
git commit -m "Track raw images dataset"

# Push to remote
dvc push

# Later... to reproduce the same dataset version:
dvc pull
```

---

## 📚 Resources
- [Official DVC Docs](https://dvc.org/doc)
- [DVC GitHub Repository](https://github.com/iterative/dvc)
- [ML Reproducibility with DVC](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

---

## 🏷 Extended Description
> **"Day 40 – DVC & Data Versioning: Mastering reproducible ML workflows by tracking datasets and models like code, from basics to advanced features."**

---
