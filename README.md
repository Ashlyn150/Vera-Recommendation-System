# 📌 Fashion Recommender System (KNN-Based)

A content-based fashion recommendation system built using **KNN
similarity search with FAISS**, integrated into an interactive Streamlit
application.

This project focuses on building a **modular, production-style
recommender pipeline**, including embedding indexing, similarity
retrieval, metadata integration, and UI deployment.

------------------------------------------------------------------------

## 🚀 Project Overview

This system enables:

-   Visual similarity-based fashion item recommendations
-   Top-K nearest neighbour retrieval using FAISS
-   Interactive product detail exploration via Streamlit
-   Controlled business-rule filtering (category, stock, gender)
-   Deterministic simulation-ready design

The project architecture mirrors real-world recommender system
pipelines.

------------------------------------------------------------------------

## 🧠 Core Methodology

### 1️⃣ Embedding-Based Representation

Each product is represented as a precomputed feature embedding vector.

These embeddings are indexed using:

-   **FAISS (Facebook AI Similarity Search)**
-   Efficient approximate nearest neighbour search

------------------------------------------------------------------------

### 2️⃣ KNN Retrieval Engine

For a given query product:

1.  Retrieve top-K nearest vectors
2.  Remove self-matches
3.  Apply business constraints:
    -   Same gender
    -   Optional same category
    -   Optional stock filter
4.  Merge product metadata

------------------------------------------------------------------------

### 3️⃣ Front-Facing Image Resolution

To ensure UI consistency:

-   Only front-facing images are used
-   Image paths are dynamically resolved from disk
-   LRU caching is applied to avoid repeated filesystem scans

------------------------------------------------------------------------

## 🖥 Streamlit Interface

### Product Detail Page

-   Main item display
-   Stable synthetic pricing
-   Size selection UI
-   Top-5 KNN similar items
-   <img width="2704" height="2028" alt="image" src="https://github.com/user-attachments/assets/e39fe72e-39dd-45e8-ae01-2f2a4c324e73" />


### Simulation Integration

-   Shares same embeddings and product artifacts
-   Ensures reproducible behaviour across modules
-   <img width="2038" height="310" alt="image" src="https://github.com/user-attachments/assets/f0847843-fdcf-402a-829b-e14f847fd19f" />


------------------------------------------------------------------------

## 🗂 Project Structure

    workflow/
    │
    ├── Simulation_KNN/
    │   ├── src/
    │   │   ├── recommender_knn.py
    │   │   ├── sim_helpers.py
    │   │
    │   ├── pages/
    │   │   ├── Simulation.py
    │   │   ├── KNN_Top5.py
    │
    ├── data/
    │   ├── img/
    │   ├── products.pkl
    │
    ├── Index/
    │   ├── FAISS index files

------------------------------------------------------------------------

## ⚙️ Installation & Running

### 1️⃣ Clone repository

``` bash
git clone <your_repo_url>
cd <repo_folder>
```

### 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

``` bash
cd workflow/Simulation_KNN
streamlit run Simulation.py
```


------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   FAISS
-   Pandas
-   NumPy
-   Streamlit
-   Pillow
-   LRU Caching
-   Deterministic randomisation design

------------------------------------------------------------------------

## 📊 Engineering Highlights

-   FAISS vector similarity search integration
-   Modular recommender system design
-   Clean separation of retrieval, business logic, and UI layers
-   Deterministic reproducibility design
-   Efficient metadata merging strategy
-   LRU-cached image path resolution
-   Structured debugging and performance optimisation

------------------------------------------------------------------------

## 🎯 Skills Demonstrated

-   Recommender system engineering
-   Vector search implementation (FAISS)
-   Similarity search optimisation
-   Business-rule modelling
-   Data integration and transformation
-   ML-to-UI deployment workflow
-   Scalable project structuring

------------------------------------------------------------------------

## 📌 Future Improvements

-   GPU-based FAISS acceleration
-   Online evaluation metrics integration
-   API-based deployment (FastAPI)
-   Logging and monitoring integration

------------------------------------------------------------------------

## 📄 License

Academic / Portfolio Use
