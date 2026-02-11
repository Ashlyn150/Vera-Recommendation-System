# 👗 Fashion Recommender System

**Visual Similarity Recommendation using FAISS (KNN) + Interactive
Streamlit Application**

------------------------------------------------------------------------

## 📌 Project Overview

This project implements a **content-based fashion recommendation
system** that retrieves visually similar products using deep image
embeddings and FAISS-powered K-Nearest Neighbour (KNN) search.

The system simulates how an e-commerce fashion retailer recommends
similar items on a product detail page, integrating:

-   Image embedding representation\
-   FAISS vector similarity search\
-   Business-rule filtering\
-   Front-facing image prioritisation\
-   Interactive product-style interface

The focus of this implementation is the **design, engineering, and
integration of a KNN-based similarity pipeline into a production-style
UI environment**.

------------------------------------------------------------------------

## 🎯 Business Motivation

In fashion e-commerce, visually similar recommendations help:

-   Increase cross-sell and basket size\
-   Improve engagement on product pages\
-   Maintain stylistic consistency\
-   Encourage browsing behavior

This project simulates a real-world workflow where similarity retrieval
must be combined with business constraints and UI requirements.

------------------------------------------------------------------------

## 🧠 System Architecture

Product Images\
↓\
Deep Feature Embeddings\
↓\
FAISS Index (Vector Store)\
↓\
KNN Similarity Retrieval\
↓\
Business Filtering Layer\
↓\
Streamlit Product Interface

------------------------------------------------------------------------

## ⚙️ Core Technical Components

### 1️⃣ Image Embedding Representation

-   Each product is represented by a fixed-length feature vector.
-   Embeddings capture visual patterns beyond metadata or category
    labels.
-   Enables scalable similarity-based comparison.

------------------------------------------------------------------------

### 2️⃣ FAISS-Based KNN Retrieval

-   FAISS used for efficient nearest-neighbour search.
-   Supports configurable parameters:
    -   top_k
    -   similarity threshold
    -   category filtering
    -   stock filtering
-   Designed to scale to large embedding collections.

FAISS is widely adopted in industry for vector search systems.

------------------------------------------------------------------------

### 3️⃣ Business Filtering Layer

After KNN retrieval, additional constraints are applied:

-   Remove the query product itself\
-   Enforce same gender consistency\
-   Optional same-category filtering\
-   Similarity threshold control (e.g., ≥ 0.50)\
-   Deduplicate results

This simulates production-level recommendation logic rather than raw
similarity output.

------------------------------------------------------------------------

### 4️⃣ Front-Facing Image Prioritisation

To ensure UI consistency:

-   Automatically selects \*\_front.jpg if available\
-   Falls back to alternative product images if necessary\
-   Preserves original metadata\
-   Uses cached lookup to avoid repeated directory scanning

This resolves common mismatches between metadata image paths and actual
file structure.

------------------------------------------------------------------------

### 5️⃣ Interactive Streamlit Product Interface

The application simulates a real e-commerce product detail page with:

-   Main product display\
-   Top-5 visually similar recommendations\
-   Gender and category filters\
-   Deterministic pricing generation for reproducibility\
-   Session-state controlled product switching\
-   Modular retrieval and rendering separation

The UI demonstrates how ML outputs integrate into a product-facing
application.

------------------------------------------------------------------------

## 📂 Project Structure

workflow/\
├── data/\
│ └── img/\
├── Index/\
├── src/\
│ ├── recommender_knn.py\
│ └── sim_helpers.py\
├── Simulation_KNN/\
│ └── Simulation.py\
└── app.py

------------------------------------------------------------------------

## 🚀 How to Run

Install dependencies:

pip install -r requirements.txt

Launch the application:

streamlit run app.py

Launch simulation environment:

cd Simulation_KNN\
streamlit run Simulation.py

------------------------------------------------------------------------

## 📊 Technical Highlights

-   FAISS vector similarity search integration\
-   Modular recommender system design\
-   Clean separation of retrieval, business logic, and UI layers\
-   Deterministic reproducibility design\
-   Efficient metadata merging strategy\
-   LRU-cached image path resolution\
-   Structured debugging and performance optimisation

------------------------------------------------------------------------

## 💡 Skills Demonstrated

-   Recommender system design\
-   Vector search engineering (FAISS)\
-   Data integration and transformation\
-   Applied business-rule modelling\
-   ML system-to-UI integration\
-   Scalable project structuring

------------------------------------------------------------------------

## 👩‍💻 Author

Developed as part of a Business Analytics / Data Mining project focused
on applied recommender systems and similarity search engineering.
