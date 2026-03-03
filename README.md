# Resume–Job Description Matching Using Deep Learning
**From keyword filtering to intelligent semantic ranking (TF-IDF → SBERT → Supervised refinement)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ready-red)
![NLP](https://img.shields.io/badge/NLP-SBERT-success)

A multi-stage ranking pipeline that improves over traditional ATS keyword matching by using **SBERT semantic embeddings** and a **supervised neural refinement layer** to produce **probability-based resume ↔ job alignment**.

---

## Table of contents
- [Problem](#problem)
- [What this repo does](#what-this-repo-does)
- [Data](#data)
- [Method](#method)
- [Results](#results)
- [How to run](#how-to-run)
- [Repo structure](#repo-structure)
- [Limitations & responsible use](#limitations--responsible-use)
- [Team](#team)
- [References](#references)

---

## Problem
Hiring pipelines frequently screen **hundreds of resumes per role**, and keyword-based ATS systems can miss strong candidates due to vocabulary mismatch (e.g., “information assurance” vs “cybersecurity”).

**Goal:** build an intelligent ranking system that:
- understands semantic meaning,
- ranks resumes by contextual relevance,
- refines ranking with supervised learning,
- reduces manual screening effort.

---

## What this repo does
This project implements a **3-stage matching system**:

1) **TF-IDF baseline (lexical matching)**  
   - sparse vectors + cosine similarity  
   - useful as a traditional ATS baseline, but limited to word overlap

2) **SBERT semantic matching (context-aware retrieval)**  
   - model: `all-MiniLM-L6-v2`  
   - dense embeddings (384-d) + cosine similarity  
   - captures meaning beyond keywords

3) **Supervised neural refinement (match probability)**  
   - transforms similarity ranking into a binary classification task  
   - outputs a calibrated **match probability** (0–1)

---

## Data
Focused on IT to keep domain consistent and evaluation realistic:
- **120 IT resumes**
- **2,277 job descriptions**
- **15 technical job roles**
- **Many-to-many** ranking scenario (each resume can match many JDs)

> Dataset filenames used by the notebook:  
`/Resume/Resume.csv` and `/JobDescription/job_title_des.csv`

---

## Method

### Pipeline overview
```mermaid
flowchart LR
  A[Resume Text] --> P1[Preprocess\n(strip HTML, normalize)]
  B[Job Description Text] --> P1

  P1 --> T1[TF-IDF\n(5k features)]
  P1 --> S1[SBERT\nall-MiniLM-L6-v2\n384-d embeddings]

  T1 --> R1[Cosine Similarity\nBaseline Rank]
  S1 --> R2[Cosine Similarity\nSemantic Rank]

  R2 --> L1[Pseudo-Labels\nTop-3 = 1\nBottom-3 = 0]
  L1 --> N1[Neural Refinement MLP\n768-d input = concat(384+384)]
  N1 --> O1[Match Probability]
  O1 --> Final[Refined Ranking]
