# KNN: Customer Purchase Prediction

## Overview

This project predicts whether a customer will purchase a product using the K-Nearest Neighbors (KNN) algorithm.

## Problem Statement

Classify users into two categories:

- 0 → Not Purchased
- 1 → Purchased

based on demographic features.

## Dataset

Dataset sourced from Kaggle (Social Network Ads).

## Features Used

- Age
- Estimated Salary

## Model

- K-Nearest Neighbors (KNN)

## Key Concept

KNN is a distance-based algorithm. Feature scaling is required to ensure fair distance calculation.

## Results

- Accuracy: ~85–90% (depends on K value)

## Visualization

A graph of **K vs Accuracy** is plotted to find the optimal number of neighbors.

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```
knn-project/
│── data/
│── src/
│── README.md
│── requirements.txt
```

## Conclusion

KNN performs well for this dataset and demonstrates how model performance depends on the choice of K.
