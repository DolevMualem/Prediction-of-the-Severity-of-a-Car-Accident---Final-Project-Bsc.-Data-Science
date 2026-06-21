# Road Accident Severity Prediction Project
## B.Sc. Data Science – Final Project

This repository contains the machine learning pipeline developed to predict road accident severity based on historical collision data. Below are the precise steps required to configure the environment, load the required data assets, and execute the model successfully.

---

## Prerequisites & Dataset Setup

Before running the model, please ensure the required dataset files are correctly placed on your local system's Desktop. The pipeline expects specific file naming conventions to correctly locate and ingest the data.

1. **Dataset 1 (Primary):** Download and place the file named exactly `Collision_Dataset_For_Model_With_Extra_Fatals` on your **Desktop**.
2. **Dataset 2 (Target Adjusted):** Download and place the file named exactly `Collision_Dataset_For_Model_With_Extra_Fatals_No_num_of_casualties` on your **Desktop**.

---

## Execution Environment Options

You can evaluate the project using one of the following two environments:

### Option A: Google Colab Notebook (Recommended for Clean Visual Presentation)
* Open the notebook file `FinalModelDataSceinceBSc.ipynb` via Google Colab.
* This environment contains complete pre-rendered visual analytics, exploratory data analysis (EDA), and cleanly structured markdown annotations for an optimal review experience.

### Option B: Local Python Script
* Open the source file `555FinalLightGBM.py` inside your preferred Integrated Development Environment (IDE).
* Ensure all dependencies (such as `lightgbm`, `pandas`, `numpy`, and `scikit-learn`) are fully updated in your active environment.

---

## Step-by-Step Execution Guide

1. **Launch the Runtime:** Execute the Python script (`555FinalLightGBM.py`) or run the notebook cells sequentially in your Colab environment.
2. **Data Ingestion Prompts:** When prompted by the interface or executing the data loading block, upload or grant path access to the two dataset files located on your **Desktop**.
3. **Model Processing:** Allow the notebook/script to complete its preprocessing pipeline, feature engineering, and LightGBM model evaluation loop.
4. **View Performance Metrics:** Once processing finishes, complete classification reports, feature importance charts, and model evaluation metrics will display directly within the output console/cells.

---

*Thank you for evaluating our project! We have invested extensive research, data engineering, and optimization efforts into this pipeline to ensure robust predictive analytics.*
