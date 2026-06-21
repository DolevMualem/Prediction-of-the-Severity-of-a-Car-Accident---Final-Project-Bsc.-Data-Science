# Road Accident Severity Prediction Project
## B.Sc. Data Science – Final Project

This repository contains the machine learning pipeline developed to predict road accident severity based on historical collision data. Below are the precise steps required to configure the environment, load the required data assets, and execute the model successfully.

---

## Prerequisites & Dataset Setup

Due to GitHub's file size limitations (the primary datasets exceed hosting constraints), the full data assets are stored securely on Google Drive. 

Please download the datasets from the official link below and place them directly on your **Desktop** before running the model pipeline:

📦 **[Download Project Datasets from Google Drive](https://drive.google.com/drive/folders/1SwST-Vn4-SSpLKjI7_aTVT2_B35ODyI2?usp=sharing)**

### Required Files Checklist:
1. `Collision_Dataset_For_Model_With_Extra_Fatals` — (Place on Desktop)
2. `Collision_Dataset_For_Model_With_Extra_Fatals_No_num_of_casualties` — (Place on Desktop)

*The execution pipeline expects these exact file names and locations to properly execute the data ingestion phase.*


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
