Anwar - MSDS25002 - Assignment 02
Linear Regression & Logistic Regression

Requirements:
- Python 3
- numpy
- pandas
- matplotlib
- torch (PyTorch)
- scikit-learn

Install:
py -m pip install numpy pandas matplotlib scikit-learn
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Folder Structure:
Anwar_MSDS25002_02/
    task1_linear_regression.py
    task2_logistic_regression.py
    saved_models/
        linear_model.pkl
        logistic_model.pkl
    results/
        task1/
        task2/
    Task-1 Dataset (California Housing)/
        california_housing_train.csv
        california_housing_test.csv
    Task-2 Dataset (Titanic)/
        train.csv
        test.csv
        gender_submission.csv
    Readme.txt
    Report.pdf

How to Run:

Task 1 - Linear Regression (California Housing):
    Single run:
        py task1_linear_regression.py
    Run all experiments:
        py task1_linear_regression.py --run_experiments

Task 2 - Logistic Regression (Titanic):
    Single run:
        py task2_logistic_regression.py
    Run all experiments:
        py task2_logistic_regression.py --run_experiments

Output:
- Trained models are saved in saved_models/
- Plots and results are saved in results/task1/ and results/task2/
- Each experiment saves its plots in a separate subfolder
