# Employee Salary Prediction System

## Project Overview
The Employee Salary Prediction System is a Machine Learning project developed using Python. The system predicts an employee's estimated annual salary based on factors such as age, education level, years of experience, job role, skills score, certifications, company type, and work hours per week.

This project demonstrates the complete machine learning workflow, including synthetic data generation, data preprocessing, exploratory data analysis (EDA), visualization, model training, evaluation, and salary prediction.

---

## Objective

The main objective of this project is to build a **Machine Learning model** that predicts an employee's annual salary based on employee-related attributes. This project provides practical experience in **data preprocessing, exploratory data analysis (EDA), model training, evaluation,** and **salary prediction** using Python.

---

## Features

* Generated a **synthetic dataset** containing **2,000 employee records**
* Performed **data preprocessing** by handling missing values, removing duplicate records, encoding categorical features, and validating the dataset
* Conducted **Exploratory Data Analysis (EDA)**
* Created **data visualizations** using **Matplotlib** and **Seaborn**
* Built a **Linear Regression** model for salary prediction
* Evaluated the model using **MAE**, **RMSE**, and **R² Score**
* Developed an **interactive salary prediction system**
* Saved the trained model using **Joblib**

---

## Dataset

The dataset contains the following attributes:

* EmployeeID
* Age
* Gender
* EducationLevel
* YearsOfExperience
* JobRole
* SkillsScore
* Certifications
* CompanyType
* WorkHoursPerWeek
* Salary (**Target Variable**)

The dataset was synthetically generated using **NumPy** and **Pandas** with realistic salary variations based on **experience, education, skills, certifications,** and **company type**.

---

## Technologies Used

* **Python**
* **Jupyter Notebook**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **GitHub**

---

## Project Workflow

1. Imported the required Python libraries.
2. Generated a **synthetic employee dataset** containing **2,000 employee records**.
3. Performed **data preprocessing** by handling missing values, removing duplicate records, encoding categorical features, and validating the dataset.
4. Conducted **Exploratory Data Analysis (EDA)**.
5. Created visualizations to understand salary trends and feature relationships.
6. Split the dataset into **training** and **testing** sets.
7. Trained the **Linear Regression** model.
8. Evaluated the model using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
9. Saved the trained model using **Joblib**.
10. Built an **interactive salary prediction system** for predicting employee salaries.

---

## Data Visualizations

The project includes the following visualizations:

* Experience vs Salary Scatter Plot
* Average Salary by Job Role Bar Chart
* Correlation Heatmap
* Salary Distribution by Education Level Box Plot

---

## Machine Learning Model

The following regression algorithm was implemented:

* **Linear Regression**

The model was evaluated using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### Model Performance

| Evaluation Metric | Result |
|-------------------|--------|
| **Mean Absolute Error (MAE)** | **14524.13** |
| **Root Mean Squared Error (RMSE)** | **16862.35** |
| **R² Score** | **0.9969** |

The **Linear Regression** model achieved excellent prediction performance on the generated dataset.

---

## Project Structure

```text
Employee-Salary-Prediction/
│
├── data/
│   └── employee_salary_dataset.csv
│
├── images/
│   ├── bar_chart.png
│   ├── boxplot.png
│   ├── heatmap.png
│   └── scatter_plot.png
│
├── notebooks/
│   └── Employee_Salary_Prediction.ipynb
│
├── Employee_Salary_Prediction.py
├── salary_prediction_model.pkl
├── requirements.txt
└── README.md
```



---

## How to Run

1. Clone or download this repository.
2. Install the required libraries using **requirements.txt**.
3. Open **Employee_Salary_Prediction.ipynb** in **Jupyter Notebook**.
4. Run all notebook cells in sequence.
5. Enter the required employee details in the **interactive prediction section**.
6. View the **predicted employee salary**.

---

## Key Insights

* Employee salary generally **increases with years of experience**.
* Employees with **higher education levels** tend to receive higher salaries.
* A higher **Skills Score** positively influences salary prediction.
* Employees with **more certifications** generally earn higher salaries.
* **Job role** and **company type** also influence employee salary.

---

## Future Enhancements

* Develop a **Streamlit** web application.
* Use a **real-world employee salary dataset**.
* Compare different **Machine Learning algorithms**.
* Improve prediction accuracy through model optimization.
* Deploy the project as a web application.

---

## Author

**Seelapureddy Harshini Reddy**

**Machine Learning Virtual Internship Project**
