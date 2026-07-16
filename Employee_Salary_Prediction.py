#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Display plots inside Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

print("All libraries imported successfully!")


# In[2]:


# Generate Synthetic Employee Dataset

np.random.seed(42)

# Number of employee records
num_records = 2000

# Generate employee data
employee_data = pd.DataFrame({
    "EmployeeID": range(1, num_records + 1),

    "Age": np.random.randint(21, 60, num_records),

    "Gender": np.random.choice(
        ["Male", "Female"],
        num_records
    ),

    "EducationLevel": np.random.choice(
        ["Bachelor", "Master", "PhD"],
        num_records,
        p=[0.5, 0.35, 0.15]
    ),

    "YearsOfExperience": np.random.randint(0, 36, num_records),

    "JobRole": np.random.choice(
        ["Developer", "Data Analyst", "Manager", "Designer", "Tester"],
        num_records
    ),

    "SkillsScore": np.random.randint(1, 11, num_records),

    "Certifications": np.random.randint(0, 11, num_records),

    "CompanyType": np.random.choice(
        ["Startup", "Medium Scale", "MNC"],
        num_records
    ),

    "WorkHoursPerWeek": np.random.randint(35, 61, num_records)
})

print("Employee data generated successfully!")


# In[4]:


salary = []

for i in range(num_records):

    base_salary = 250000

    # Experience
    experience_bonus = employee_data.loc[i, "YearsOfExperience"] * 30000

    # Skills
    skills_bonus = employee_data.loc[i, "SkillsScore"] * 15000

    # Certifications
    certification_bonus = employee_data.loc[i, "Certifications"] * 10000

    # Education
    education = employee_data.loc[i, "EducationLevel"]

    if education == "Bachelor":
        education_bonus = 50000
    elif education == "Master":
        education_bonus = 120000
    else:
        education_bonus = 200000

    # Company Type
    company = employee_data.loc[i, "CompanyType"]

    if company == "Startup":
        company_bonus = 30000
    elif company == "Medium Scale":
        company_bonus = 70000
    else:
        company_bonus = 120000

    # Random variation
    noise = np.random.randint(-30000, 30001)

    total_salary = (
        base_salary
        + experience_bonus
        + skills_bonus
        + certification_bonus
        + education_bonus
        + company_bonus
        + noise
    )

    salary.append(total_salary)

employee_data["Salary"] = salary

print("Salary column created successfully!")


# In[5]:


employee_data.to_csv(
    "../data/employee_salary_dataset.csv",
    index=False
)

print("Dataset saved successfully!")

employee_data.head()


# In[6]:


print("Dataset Shape:", employee_data.shape)


# In[7]:


employee_data.info()


# In[8]:


employee_data.isnull().sum()


# In[9]:


print("Duplicate Records:", employee_data.duplicated().sum())


# In[10]:


employee_data = employee_data.drop_duplicates()

print("Duplicates Removed Successfully!")


# In[11]:


# Statistical Summary

employee_data.describe()


# In[13]:


# Average Salary by Education

employee_data.groupby("EducationLevel")["Salary"].mean()


# In[14]:


# Average Salary by Job Role

employee_data.groupby("JobRole")["Salary"].mean()


# In[15]:


# Average Salary by Company Type

employee_data.groupby("CompanyType")["Salary"].mean()


# In[16]:


# Scatter Plot: Experience vs Salary

plt.figure(figsize=(8,5))

plt.scatter(
    employee_data["YearsOfExperience"],
    employee_data["Salary"],
    alpha=0.6
)

plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.savefig("../images/scatter_plot.png")

plt.show()


# In[17]:


# Average Salary by Job Role

plt.figure(figsize=(8,5))

employee_data.groupby("JobRole")["Salary"].mean().plot(kind="bar")

plt.title("Average Salary by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Average Salary")

plt.savefig("../images/bar_chart.png")

plt.show()


# In[18]:


# Correlation Heatmap

plt.figure(figsize=(10,6))

corr = employee_data.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.savefig("../images/heatmap.png")

plt.show()


# In[19]:


# Salary Distribution by Education Level

plt.figure(figsize=(8,5))

sns.boxplot(
    x="EducationLevel",
    y="Salary",
    data=employee_data
)

plt.title("Salary Distribution by Education Level")

plt.savefig("../images/boxplot.png")

plt.show()


# In[20]:


# Convert Categorical Data into Numerical Data

employee_data_ml = pd.get_dummies(
    employee_data,
    columns=["Gender", "EducationLevel", "JobRole", "CompanyType"],
    drop_first=True
)

print("Categorical columns converted successfully!")

employee_data_ml.head()


# In[21]:


# Features and Target

X = employee_data_ml.drop(["EmployeeID", "Salary"], axis=1)

y = employee_data_ml["Salary"]

print("Features Shape :", X.shape)
print("Target Shape :", y.shape)


# In[22]:


# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)


# In[23]:


# Linear Regression Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully!")


# In[24]:


# Predict Salary

y_pred = model.predict(X_test)

print("Prediction Completed!")


# In[25]:


# Model Evaluation

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae,2))
print("Root Mean Squared Error (RMSE):", round(rmse,2))
print("R² Score:", round(r2,4))


# In[26]:


# Take User Input

age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
education = input("Enter Education Level (Bachelor/Master/PhD): ")
experience = int(input("Enter Years of Experience: "))
job_role = input("Enter Job Role (Developer/Data Analyst/Manager/Designer/Tester): ")
skills = int(input("Enter Skills Score (1-10): "))
certifications = int(input("Enter Number of Certifications: "))
company = input("Enter Company Type (Startup/Medium Scale/MNC): ")
work_hours = int(input("Enter Work Hours Per Week: "))


# In[27]:


# Create User Data

user_data = pd.DataFrame({
    "Age": [age],
    "YearsOfExperience": [experience],
    "SkillsScore": [skills],
    "Certifications": [certifications],
    "WorkHoursPerWeek": [work_hours],
    "Gender": [gender],
    "EducationLevel": [education],
    "JobRole": [job_role],
    "CompanyType": [company]
})


# In[28]:


# Convert User Data

user_data = pd.get_dummies(user_data)

# Match training columns
user_data = user_data.reindex(columns=X.columns, fill_value=0)

user_data


# In[29]:


# Predict Salary

predicted_salary = model.predict(user_data)

print(f"\nPredicted Employee Salary: ₹{predicted_salary[0]:,.2f}")


# In[30]:


# Save Model

import joblib

joblib.dump(model, "../salary_prediction_model.pkl")

print("Model saved successfully!")


# In[ ]:




