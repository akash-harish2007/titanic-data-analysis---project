# 🚢 Titanic Survival Analysis

## 📋 Project Overview

This project analyzes the Titanic passenger dataset to understand what factors most influenced a person's chances of survival. Using Python and pandas, I explored patterns in the data, created visualizations, and uncovered key insights about who survived and why.

The analysis covers **6 key factors**:
- Gender
- Passenger Class
- Age
- Fare
- Embarkation Port
- Family Size

## 🎯 Main Goal

**To identify the key factors that determined survival** by analyzing passenger characteristics and uncovering patterns in the data.

## 📊 Dataset

- **Source**: Kaggle Titanic competition
- **Training Data**: 891 passengers, 12 features
- **Test Data**: 418 passengers, 11 features
- **Target Variable**: `Survived` (0 = No, 1 = Yes)

### Columns

| Column | Description |
|--------|-------------|
| `Survived` | Survival (0=No, 1=Yes) |
| `Pclass` | Ticket class (1=1st, 2=2nd, 3=3rd) |
| `Sex` | Gender (male/female) |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton) |

## 🔍 Key Findings

| Factor | Most Likely to Survive | Survival Rate | Least Likely to Survive | Survival Rate |
|--------|------------------------|---------------|-------------------------|---------------|
| **Gender** | Female | **74.2%** | Male | **18.9%** |
| **Class** | 1st Class | **63.0%** | 3rd Class | **24.2%** |
| **Age** | Child (0-12) | **58.0%** | Senior (60+) | **22.7%** |
| **Fare** | High Fare | **70.0%** | Very Low Fare | **32.4%** |
| **Embarkation** | Cherbourg (C) | **55.4%** | Southampton (S) | **33.7%** |
| **Family** | 1 family member | **53.6%** | Alone | **34.5%** |

### The "Ideal Survivor" Profile

Based on the analysis, the passenger with the **highest chance of survival** was:
- ✅ **Female**
- ✅ **1st Class**
- ✅ **Child** (under 12)
- ✅ **High Fare**
- ✅ **Boarded at Cherbourg**
- ✅ **Traveling with 1 family member**

### The "Least Likely to Survive" Profile

The passenger with the **lowest chance of survival** was:
- ❌ **Male**
- ❌ **3rd Class**
- ❌ **Senior** (60+)
- ❌ **Very Low Fare**
- ❌ **Boarded at Southampton**
- ❌ **Traveling Alone**

## 📈 Visualizations

All visualizations are saved in the `outputs/charts/` folder:

1. **Survival by Gender** - Bar chart showing female vs male survival
2. **Survival by Class** - Bar chart showing 1st, 2nd, and 3rd class survival
3. **Survival by Age Group** - Bar chart showing survival across age groups
4. **Survival by Fare Category** - Bar chart showing fare impact on survival
5. **Survival by Embarkation Port** - Bar chart showing port impact
6. **Survival by Family Size** - Bar chart showing family size impact

## 🛠️ Technologies Used

- **Python 3.9+** - Programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive analysis environment

  
## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-data-analysis.git
cd titanic-data-analysis
