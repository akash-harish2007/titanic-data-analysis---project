import pandas as pd
import matplotlib.pyplot as plt

# # Load the dataset
df = pd.read_csv('Data/train.csv')

# # Display the first few rows of the dataset
# print(df.head())

# # Display summary statistics of the dataset
# print(df.describe())

# #check for missing values
# print(df.isnull().sum())

# # Visualize the distribution of the target variable
# plt.figure(figsize=(8, 6))

# #check info about the dataset
# print(df.info())

# #check for duplicate rows
# print(df.duplicated().sum())

# #checking unique values in each column
# for column in df.columns:
#     print(f"Unique values in {column}: {df[column].nunique()}")
    
    
# #checking survival rate of passengers based on gender
# gender_survival = df.groupby('Sex')['Survived'].mean()

# print(gender_survival)

# gender_survival.plot(kind='bar', color=['pink', 'blue'])
# plt.xlabel('Sex')
# plt.ylabel('Survival Rate')
# plt.title('Survival Rate by Gender')
# plt.xticks(rotation=0)
# plt.show()

##checking survival rate of passengers based on passenger class
# class_survival = df.groupby('Pclass')['Survived'].mean()

# print(class_survival)

# class_survival.plot(kind='bar', color=['green', 'orange', 'red'])
# plt.xlabel('Passenger Class')
# plt.ylabel('Survival Rate')
# plt.title('Survival Rate by Passenger Class')
# plt.xticks(rotation=0)
# plt.show()


# #checking survival rate of passengers based on age groups
# df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
# age_group_survival = df.groupby('AgeGroup')['Survived'].mean()

# print(age_group_survival)

# age_group_survival.plot(kind='bar', color=['cyan', 'magenta', 'yellow', 'blue', 'gray'])
# plt.xlabel('Age Group')
# plt.ylabel('Survival Rate')
# plt.title('Survival Rate by Age Group')
# plt.xticks(rotation=45)
# plt.show()

# #checking survival rate of passengers based on fare

# df['Fare_Category'] = pd.cut(df['Fare'], 
#                              bins=[0, 50, 100, 200, 600],
#                              labels=['Very Low', 'Low', 'Medium', 'High'])

# # Calculate survival rate by category
# fare_category_survival = df.groupby('Fare_Category')['Survived'].mean() * 100
# print(fare_category_survival)
# fare_category_survival.plot(kind='bar', color=['lightcoral', 'lightsalmon', 'lightseagreen', 'lightsteelblue'])
# plt.xlabel('Fare Category')
# plt.ylabel('Survival Rate (%)')
# plt.title('Survival Rate by Fare Category')
# plt.xticks(rotation=45)
# plt.show()


# #checking survival rate of passengers based on number of embarked class

# embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
# print(embarked_survival)

# embarked_survival.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
# plt.xlabel('Port of Embarkation')
# plt.ylabel('Survival Rate (%)')
# plt.title('Survival Rate by Port of Embarkation')
# plt.xticks(rotation=0)
# plt.show()


# #checking survival rate of passengers based on number of siblings/spouses aboard

# sibsp_survival = df.groupby('SibSp')['Survived'].mean() * 100
# print(sibsp_survival)

# sibsp_survival.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
# plt.xlabel('Number of Siblings/Spouses Aboard')
# plt.ylabel('Survival Rate (%)')
# plt.title('Survival Rate by Number of Siblings/Spouses Aboard')
# plt.xticks(rotation=0)
# plt.show()