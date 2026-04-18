#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("C:/Users/aksha/Documents/Python Tool box/new_insurance_data.csv")

# Display dataset preview
print(df.head())
print(df.tail())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Dataset information
print("\nDataset Info:")
print(df.info())


#Data Cleaning

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Remove duplicate rows
df.drop_duplicates(inplace=True)



#Exploratory Data Analysis(EDA)
# Outlier Detection using Boxplots

sns.boxplot(y=df['age'])
plt.title("Outliers in Age")
plt.show()

sns.boxplot(y=df['bmi'])
plt.title("Outliers in BMI")
plt.show()

sns.boxplot(y=df['children'])
plt.title("Outliers in Children")
plt.show()

sns.boxplot(y=df['charges'])
plt.title("Outliers in Charges")
plt.show()

# Remove outliers using IQR method
for col in df.columns:
    if df[col].dtype != 'object':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Set visualization style
sns.set_theme(style="whitegrid")


# Objective 1: Distribution of Insurance Charges

plt.figure(figsize=(8,5))
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.show()


# Objective 2: BMI vs Charges

plt.figure(figsize=(8,5))

sns.scatterplot(
    x='bmi',
    y='charges',
    data=df,
    color='navy',
    s=60
)

plt.title("BMI vs Insurance Charges")
plt.xlabel("Body Mass Index (BMI)")
plt.ylabel("Insurance Charges")

plt.grid(True)

plt.show()


# Objective 3: Number of Children vs Charges

plt.figure(figsize=(8,5))
sns.barplot(x='children', y='charges', data=df, palette='viridis',ci=None)
plt.title("Impact of Number of Children on Insurance Charges")
plt.xlabel("Number of Children")
plt.ylabel("Insurance Charges")
plt.show()



# Objective 4: Smoking Status vs Charges

plt.figure(figsize=(8,5))
sns.boxplot(x='smoker_yes', y='charges', data=df,palette='coolwarm')
plt.title("Impact of Smoking on Insurance Charges")
plt.xlabel("Smoking Status (0 = No, 1 = Yes)")
plt.ylabel("Insurance Charges")
plt.show()



# Objective 5: Correlation Analysis

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# Model Building: Linear Regression

# Define features and target
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = df.drop(columns = ['charges'])
y = df['charges']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.70,random_state = 42)

my_model = LinearRegression()

# training
my_model.fit(x_train,y_train)

# testing
y_pred = my_model.predict(x_test)

y_pred

result = pd.DataFrame(columns = ['Actual_value',"Predicted_value"])
result['Actual_value'] = y_test
result['Predicted_value'] = y_pred

result

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

result = r2_score(y_test, y_pred)
print(result)

mae = mean_absolute_error(y_test, y_pred)
print(mae)

mse = mean_squared_error(y_test, y_pred)
print(mse)

# rmse
rmse = np.sqrt(mse)
rmse

sns.regplot(x=y_pred,y = y_test)
plt.xlabel("Predicted charges")
plt.ylabel("Actual charges")
plt.title("Actual vs Predicted Insurance Charges(Linear Regression)")
plt.show()