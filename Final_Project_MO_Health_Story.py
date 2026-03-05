#!/usr/bin/env python
# coding: utf-8

# # Module 8 - Principles of Data Science - Final Project Story
#
# You have made it to the end of the course, and you have worked hard to develop your DSA perspectives and skills.  So far we have been internally focused on the operations of performing data science and analytics.  Now we will extend our work to the development of a data story that is externally focused.
#
# In the Module8 labs, you saw simplified examples of constructing data stories. Throughout the course, there are components and parts useful to consider as a basis for developing a short, unique, focused data story.
#
#
# For this final project, you will
#
# - Step 0: Choose your Language for this Adventure
#
# - Step 1: Find a Story
#
# - Step 2: Remember your Audience
#
# - Step 3: Find and Stage Your Data
#
# - Step 4: Vet Data Sources
#
# - Step 5: Filter Results and Build/Validate Models
#
# - Step 6: Visualize Results
#
# - Step 7: Communicate the Story to your intended audience using visualizations and narratives
#
# - Final Step: Connect your workflow/process to the DSA-Project Life Cycle
#
# ---
# Here are some recommendations for managing the scope and quality of this project:
#
# - Narrow down the issue, problem, question, or hypothesis for your data story to a single, relatively simple perspective.
#
# - Identify what aspects or attributes from the provided data that affords addressing your problem.  If incorporating another, completely new, data set - explain it as well.
#
# - Address the data relative to the statistical/machine learning model(s) chosen to minimize any issues.
#
# - Internally document your code using comments and markdown cells that explain the purpose of the operation(s) as well as interpreting the results of those operations.
#
#
# You can make your project more unique by:
#
# - Comparing two or more different statistical / machine learning models using the same data.
# - Refrain from identically replicating any existing projects obtained from external sources or in class collaborations - this should be your idea and analysis!
# - Running a single model multiple times and changing a different single parameter each time for comparison.
# - Changing the sampling proportions for building the hold-out data and comparing the same model performance repeatedly.
# - Select something you find interesting or unique in the data and write an analytical story around it.
#
#
#

# ## Step 0: Choose your Language for this Adventure:
#
# You can do this project in either *R* or *Python*.
#
# To change the kernel of this notebook, do the following with the `Kernel` menu.
#
#  * `Kernel > Change Kernel > Python 3`
#  * `Kernel > Change Kernel > R`
#
# ![FP_Change_Kernel.png MISSING](../images/FP_Change_Kernel.png)
#

# ---
# ## Step 1: Find a Story
#
# Think about the data file that has been provided for this project to use in this class.
# Additionally, you can search online for potential data to incorporate **with** the provided data to support the story idea.
#
# In the cell below, please detail the source of any additional data (with link) as welll as preview the data story you hope to uncover.

#
#
#
#

# ## Step 2: Remember your Audience
#
# In the cell below, describe the audience for this analysis!
#  * Who will the audience be?
#  * What value will they derive from your story?

#
#
#
#
#
#
#

# ## Step 3: Find and Stage Your Data
#
# If you incorporate data from another source, you must download it to your local computer, then upload the data to JuptyerHub.
#
# #### If you are uploading files:
#  * Use folder navigation of your first JupyterTab to get to course's `/modules/module8/exercises/` folder.
# ![FP_Folder_Navigation.png MISSING](../images/FP_Folder_Navigation.png)
#  * Click the Upload Button and Choose File(s)
# ![FP_Upload_Button.png MISSING](../images/FP_Upload_Button.png)
#  * Activate the upload
# ![FP_UploadFile_2.png MISSING](../images/FP_UploadFile_2.png)
#
#
# ### In the cell below, please list the name(s) of the file(s) that are now accessible on the JupyterHub environment.
#
# **Note**
# If you uploaded a file to your `module8/exercises` folder, the file name is all you need to load it into the data frame in the usual manner.
# If you are using a file from another module of the course, you should be able to copy the full pathname and use it as is in this notebook.
# The full pathname has been provided for the Missouri County-based data that has been provided.  Be sure to include it below as well!!!

get_ipython().system("pip install pandas")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Load the dataset
df = pd.read_csv("MO_County_Data_CSV_DSA7010_Project_Data.csv")

# Display the first few rows of the dataframe
df.head()


# ## Step 4: Explore Data Sources
#
# Use the cells below to load the data, inspect it, conduct data carpentry and shaping; perform exploratory data analysis.
#
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.  Also add cells that help visualize your exploration and what you found and how those findings directed your story.

# Display information about the dataset (columns, data types, non-null values)
print("\nDataset Info:")
print(df.info())


# Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())


# List all the column names

df.columns


# Get the unique values in each categorical column
categorical_columns = df.select_dtypes(include=["object"]).columns
print("\nUnique values in categorical columns:")
for col in categorical_columns:
    print(f"\nColumn '{col}':")
    print(df[col].unique())


# Selecting relevant columns for analysis
columns_of_interest = [
    "cnty_name",
    "povpct",
    "P_64UnEmrt",
    "pct_unempl",
    "E_EDUC_LTH",
    "PctUninsur",
    "AllC_AAIRt",
    "mentalhp_p",
    "pricarephy",
    "PctDisable",
    "LeisTime",
    "food_insec",
    "Work_PubTran",
    "TotCardDis",
    "mentalhp_t",
    "E_EDUC_HS",
    "P_64UnEmrt",
    "GPwChu18Pct",
    "female_pct",
    "urban_pct",
    "PctAfAmer",
    "PctHispani",
    "PctDisable",
    "PctAMam",
    "PctFluVac",
    "PctPapP3yr",
    "PctColScr",
    "AvePM2p5",
    "Obesity",
    "Diabetes",
    "StrokeHosp",
    "Age65plus",
    "PrvHspRate",
    "Hyperten",
    "HeartDis",
    "PctUnins65",
    "NurseHmBeds",
    "PopDensqmi",
    "P_BPovLev",
]


columns_of_interest


# Subset the DataFrame to keep only the columns of interest
df_filtered = df[columns_of_interest]

# Display the first few rows
print("First 5 rows of the filtered data:")
print(df_filtered.head())

# Display information about the filtered dataset
print("\nFiltered Dataset Info:")
print(df_filtered.info())

# Check for missing values
print("\nMissing values in filtered dataset:")
print(df_filtered.isnull().sum())


# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df_filtered.describe())


# Mapping column names to descriptive names based on the metadata
column_name_mapping = {
    "povpct": "Poverty Percentage",
    "P_64UnEmrt": "Unemployment Rate for Ages 16-64",
    "pct_unempl": "Overall Unemployment Percentage",
    "E_EDUC_LTH": "Less than High School Education",
    "PctUninsur": "Percentage Uninsured",
    "AllC_AAIRt": "All-Cause Age-Adjusted Mortality Rate",
    "mentalhp_p": "Mental Health Provider Rate",
    "pricarephy": "Primary Care Physicians",
    "PctDisable": "Percentage Disabled",
    "LeisTime": "Leisure Time Physical Inactivity",
    "food_insec": "Food Insecurity Rate",
    "Work_PubTran": "Public Transportation Use",
    "TotCardDis": "Total Cardiovascular Disease Rate",
    "mentalhp_t": "Mental Health Treatment Rate",
    "E_EDUC_HS": "High School Education",
    "GPwChu18Pct": "Grandparents Living with Children Under 18",
    "female_pct": "Percentage Female",
    "urban_pct": "Urban Percentage",
    "PctAfAmer": "Percentage African American",
    "PctHispani": "Percentage Hispanic",
    "PctAMam": "Percentage American Indian/Alaska Native",
    "PctFluVac": "Percentage Flu Vaccinated",
    "PctPapP3yr": "Percentage Pap Smear Last 3 Years",
    "PctColScr": "Percentage Colon Cancer Screening",
    "AvePM2p5": "Average PM2.5 (Air Quality)",
    "Obesity": "Obesity Rate",
    "Diabetes": "Diabetes Rate",
    "StrokeHosp": "Stroke Hospitalization Rate",
    "Age65plus": "Population Aged 65+",
    "PrvHspRate": "Preventable Hospitalization Rate",
    "Hyperten": "Hypertension Rate",
    "HeartDis": "Heart Disease Rate",
    "PctUnins65": "Percentage Uninsured Age 65+",
    "NurseHmBeds": "Nursing Home Beds",
    "PopDensqmi": "Population Density per Square Mile",
    "P_BPovLev": "Percentage Below Poverty Level",
}


# Rename the columns with descriptive names
df_filtered_copy.rename(columns=column_name_mapping, inplace=True)

# Plotting histograms for a few continuous variables with descriptive names
continuous_columns = [
    "Poverty Percentage",
    "Percentage Uninsured",
    "Obesity Rate",
    "Diabetes Rate",
    "Population Density per Square Mile",
]

plt.figure(figsize=(14, 8))
for i, col in enumerate(continuous_columns):
    plt.subplot(2, 3, i + 1)

    # Check if seaborn has histplot, otherwise use distplot
    if hasattr(sns, "histplot"):
        sns.histplot(df_filtered_copy[col], kde=True, bins=20)
    else:
        sns.distplot(df_filtered_copy[col], kde=True, bins=20)

    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# Box plot for comparing Poverty Percentage by Urban Percentage
plt.figure(figsize=(30, 6))
sns.boxplot(x="Urban Percentage", y="Poverty Percentage", data=df_filtered)
plt.title("Poverty Percentage by Urban Percentage")
plt.show()


# Line chart for comparing female percentage across counties
plt.figure(figsize=(30, 10))
sns.lineplot(x="cnty_name", y="Percentage Female", data=df_filtered, marker="o")
plt.xticks(rotation=90)
plt.title("Percentage Female by County")
plt.xlabel("County")
plt.ylabel("Percentage Female")
plt.grid(True)
plt.show()

# Line chart for comparing urban percentage across counties
plt.figure(figsize=(30, 10))
sns.lineplot(
    x="cnty_name", y="Urban Percentage", data=df_filtered, marker="o", color="orange"
)
plt.xticks(rotation=90)
plt.title("Urban Percentage by County")
plt.xlabel("County")
plt.ylabel("Urban Percentage")
plt.grid(True)
plt.show()


# Box plot for comparing diabetes rates by urban percentage
plt.figure(figsize=(50, 20))
sns.boxplot(x="Urban Percentage", y="Diabetes Rate", data=df_filtered)
plt.title("Diabetes Rate by Urban Percentage")
plt.show()

# Box plot for comparing obesity rates by urban percentage
plt.figure(figsize=(50, 15))
sns.boxplot(x="Urban Percentage", y="Obesity Rate", data=df_filtered)
plt.title("Obesity Rate by Urban Percentage")
plt.show()


# Scatter plot of Diabetes Rate vs Obesity Rate
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Obesity Rate", y="Diabetes Rate", data=df_filtered)
plt.title("Diabetes Rate vs Obesity Rate")
plt.show()

# Scatter plot of Mental Health Provider Rate vs Population Density
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Population Density per Square Mile",
    y="Mental Health Provider Rate",
    data=df_filtered,
)
plt.title("Mental Health Provider Rate vs Population Density")
plt.show()


# Distribution plot for Poverty Percentage
plt.figure(figsize=(8, 6))
sns.kdeplot(df_filtered["Poverty Percentage"], shade=True, color="blue")
plt.title("Distribution of Poverty Percentage")
plt.show()

# Distribution plot for Percentage Uninsured
plt.figure(figsize=(8, 6))
sns.kdeplot(df_filtered["Percentage Uninsured"], shade=True, color="green")
plt.title("Distribution of Percentage Uninsured")
plt.show()


# Heatmap for visualizing health-related variables across counties
plt.figure(figsize=(16, 10))
health_variables = [
    "Obesity Rate",
    "Diabetes Rate",
    "Heart Disease Rate",
    "Hypertension Rate",
]
sns.heatmap(df_filtered[health_variables].T, cmap="YlGnBu", linewidths=1)
plt.title("Health Indicators Across Counties")
plt.show()


# Heatmap for visualizing health-related variables across counties with county names
plt.figure(figsize=(50, 20))
health_variables = [
    "Obesity Rate",
    "Diabetes Rate",
    "Heart Disease Rate",
    "Hypertension Rate",
]

# Create the heatmap using county names on the x-axis
sns.heatmap(
    df_filtered.set_index("cnty_name")[health_variables].T,
    cmap="YlGnBu",
    linewidths=1,
    annot=True,
)
plt.title("Health Indicators Across Counties")
plt.xlabel("County")
plt.ylabel("Health Indicators")
plt.show()


# Pairplot for Poverty Percentage, Unemployment, and Obesity Rate
pairplot_columns = [
    "Poverty Percentage",
    "Percentage Uninsured",
    "Obesity Rate",
    "Diabetes Rate",
]
sns.pairplot(df_filtered[pairplot_columns])
plt.show()


# Melt the dataframe to allow for easy plotting of multiple metrics
df_filtered_melt = df_filtered.melt(
    id_vars=["cnty_name"],
    value_vars=[
        "Poverty Percentage",
        "Obesity Rate",
        "Diabetes Rate",
        "Heart Disease Rate",
    ],
)

# Line chart for health and demographic metrics by county
plt.figure(figsize=(26, 8))
sns.lineplot(
    x="cnty_name", y="value", hue="variable", data=df_filtered_melt, marker="o"
)

# Customize the x-axis and chart
plt.xticks(rotation=90)
plt.title("Health and Demographic Metrics by County")
plt.xlabel("County")
plt.ylabel("Metric Value")
plt.grid(True)
plt.legend(title="Metrics", loc="upper right")
plt.show()


# Set the figure size
plt.figure(figsize=(26, 8))

# Bar chart for Poverty Percentage
sns.barplot(
    x="cnty_name",
    y="Poverty Percentage",
    data=df_filtered,
    color="blue",
    label="Poverty Percentage",
)

# Bar chart for Percentage Uninsured (overlaid)
sns.barplot(
    x="cnty_name",
    y="Percentage Uninsured",
    data=df_filtered,
    color="orange",
    label="Percentage Uninsured",
    alpha=0.7,
)

# Customize the x-axis and chart
plt.xticks(rotation=90)
plt.title("Poverty Percentage and Uninsured Percentage by County")
plt.xlabel("County")
plt.ylabel("Percentage")
plt.legend(title="Metrics", loc="upper right")
plt.grid(True)
plt.show()


# Select only the numeric columns from the filtered DataFrame
df_numeric = df_filtered.select_dtypes(include=["float64", "int64"])

# Visualize the correlation matrix for the numeric columns
plt.figure(figsize=(15, 10))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Selected Columns")
plt.show()


# ## Step 5: Filter Results and Build and Validate Models
#
#
# Perform any additional data carpentry and begin filtering results/data and then build, validate, and describe your model(s). Make sure to provide interpretations of results and analytics!!!
#
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.

# List of predictors
predictors = [
    "pop_2014",
    "age_18_65",
    "age_gt_65",
    "povpct",
    "E_Emp_64",
    "E_MEDINC",
    "PEmpStat_6",
    "WOHlthIns",
    "limit_food",
    "SmokePrev2",
    "SmokeROC96",
    "Work_AtHome",
    "age_lt18",
    "HspanyRace",
    "AfrAmer",
    "PctAfAmer",
    "InsTpop",
    "PInsAfAm",
    "PInsHisp",
    "E_pop18up",
]

# List of target variables
targets = [
    "pop_densit",
    "Obesity",
    "Diabetes",
    "HeartDis",
    "Hyperten",
    "StrokeHosp",
    "E_POV_STAT",
    "StrokeHosp",
    "TotCardDis",
]


# Select health-related columns
healthpov_columns = [
    "Obesity",
    "Diabetes",
    "HeartDis",
    "Hyperten",
    "povpct",
    "E_Emp_64",
    "PEmpStat_6",
    "WOHlthIns",
    "limit_food",
    "E_MEDINC",
    "StrokeHosp",
    "TotCardDis",
]  # Add mental health column if available

# Display basic statistics of health-related variables
print(df[healthpov_columns].describe())


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Define a function to build and evaluate models for multiple targets
def build_and_evaluate_models(df, predictors, targets):
    models = {}
    results = {}

    # Looping through each target variable
    for target in targets:
        # Define features (X) and target (y)
        X = df[predictors]
        y = df[target]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Build the regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Store the model and predictions
        models[target] = model
        results[target] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        # Print the evaluation metrics
        print(f"Results for {target.replace('_', ' ').capitalize()}:")
        print(f"Mean Squared Error: {results[target]['mse']}")
        print(f"R-squared: {results[target]['r2']}\n")

    return models, results


# Build models and evaluate them for each target variable
models, results = build_and_evaluate_models(df, predictors, targets)


import matplotlib.pyplot as plt
import pandas as pd


# Defining a function to visualize the actual vs. predicted values in descending order by actual values
def visualize_predictions(results, target, county_names):
    # Extract the actual and predicted values
    y_test = results[target]["y_test"]
    y_pred = results[target]["y_pred"]

    # Creating a DataFrame with actual and predicted values and include county names
    predicted_df = pd.DataFrame(
        {"County": county_names[y_test.index], "Actual": y_test, "Predicted": y_pred}
    )

    # Sort the DataFrame by actual values in descending order
    predicted_df = predicted_df.sort_values(by="Actual", ascending=False)

    # Create a bar plot comparing actual and predicted values for all counties
    predicted_df.plot(
        x="County",
        y=["Actual", "Predicted"],
        kind="bar",
        figsize=(14, 8),
        color=["blue", "orange"],
    )

    # Set the plot title and labels
    plt.title(
        f'Actual vs Predicted {target.replace("_", " ").capitalize()} (All Counties, Descending Order)'
    )
    plt.ylabel(f'{target.replace("_", " ").capitalize()} Rate')
    plt.xlabel("County")
    plt.xticks(rotation=90)  # Rotate county names for better readability
    plt.tight_layout()
    plt.show()


# Assume 'county_names' is a list or series of county names corresponding to the y_test index
county_names = df["cnty_name"]

# Visualize predictions for each target variable
for target in targets:
    visualize_predictions(results, target, county_names)


import matplotlib.pyplot as plt
import pandas as pd


# Define a function to visualize the actual vs. predicted values and save the plot as an image
def visualize_predictions(results, target, county_names):
    # Extract the actual and predicted values
    y_test = results[target]["y_test"]
    y_pred = results[target]["y_pred"]

    # Create a DataFrame with actual and predicted values and include county names
    predicted_df = pd.DataFrame(
        {"County": county_names[y_test.index], "Actual": y_test, "Predicted": y_pred}
    )

    # Sort the DataFrame by actual values in descending order
    predicted_df = predicted_df.sort_values(by="Actual", ascending=False)

    # Create a bar plot comparing actual and predicted values for all counties
    predicted_df.plot(
        x="County",
        y=["Actual", "Predicted"],
        kind="bar",
        figsize=(14, 8),
        color=["blue", "orange"],
    )

    # Set the plot title and labels
    plt.title(
        f'Actual vs Predicted {target.replace("_", " ").capitalize()} (All Counties, Descending Order)'
    )
    plt.ylabel(f'{target.replace("_", " ").capitalize()} Rate')
    plt.xlabel("County")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot as an image file (PNG format)
    plt.savefig(f"actual_vs_predicted_{target}.png", format="png")
    plt.show()


# Assume 'county_names' is a list or series of county names corresponding to the y_test index
county_names = df[
    "cnty_name"
]  # Adjust 'cnty_name' to match the actual column name for county names

# Visualize predictions for each target variable and save the images
for target in targets:
    visualize_predictions(results, target, county_names)


# Define a function to visualize the actual vs. predicted values
def visualize_predictions(results, target):
    # Extract the actual and predicted values
    y_test = results[target]["y_test"]
    y_pred = results[target]["y_pred"]

    # Create scatter plot for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="b")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.title(f'Actual vs Predicted {target.replace("_", " ").capitalize()}')
    plt.xlabel(f'Actual {target.replace("_", " ").capitalize()}')
    plt.ylabel(f'Predicted {target.replace("_", " ").capitalize()}')
    plt.show()

    # Create bar plot comparing actual and predicted values (sorted by actual values)
    predicted_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    predicted_df = predicted_df.sort_values(by="Actual", ascending=False)

    predicted_df.plot(kind="bar", figsize=(12, 8))
    plt.title(f'Actual vs Predicted {target.replace("_", " ").capitalize()} (Bar Plot)')
    plt.ylabel(f'{target.replace("_", " ").capitalize()} Rate')
    plt.xticks(rotation=90)
    plt.show()


# Visualize predictions for each target variable
for target in targets:
    visualize_predictions(results, target)


# ## Step 6: Visualize Key Results
#
# Build up your **key** visual story elements!
#
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.

import matplotlib.pyplot as plt
import pandas as pd


# Defining a function to visualize the actual vs. predicted values in descending order by predicted values
def visualize_predictions(results, target, county_names):
    # Extract the actual and predicted values
    y_test = results[target]["y_test"]
    y_pred = results[target]["y_pred"]

    # Create a DataFrame with actual and predicted values and include county names
    predicted_df = pd.DataFrame(
        {"County": county_names[y_test.index], "Actual": y_test, "Predicted": y_pred}
    )

    # Sort the DataFrame by predicted values in descending order
    predicted_df = predicted_df.sort_values(by="Predicted", ascending=False)

    # Creating a bar plot comparing actual and predicted values for all counties
    predicted_df.plot(
        x="County",
        y=["Actual", "Predicted"],
        kind="bar",
        figsize=(14, 8),
        color=["blue", "orange"],
    )

    # Set the plot title and labels
    plt.title(
        f'Actual vs Predicted {target.replace("_", " ").capitalize()} (All Counties, Descending Order by Predicted Values)'
    )
    plt.ylabel(f'{target.replace("_", " ").capitalize()} Rate')
    plt.xlabel("County")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Assume 'county_names' is a list or series of county names corresponding to the y_test index
county_names = df["cnty_name"]

# Visualize predictions for each target variable
for target in targets:
    visualize_predictions(results, target, county_names)


# Sort the counties in descending order by predicted population density
sorted_counties = df.sort_values(by="predicted_pop_density", ascending=False)

# Visualize all counties by predicted population density in descending order (switch x and y axes)
plt.figure(figsize=(20, 12))
sns.barplot(
    x="cnty_name", y="predicted_pop_density", data=sorted_counties, palette="viridis"
)
plt.title("All Counties by Predicted Population Density (Descending Order)")
plt.ylabel("Predicted Population Density")
plt.xlabel("County Name")
plt.xticks(rotation=90)
plt.show()


# ## Step 7: Communicate the Story to your intended audience using visualizations and narrative
#
#
# Briefly describe the story you are trying to tell with these data. Make sure to address the "So What!" aspect of the investigation.  Why did you coose the topic for analysis? What made it unique? Why would you consider it a data science project?
#

# #  Answer Narrative of the Data Story
#
# In this analysis, we used predictive models to estimate future health-related issues and population growth across Missouri counties. The primary focus was to identify counties that are expected to experience significant health burdens in the near future, including high rates of cardiovascular diseases, obesity, diabetes, hypertension, stroke hospitalizations, and poverty. Using actual vs. predicted comparisons for these health metrics, we can clearly see which counties are likely to face worsening health conditions and require increased healthcare resources and intervention.
#
# The predictions reveal that certain counties, notably **Pulaski County**, **Taney County**, and **Washington County**, are predicted to experience alarming increases in health issues, while others, such as **Jackson County** and **Boone County**, show rapid population growth and rising poverty levels. These insights can help healthcare planners and policymakers allocate resources more effectively to the areas that need them the most.
#
# ---
#
# ## So What? Why Is This Important?
#
# This investigation is important because it allows decision-makers to be **proactive rather than reactive** in addressing health crises. The **"So What"** of this analysis lies in the ability to forecast where future health issues are likely to arise, thus allowing state and local health authorities to focus their efforts on those counties that will need the most support.
#
# ### The data show that:
# - **Pulaski County** is expected to face severe increases in cardiovascular disease, hypertension, diabetes, and other major health problems. This means that without proper intervention, the county could experience a surge in healthcare needs, which would strain local healthcare infrastructure.
# - **Taney County** and **Washington County** are similarly predicted to have major health burdens, which may indicate a need for new public health initiatives, outreach programs, and better access to healthcare services.
# - **Jackson County** and **Boone County**, on the other hand, are expected to see population booms and higher poverty rates, which could lead to greater disparities in healthcare access, worsening public health outcomes, and greater demand for affordable care.
#
# By understanding these trends, healthcare providers and policymakers can implement preventive strategies—such as expanding healthcare facilities, launching public health campaigns, and improving access to services—to mitigate the predicted increases in chronic diseases and health risks.
#
# ---
#
# ## Why Was This Topic Chosen for Analysis?
#
# This topic was chosen because healthcare planning and resource allocation are **critical in addressing public health needs**, especially in underserved areas. By predicting which counties will face the greatest health-related challenges, this project aims to support data-driven decision-making that improves community health outcomes and reduces disparities. The predictive power of data science can play a key role in helping governments and organizations plan ahead to address public health issues before they become full-blown crises.
#
# Additionally, this topic allows for the practical application of **machine learning techniques** to real-world issues, showcasing the power of data science to make meaningful, life-changing predictions. The unique combination of healthcare, demographic, and socioeconomic factors makes this a compelling data science project that highlights the intersection of public health and predictive analytics.
#
# ---
#
# ## Why Is This a Data Science Project?
#
# This is a classic data science project because it involves:
#
# - **Data Preparation**: Cleaning and preparing the data for analysis, including selecting relevant features (predictors) and target variables.
# - **Model Building**: Using machine learning (linear regression) to build models that predict various health outcomes.
# - **Model Evaluation**: Evaluating the model’s performance using metrics like Mean Squared Error and R-squared, and comparing actual vs. predicted values.
# - **Data Visualization**: Communicating the results through visualizations that highlight the key findings.
# - **Insight Generation**: Producing actionable insights that inform decision-making in public health and resource allocation.
#
# The project uniquely combines data science techniques with a real-world application that has the potential to improve lives by forecasting where and when health interventions will be needed.
#
# ---
#
# ## Final Message:
#
# The story that emerges from this analysis is clear: some counties in Missouri are heading towards **significant health challenges**, while others will see **rapid population growth** and **rising poverty**. The insights drawn from this project can help public health officials and policymakers prioritize resources and focus on **prevention and intervention** in counties that need it most. **Data science** is a powerful tool for transforming public health planning, allowing us to anticipate and mitigate future health crises before they happen.
#

# ## Step 8: Connect your workflow / process to the DSA-Project Life Cycle
# - List **each** stage and then briefly discuss how important details from the [DSA-PLC](../../module1/resources/DSA-ProjectLifecycle-slidedeck.pdf) played a role in your story development.
# - Use markdown to provide this overview below:
# <hr/>
#
# <h1 align="center"><u>DSA-Project Life Cycle Discussion</u></h1>
#
#

# # Answer: Connecting the Workflow/Process to the DSA-Project Life Cycle
#
# ## Stage 1: Project Definition
#
# **Goal**:
# Predict and compare actual vs. predicted values for various health factors (e.g., obesity, diabetes, heart disease) and population density across counties in Missouri. The project aims to highlight disparities in health outcomes and forecast future trends.
#
# **Problem Statement**:
# How well can we predict health-related variables and population density across counties in Missouri using regression models?
#
# ---
#
# ## Stage 2: Data Acquisition
#
# **Data Source**:
# The dataset used includes health-related variables such as obesity, diabetes, heart disease, and other factors like population density and poverty rates for different counties in Missouri.
#
# **Data Import**:
# The dataset was loaded into the Python environment for analysis, with columns corresponding to the different predictors and target variables.
#
# ---
#
# ## Stage 3: Data Preparation
#
# **Feature Selection**:
# The predictors (`pop_2014`, `age_18_65`, `age_gt_65`, `povpct`, etc.) and target variables (`Obesity`, `Diabetes`, `HeartDis`, `Hyperten`, etc.) were selected.
#
# **Data Splitting**:
# The data was split into training and testing sets using the `train_test_split` method to ensure that model performance could be validated.
#
# ---
#
# ## Stage 4: Exploratory Data Analysis (EDA)
#
# **EDA Techniques**:
# Descriptive statistics and correlation analysis were conducted to understand the relationships between the predictors and target variables.
#
# **Visualization**:
# Histograms and summary statistics provided insights into the distributions of health-related variables and population density.
#
# ---
#
# ## Stage 5: Model Building
#
# **Model Type**:
# Linear regression was chosen as the modeling approach for predicting each health-related target variable and population density.
#
# **Model Training**:
# Each model was trained using the selected predictors, and predictions were generated for the test set.
#
# ---
#
# ## Stage 6: Model Validation
#
# **Performance Metrics**:
# For each model, performance metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)** were calculated to evaluate how well the model predicted the target variable.
#
# **Model Performance**:
# Some models had poor performance, indicated by negative R-squared values, while others, such as the model for predicting poverty rates, showed very high accuracy.
#
# ---
#
# ## Stage 7: Data Storytelling
#
# **Visualization of Actual vs. Predicted Values**:
# The `visualize_predictions` function was used to create bar plots comparing actual vs. predicted values for each target variable. These visualizations were saved as image files and uploaded.
#
# **Narrative**:
# The comparison of actual and predicted values across counties highlights areas where the model performs well and areas where predictions deviate from actual values. These insights are crucial for identifying counties at higher risk of poor health outcomes.
#
# ---
#
# ## Stage 8: Deployment
#
# **Communicating Results**:
# The visualizations provide a comprehensive story about the accuracy of the predictive models. The results can be shared with healthcare policymakers and county officials to make informed decisions about resource allocation and health interventions.
#
# **Saving Results**:
# The visualizations have been saved as image files, and these results can now be presented or embedded in reports for stakeholders.
#

#

# ## Step 9: Post the Story to Slack
#
#
# Final step, **POST your most compelling visual and provide a brief description of what it conveys, to the mutual aid channel (the slack channel for the course).** Feel free to post more examples for people to look at and provide feedback.
#
# You might consider posting some aspects during the process. Your classmates will be vital providers of feedback in this process. Utilize them.

# # Save your notebook, then `File > Close and Halt`
