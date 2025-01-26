#importing required libraries
import yaml
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from statsmodels.tsa.arima.model import ARIMA

#Read the Excel file
df = pd.read_csv("TB_Burden_Country_original.csv")

#Create an engine to connect to the SQLite database
engine = create_engine('sqlite:///FinalProject.db')

#establishing connection with sql database
conn = sqlite3.connect("TB_Burden_Country_original.db")
cursor = conn.cursor()

#Write the DataFrame to a SQL table
df.to_sql('TB_Burden_Country_one',con=engine, if_exists='replace', index=False)

#Verify the table creation by reading the data back
df_from_sql = pd.read_sql('SELECT * FROM TB_Burden_Country_one', con=engine)
df_from_sql.head()

#Select the desired columns (example: selecting columns 'A', 'B', and 'C')
selected_columns = df_from_sql[["Country or territory name",
"Year",
"Estimated total population number",
"Estimated prevalence of TB (all forms)",
"Method to derive prevalence estimates",
"Estimated number of deaths from TB (all forms, excluding HIV)",
"Estimated number of deaths from TB in people who are HIV-positive",
"Method to derive mortality estimates",
"Estimated number of incident cases (all forms)",
"Method to derive incidence estimates",
"Estimated HIV in incident TB (percent)",
"Estimated incidence of TB cases who are HIV-positive",
"Method to derive TBHIV estimates",
"Case detection rate (all forms), percent"]]

#Write the selected columns to a new SQL table
selected_columns.to_sql('NEW_TB_Burden_Country', con=engine, if_exists='replace', index=False)

#Verify the new table creation by reading the data back
df_from_sql = pd.read_sql('SELECT * FROM NEW_TB_Burden_Country', con=engine)
df_from_sql.head()

#saving the progress in sql
conn.commit() 
conn.close()

#displaying the shape and the list of columns of the dataframe
print(df_from_sql.shape)
print(df_from_sql.columns)

#Handle the missing values
"""
Method to derive incidence estimates has 2133 missing values out of 5120,
so we are considering to fill the not available columns as "Other"
"""
df_from_sql['Method to derive incidence estimates'] = df_from_sql['Method to derive incidence estimates'].fillna('Other')


"""
Case detection rate (all forms), percent has only 449 missing values 
so, we have handled it by updating with the mean of each of its country
"""
df_from_sql['Case detection rate (all forms), percent'] = df_from_sql.groupby('Country or territory name')['Case detection rate (all forms), percent'].transform(lambda x: x.fillna(x.mean()))

#saving the dataframe after dropping the unnecessary columns
# Save the DataFrame to a new SQL table or overwrite the existing table
df_from_sql.to_sql('my_table', con=engine, if_exists='replace', index=False)
df_from_sql

#Exploratory data analysis
#Storing all numerical columns in the list for Summary analysis
cols=['Estimated total population number',
 'Estimated prevalence of TB (all forms)',
 'Estimated number of deaths from TB (all forms, excluding HIV)',
 'Estimated number of deaths from TB in people who are HIV-positive',
 'Estimated number of incident cases (all forms)',
 'Case detection rate (all forms), percent']
stats = {
    'mean': {},
    'median': {},
    'std_dev': {},
    'variance': {},
    'coef_variation': {}
}

# Loop through each column in the DataFrame
for column in cols:
    mean_value = df_from_sql[column].mean()
    median_value = df_from_sql[column].median()
    std_deviation = df_from_sql[column].std()
    variance_value = df_from_sql[column].var()
    coef_variation = std_deviation / mean_value if mean_value != 0 else np.nan

    # Store the computed statistics in the dictionary
    stats['mean'][column] = mean_value
    stats['median'][column] = median_value
    stats['std_dev'][column] = std_deviation
    stats['variance'][column] = variance_value
    stats['coef_variation'][column] = coef_variation

# Print the results for each column
for stat, values in stats.items():
    print(f"\n{stat.capitalize()}:")
    for column, value in values.items():
        print(f"{column}: {value}")

# for categorical columns
cat=['Year','Country or territory name','Method to derive prevalence estimates','Method to derive mortality estimates','Method to derive incidence estimates']
for i in cat:
    for j in cols:
        print(df_from_sql.groupby(i)[j].mean())
        print("\n")

#Distribution of TB Prevalence:
plt.figure(figsize=(10, 6))
sns.histplot(df_from_sql['Estimated prevalence of TB (all forms)'], bins=30, kde=True)
plt.title('Distribution of TB Prevalence')
plt.xlabel('Estimated Prevalence of TB (all forms)')
plt.ylabel('Frequency')
plt.show()

#TB Prevalence:
plt.figure(figsize=(15, 8))
top_countries = df_from_sql.groupby('Country or territory name')['Estimated prevalence of TB (all forms)'].sum().nlargest(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by TB prevalence')
plt.xlabel('Estimated prevalence of TB (all forms)')
plt.ylabel('Country')
plt.show()

#TB Mortality Rates by Country excluding HIV:
plt.figure(figsize=(15, 8))
top_countries = df_from_sql.groupby('Country or territory name')['Estimated number of deaths from TB (all forms, excluding HIV)'].sum().nlargest(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by TB Mortality Rates')
plt.xlabel('Estimated Number of Deaths from TB (all forms, excluding HIV)')
plt.ylabel('Country')
plt.show()

#TB Mortality Rates by Country including HIV positive:
plt.figure(figsize=(15, 8))
top_countries = df_from_sql.groupby('Country or territory name')['Estimated number of deaths from TB in people who are HIV-positive'].sum().nlargest(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by TB Mortality Rates')
plt.xlabel('Estimated Number of Deaths from TB in people who are HIV-positive')
plt.ylabel('Country')
plt.show()

#Correlation Analysis:
# Select only numeric columns for correlation
numeric_df = df_from_sql.select_dtypes(include=['float64', 'int64'])

# Correlation Analysis
plt.figure(figsize=(12, 10))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Global Trend Analysis Over Time:
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_from_sql, x='Year', y='Estimated prevalence of TB (all forms)')#, hue='Country')
plt.title('Estimated prevalence of TB (all forms) globally')
plt.xlabel('Year')
plt.ylabel('Estimated prevalence of TB (all forms)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Path to the downloaded shapefile
shapefile_path = "110m_cultural/ne_110m_admin_0_countries.shp"

# Load the shapefile
world = gpd.read_file(shapefile_path)

#Assuming df_geo is already prepared with the required columns
df_geo = df_from_sql.groupby('Country or territory name', as_index=False).sum()

# Merge GeoDataFrame with the data
world = world.merge(df_geo, how='left', left_on='NAME', right_on='Country or territory name')

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(
    column='Estimated number of deaths from TB (all forms, excluding HIV)', 
    ax=ax,
    legend=True,
    legend_kwds={
        'label': "Estimated Number of Deaths from TB (all forms, excluding HIV)",
        'orientation': "horizontal"
    }
)
plt.title('Geographical Distribution of TB Mortality')
plt.show()

#TB mortality
#Assuming df_geo is already prepared with the required columns
df_geo = df_from_sql.groupby('Country or territory name', as_index=False).sum()
# Merge GeoDataFrame with the data
world = world.merge(df_geo, how='left', left_on='NAME', right_on='Country or territory name')
# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(
    column='Estimated number of deaths from TB in people who are HIV-positive',
    ax=ax,
    legend=True,
    legend_kwds={
        'label':'Estimated number of deaths from TB in people who are HIV-positive',
        'orientation': "horizontal"
    }
)
plt.title('Geographical Distribution of TB Mortality')
plt.show()

#Summary Statistics:
df_from_sql.describe()

#Country-wise Analysis for TB incident cases:
highest_tb_countries = df_from_sql.groupby('Country or territory name')['Estimated number of incident cases (all forms)'].sum().nlargest(5)
lowest_tb_countries = df_from_sql.groupby('Country or territory name')['Estimated number of incident cases (all forms)'].sum().nsmallest(5)

print("Countries with highest TB incident cases:")
print(highest_tb_countries)

print("\nCountries with lowest TB incident cases:")
print(lowest_tb_countries)

#Country wise Total population:
#Country-wise Analysis for TB incident cases:
highest_tb_countries = df_from_sql.groupby('Country or territory name')['Estimated total population number'].sum().nlargest(5)
lowest_tb_countries = df_from_sql.groupby('Country or territory name')['Estimated total population number'].sum().nsmallest(5)

print("Countries with highest population between 1990-2013: ")
print(highest_tb_countries)

print("\nCountries with lowest population between 1990-2013: ")
print(lowest_tb_countries)

#Multiple Linear Regression
# One-Hot Encoding for the 'Country or territory name' column
data_encoded = pd.get_dummies(df_from_sql, columns=['Country or territory name'], drop_first=True)

# Select independent variables (including country)
X = data_encoded[['Estimated total population number', 
                  'Estimated prevalence of TB (all forms)', 
                  'Estimated number of incident cases (all forms)',
                  'Case detection rate (all forms), percent'] + 
                 [col for col in data_encoded.columns if 'Country or territory name' in col]]

# Select dependent variable (TB deaths, excluding HIV)
y = df_from_sql['Estimated number of deaths from TB (all forms, excluding HIV)']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics and model coefficients
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Regression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Create a scatter plot for predicted vs. actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7)

# Plot a diagonal line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

# Labeling the axes and title
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted TB Deaths', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#Document Summarization using LLM's and generating an overall global report
# Load your OpenAI key
OpenAI_Key = yaml.safe_load(open("credentials1.yml"))["openai"]

# Define the prompt template for a global report
global_prompt_template = """
write a business report based on the following TB dataset summary at a global level:
- Total countries: {total_countries}
- Total estimated deaths: {total_deaths}
- Global trends: {global_trends}
Use the following Markdown format:
# Global TB Mortality Report
## Summary
Provide an overview based on the data provided, including a summary of the total estimated deaths and trends.
## Important Financials
Discuss any relevant global financial impacts.
## Key Business Risks
Highlight key global risks related to TB mortality.
## Conclusions
Conclude with overarching global actions and implications.
"""

# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OpenAI_Key)

def generate_global_report(data):
    # Calculate global statistics
    total_countries = len(data)
    total_deaths = data['Estimated number of deaths from TB (all forms, excluding HIV)'].sum()
    
    # Extract trends and other global insights from the data
    # (For example, a simple analysis of the top 5 countries with the highest deaths)
    global_trends = data[['Country or territory name', 'Estimated number of deaths from TB (all forms, excluding HIV)']] \
                    .sort_values(by='Estimated number of deaths from TB (all forms, excluding HIV)', ascending=False) \
                    .head(5).to_dict(orient='records')
    
    # Format the prompt with global data
    prompt = global_prompt_template.format(
        total_countries=total_countries,
        total_deaths=total_deaths,
        global_trends=global_trends
    )
    
    # Generate the global report by calling the OpenAI model directly
    response = model.invoke(prompt)
    
    # Access the response content (text) from the AIMessage object
    report_text = response.content  # Extracting the 'content' field
    
    return report_text

# Generate the global report
global_report = generate_global_report(df_from_sql)

# Print the global report
print(global_report)

# Time series forecasting using ARIMA for India and Nigeria
#FINAL Code for ARIMA for INDIA because it is highest in no of deaths excluding HIV
# Inspect the column names
print(df_from_sql.columns)

# Filter for India and select relevant columns
df_from_sql = df_from_sql[df_from_sql["Country or territory name"] == "India"]  # Use the correct column name for 'Country'
df_from_sql = df_from_sql[["Year", "Estimated number of deaths from TB (all forms, excluding HIV)"]]
df_from_sql.rename(columns={"Year": "Year", "Estimated number of deaths from TB (all forms, excluding HIV)": "deaths"}, inplace=True)

# Convert 'year' to integers and set it as the index
df_from_sql["Year"] = df_from_sql["Year"].astype(int)
df_from_sql.set_index("Year", inplace=True)

# Filter data for years 1990–2013
df_from_sql = df_from_sql.loc[1990:2013]

# Train the ARIMA model
model = ARIMA(df_from_sql["deaths"], order=(1, 1, 1))  # Adjust the order (p, d, q) as needed
model_fit = model.fit()

# Forecast for 2025–2030
forecast_years = range(2025, 2031)
forecast = model_fit.forecast(steps=len(forecast_years))

# Create a DataFrame for forecasted values
forecast_df = pd.DataFrame({
    "Year": forecast_years,
    "deaths": forecast
}).set_index("Year")

# Combine historical and forecasted data
combined_data = pd.concat([df_from_sql, forecast_df])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df_from_sql.index, df_from_sql["deaths"], label="Historical Data", color="blue")
plt.plot(forecast_df.index, forecast_df["deaths"], label="Future Forecast (2025–2030)", color="orange")
plt.title("Forecasting Estimated Number of Deaths from TB (all forms, excluding HIV) for India")
plt.xlabel("Year")
plt.ylabel("Estimated Number of Deaths")
plt.legend()
plt.grid()
plt.show()

# Display forecasted values
print("\nForecasted values (2025–2030):")
print(forecast_df)

#FINAL Code for ARIMA for NIGERIA because it is highest in no of deaths excluding HIV
# Inspect the column names
df_from_sql = pd.read_sql('NEW_TB_Burden_Country', con=engine)
print(df_from_sql.columns)

# Filter for Nigeria and select relevant columns
df_from_sql = df_from_sql[df_from_sql["Country or territory name"] == "Nigeria"]  # Use the correct column name for 'Country'
df_from_sql = df_from_sql[["Year",'Estimated number of deaths from TB in people who are HIV-positive']]
df_from_sql.rename(columns={"Year": "Year",'Estimated number of deaths from TB in people who are HIV-positive': "deaths"}, inplace=True)

# Convert 'year' to integers and set it as the index
df_from_sql["Year"] = df_from_sql["Year"].astype(int)
df_from_sql.set_index("Year", inplace=True)

# Filter data for years 1990–2013
df_from_sql = df_from_sql.loc[1990:2013]

# Train the ARIMA model
model = ARIMA(df_from_sql["deaths"], order=(1, 1, 1))  # Adjust the order (p, d, q) as needed
model_fit = model.fit()

# Forecast for 2025–2030
forecast_years = range(2025, 2031)
forecast = model_fit.forecast(steps=len(forecast_years))

# Create a DataFrame for forecasted values
forecast_df = pd.DataFrame({
    "Year": forecast_years,
    "deaths": forecast
}).set_index("Year")

# Combine historical and forecasted data
combined_data = pd.concat([df_from_sql, forecast_df])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df_from_sql.index, df_from_sql["deaths"], label="Historical Data", color="blue")
plt.plot(forecast_df.index, forecast_df["deaths"], label="Future Forecast (2025–2030)", color="orange")
plt.title("Forecasting Estimated number of deaths from TB in people who are HIV-positive for Nigeria")
plt.xlabel("Year")
plt.ylabel("Estimated Number of Deaths")
plt.legend()
plt.grid()
plt.show()

# Display forecasted values
print("\nForecasted values (2025–2030):")
print(forecast_df)

#Forecast of incident cases for India and Nigeria
# Filter data for India and Nigeria
data_india = df_from_sql[df_from_sql["Country or territory name"] == "India"]
data_nigeria = df_from_sql[df_from_sql["Country or territory name"] == "Nigeria"]

# Ensure data is sorted by year and set year as index
data_india = data_india.sort_values(by="Year").set_index("Year")
data_nigeria = data_nigeria.sort_values(by="Year").set_index("Year")

# Train ARIMA model for India
model_india = ARIMA(data_india['Estimated number of incident cases (all forms)'], order=(1, 1, 1))
fit_india = model_india.fit()

# Train ARIMA model for Nigeria
model_nigeria = ARIMA(data_nigeria['Estimated number of incident cases (all forms)'], order=(1, 1, 1))
fit_nigeria = model_nigeria.fit()

# Forecast from 2025 to 2030
forecast_years = [2025, 2026, 2027, 2028, 2029, 2030]
forecast_india = fit_india.forecast(steps=len(forecast_years))
forecast_nigeria = fit_nigeria.forecast(steps=len(forecast_years))

# Create forecast DataFrames
forecast_india_df = pd.DataFrame({"Year": forecast_years, "Forecast_India": forecast_india})
forecast_nigeria_df = pd.DataFrame({"Year": forecast_years, "Forecast_Nigeria": forecast_nigeria})

# Merge forecasts with historical data
historical_india = data_india.reset_index()
historical_nigeria = data_nigeria.reset_index()

# Plot India
plt.figure(figsize=(10, 6))
plt.plot(historical_india["Year"], historical_india['Estimated number of incident cases (all forms)'], label="India - Historical")
plt.plot(forecast_india_df["Year"], forecast_india_df["Forecast_India"], label="India - Forecast", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Estimated number of incident cases (all forms)")
plt.title("India - Estimated Number of Incident Cases (2025-2030)")
plt.legend()
plt.show()

# Plot Nigeria
plt.figure(figsize=(10, 6))
plt.plot(historical_nigeria["Year"], historical_nigeria['Estimated number of incident cases (all forms)'], label="Nigeria - Historical")
plt.plot(forecast_nigeria_df["Year"], forecast_nigeria_df["Forecast_Nigeria"], label="Nigeria - Forecast", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Incident Cases")
plt.title("Nigeria - Estimated Number of Incident Cases (2025-2030)")
plt.legend()
plt.show()

# Save results to CSV
forecast_india_df.to_csv("forecast_india_2025_2030.csv", index=False)
forecast_nigeria_df.to_csv("forecast_nigeria_2025_2030.csv", index=False)