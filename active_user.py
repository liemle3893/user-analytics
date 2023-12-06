import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta


st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    return data

# Function to calculate DAU
def calculate_dau(data):
    return data.groupby('date')['user_id'].nunique().reset_index()


def calculate_retention(data, date_column='date'):
    first_activity = data.groupby('user_id')[date_column].min().reset_index()
    first_activity.columns = ['user_id', 'first_activity_date']
    data_with_first_activity = pd.merge(data, first_activity, on='user_id')
    data_with_first_activity['within_7_days'] = data_with_first_activity[date_column] <= data_with_first_activity['first_activity_date'] + timedelta(days=7)
    data_with_first_activity['within_30_days'] = data_with_first_activity[date_column] <= data_with_first_activity['first_activity_date'] + timedelta(days=30)
    unique_7_day = data_with_first_activity[data_with_first_activity['within_7_days']].drop_duplicates(subset=['user_id'])
    unique_30_day = data_with_first_activity[data_with_first_activity['within_30_days']].drop_duplicates(subset=['user_id'])
    return len(unique_7_day) / len(first_activity), len(unique_30_day) / len(first_activity)

# Function to calculate A30
def calculate_a30(data, date_range):
    def active_users(date):
        end_date = date
        start_date = end_date - timedelta(days=29)
        return data[(data['date'] >= start_date) & (data['date'] <= end_date)]['user_id'].nunique()
    return [active_users(date) for date in date_range]


# Function to perform cohort analysis
def cohort_analysis(data):
    data['Cohort'] = data.groupby('user_id')['date'].transform('min').dt.to_period('M')
    data['CohortPeriod'] = (data['date'].dt.to_period('M') - data['Cohort']).apply(lambda x: x.n)
    cohort_data = data.groupby(['Cohort', 'CohortPeriod'])['user_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot_table(index='Cohort', columns='CohortPeriod', values='user_id')
    cohort_size = cohort_pivot.iloc[:,0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0).round(3)
    return retention_matrix

# Function to plot the cohort analysis
def plot_cohort(retention_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='Blues')
    plt.title('Cohort Analysis - User Retention')
    plt.ylabel('Cohort')
    plt.xlabel('Months Since First Activity')
    plt.yticks(rotation=0)
    plt.show()

def calculate_active_users(data, period='D'):
    # Group by the specified period and count unique users
    active_users = data.groupby(data['date'].dt.to_period(period))['user_id'].nunique()

    # Converting PeriodIndex to DateTimeIndex for accurate plotting
    active_users.index = active_users.index.to_timestamp()

    return active_users.sort_index()

def user_acquisition(data):
    # Identifying the first activity date for each user
    first_activity = data.groupby('user_id')['date'].min().reset_index()
    
    # Grouping by the first activity date to count new users per period (e.g., monthly)
    acquisition_data = first_activity.groupby(first_activity['date'].dt.to_period('M'))['user_id'].nunique()
    
    # Converting PeriodIndex to DateTimeIndex for accurate plotting
    acquisition_data.index = acquisition_data.index.to_timestamp()
    
    return acquisition_data


def calculate_monthly_churn(data):
    # Convert dates to monthly periods
    data['Month'] = data['date'].dt.to_period('M')

    # Getting the list of unique users for each month
    monthly_users = data.groupby('Month')['user_id'].unique()

    # Calculating churned users
    churned_users = {}
    for month in sorted(monthly_users.index[:-1]):  # Exclude the last month as it has no next month to compare
        current_month_users = set(monthly_users[month])
        next_month_users = set(monthly_users[month + 1])
        churned_users[month] = len(current_month_users - next_month_users)

    # Converting to a pandas Series for easy plotting
    churned_users_series = pd.Series(churned_users, name='Churned Users')

    # Converting PeriodIndex to DateTimeIndex for accurate plotting
    churned_users_series.index = churned_users_series.index.to_timestamp()

    return churned_users_series

def calculate_average_churn(data):
    # Calculate monthly churn using the previously defined function
    monthly_churn = calculate_monthly_churn(data)

    # Calculate the average churn
    average_churn = monthly_churn.mean()

    return average_churn


# Main Streamlit app
def main():
    st.title("User Activity Analysis")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is None:
        st.write("Please upload a CSV file to continue")
        return
    data = load_data(uploaded_file)

    # Active Users
    st.header("Active Users")
    period = st.selectbox("Select Period", ['M','D','W'])
    active_users = calculate_active_users(data, period)
    st.bar_chart(active_users)

    # Churn
    st.header("Monthly Churned Users")
    monthly_churned_users = calculate_monthly_churn(data)
    st.bar_chart(monthly_churned_users)    
    # Calculate and display the average churn
    average_churn = calculate_average_churn(data)
    print(f"The average monthly churn is: {average_churn}")    
    st.write(f"The average monthly churn is: {average_churn:.2f} users")

    # New Users
    st.header("User Acquisition Trends")
    acquisition = user_acquisition(data)
    st.bar_chart(acquisition)    

    # Cohort Analysis
    st.header("Cohort Analysis")
    retention_matrix = cohort_analysis(data)
    st.pyplot(plot_cohort(retention_matrix))        


if __name__ == "__main__":
    main()
