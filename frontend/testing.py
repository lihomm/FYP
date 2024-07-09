import pandas as pd
import plotly.express as px
import streamlit as st
import datetime
from datetime import timedelta, datetime

# Define a function to filter the DataFrame based on the timeframe
def filter_by_timeframe(df, timeframe):
    if timeframe == 'Last 7 days':
        start_date = df['DateTime'].max() - timedelta(days=7)
    elif timeframe == 'Last 30 days':
        start_date = df['DateTime'].max() - timedelta(days=30)
    elif timeframe == 'Last 90 days':
        start_date = df['DateTime'].max() - timedelta(days=90)
    else:  # 'All time'
        start_date = df['DateTime'].min()
    return df[df['DateTime'] >= start_date]


# Clean and transform 'id_30'
def clean_id_30(entry):
    entry = entry.lower() if pd.notnull(entry) else entry
    if pd.isnull(entry):
        return "Missing"
    if "android" in entry:
        return "Android"
    if "ios" in entry:
        return "iOS"
    if "mac os" in entry:
        return "Mac"
    if "windows" in entry:
        return "Windows"
    if "linux" in entry:
        return "Linux"
    return "Other"


# Clean and transform 'id_31'
def clean_id_31(entry):
    entry = entry.lower() if pd.notnull(entry) else entry
    if pd.isnull(entry):
        return "Missing"
    browsers = ["android webview", "chrome", "firefox", "edge", "ie", "opera", "safari", "samsung browser"]
    for browser in browsers:
        if browser in entry:
            return browser.title()  # Convert first character of each word to uppercase
    return "Other"



# Bar plots for each category with fraud transactions
def plot_fraud_distribution(dataframe, column, title):
    # Count the frequency of each category
    fraud_dist = dataframe[column].value_counts().reset_index()
    fraud_dist.columns = [column, 'Frequency']
    
    # Create the bar plot
    fig = px.bar(fraud_dist, x=column, y='Frequency', title=title)
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)



# Import dataset
transaction_file = pd.read_csv('C:/Users/kohli/Downloads/ieee-fraud-detection/train_transaction.csv')
identity_file = pd.read_csv('C:/Users/kohli/Downloads/ieee-fraud-detection/train_identity.csv')

master_file = pd.merge(transaction_file, identity_file, on='TransactionID', how='left')


# Read the data and convert TransactionDT to datetime
START_DATE = pd.Timestamp('2017-11-30')
master_file['DateTime'] = START_DATE + pd.to_timedelta(transaction_file['TransactionDT'], unit='s')

# Dropdown to select the time frame
timeframe = st.selectbox(
    'Select Time Frame',
    options=['Last 7 days', 'Last 30 days', 'Last 90 days', 'All time'],
    index=1  # Default selection
)

# Filter the DataFrame based on the selected timeframe
filtered_df = filter_by_timeframe(master_file, timeframe)

# Calculate metrics based on the filtered dataframe
fraud_count = filtered_df['isFraud'].sum()
transaction_count = len(filtered_df)
fraud_amount = "${:,.2f}".format(filtered_df.loc[filtered_df['isFraud'] == 1, 'TransactionAmt'].sum())
total_transaction_amount = "${:,.2f}".format(filtered_df['TransactionAmt'].sum())


st.title('Fraud Statistics')
# Using columns for a side-by-side layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Fraud Count", fraud_count)
with col2:
    st.metric("Transaction Count", transaction_count)

col3, col4 = st.columns(2)
with col3:
    st.metric("Fraud Amount", fraud_amount)

with col4:
    st.metric("Total Transaction Amount", total_transaction_amount)



# Fraud Trend Plot
fraud_trends = filtered_df[filtered_df['isFraud'] == 1].groupby(filtered_df['DateTime'].dt.date).size().reset_index(name='Fraud Count')
fig_fraud_trend = px.line(
    fraud_trends, 
    x='DateTime', 
    y='Fraud Count', 
    title='Fraud Trends by Time',
    labels={'DateTime': 'Date', 'Fraud Count': 'Count of Fraudulent Transactions'}
)
st.plotly_chart(fig_fraud_trend, use_container_width=True)



# Fraud Amount over Time
fraud_amount_by_date = filtered_df[filtered_df['isFraud'] == 1].groupby(filtered_df['DateTime'].dt.date)['TransactionAmt'].sum()
fig_fraud_amount = px.line(
    x=fraud_amount_by_date.index,
    y=fraud_amount_by_date.values,
    title='Fraud Amount Over Time',
    labels={'x': 'Date', 'y': 'Total Fraud Amount'}
)
st.plotly_chart(fig_fraud_amount, use_container_width=True)



# Set up two columns for the plots
col1, col2 = st.columns(2)
# Plot in the first column
with col1:
    # ProductCD Distribution in Fraud Transactions
    # Filter out fraudulent transactions
    fraud_transactions = filtered_df[filtered_df['isFraud'] == 1]

    # Create a bar plot for ProductCD
    fig_product_cd = px.bar(fraud_transactions['ProductCD'].value_counts(), title='ProductCD Distribution in Fraud Transactions')
    fig_product_cd.update_layout(
        xaxis_title='ProductCD',
        yaxis_title='Amount',
    )
    st.plotly_chart(fig_product_cd, use_container_width=True)

# Plot in the second column
with col2:
    # Card Type Usage
    # Histogram of TransactionAmt for fraud and non-fraud transactions
    # Create a bar plot for card type usage
    fig_card_type = px.bar(filtered_df['card4'].value_counts(), title='Card Type Usage')
    fig_card_type.update_layout(
        xaxis_title='Payment Gateways',
        yaxis_title='Amount',
    )
    st.plotly_chart(fig_card_type, use_container_width=True)




# Set up two columns for the plots
col1, col2 = st.columns(2)
# Plot in the first column
with col1:
    # ProductCD Distribution in Fraud Transactions
    # Filter out fraudulent transactions
    fraud_transactions = filtered_df[filtered_df['isFraud'] == 1]

    # Create a bar plot for ProductCD
    fig_product_cd = px.bar(fraud_transactions['ProductCD'].value_counts(), title='ProductCD Distribution in Fraud Transactions')
    fig_product_cd.update_layout(
        xaxis_title='ProductCD',
        yaxis_title='Amount',
    )
    st.plotly_chart(fig_product_cd, use_container_width=True)

# Plot in the second column
with col2:
    # Card Type Usage
    # Histogram of TransactionAmt for fraud and non-fraud transactions
    # Create a bar plot for card type usage
    fig_card_type = px.bar(filtered_df['card4'].value_counts(), title='Card Type Usage')
    fig_card_type.update_layout(
        xaxis_title='Payment Gateways',
        yaxis_title='Amount',
    )
    st.plotly_chart(fig_card_type, use_container_width=True)



# Email Domain Analysis
# Bar plot for email domains in fraudulent transactions
fig_email_domain = px.bar(fraud_transactions['P_emaildomain'].value_counts(), title='Email Domain Analysis for Fraud Transactions')
fig_email_domain.update_layout(
    xaxis_title='Payer Email Domain',
    yaxis_title='Amount',
)
st.plotly_chart(fig_email_domain, use_container_width=True)







# Identity Plots
filtered_df['id_30'] = filtered_df['id_30'].apply(clean_id_30)
filtered_df['id_31'] = filtered_df['id_31'].apply(clean_id_31)

# Filter out fraudulent transactions
fraud_identity = filtered_df[filtered_df['isFraud'] == 1]

# Plot the barplots
plot_fraud_distribution(fraud_identity, 'id_30', 'Fraud Distribution for OS')

plot_fraud_distribution(fraud_identity, 'id_31', 'Fraud Distribution for Browsers')

plot_fraud_distribution(fraud_identity, 'DeviceType', 'Fraud Distribution for Device Types')


