import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(layout="wide")  # Set page layout to wide mode
st.title("Payment Transaction Fraud Detection")

# Initialize session state variables for predictions and transaction ID
if 'predictions_df' not in st.session_state:
    st.session_state['predictions_df'] = pd.DataFrame()

# Model selection
model_options = ['DNN.keras', 'LSTM.keras', 'LGBM.txt', 'RF.joblib']
model_selected = st.selectbox("Select Model", model_options)

# File uploads
transaction_file = st.file_uploader("Upload your Transaction CSV file", type=["csv"], key="transaction")
identity_file = st.file_uploader("Upload your Identity CSV file", type=["csv"], key="identity")


# Create tabs
tab1, tab2 = st.tabs(["Predictions", "Explanations"])



#                   TRIGGER PREDICTION                   #
with tab1:
    if st.button("Predict", key="predict") and transaction_file and identity_file:
        files = {
            'transaction_file': ('transaction.csv', transaction_file, 'text/csv'),
            'identity_file': ('identity.csv', identity_file, 'text/csv')
        }
        response = requests.post(
            f'http://localhost:8000/predict/?model_name={model_selected}',
            files=files
        )
        if response.status_code == 200:
            try:
                st.subheader("Tranasction Predictions")
                response_data = response.json()  # Decode JSON response
                # Load the data into the session state
                st.session_state['predictions_df'] = pd.read_json(response_data['data'], orient='records')
                st.session_state['predictions_df'].rename(columns={
                    'predictions': 'isFraud',
                    'DT': 'Transaction Date Time',
                    'TransactionAmt': 'Transaction Amount',
                    'ProductCD': 'Product Category',
                    'card4': 'Payment Gateway',
                    'card6': 'Type',
                    'P_emaildomain': 'Payer Email Domain',
                    'DeviceType': 'Device Type',
                }, inplace=True)

                # Reset file pointer to the beginning
                transaction_file.seek(0)
                identity_file.seek(0)

                # Read the files as DataFrames
                df_transaction = pd.read_csv(transaction_file)
                df_identity = pd.read_csv(identity_file)

                # Merge transaction and identity data
                master_df = pd.merge(df_transaction, df_identity, on='TransactionID', how='left')
                # Merge the predictions into the master dataframe
                if 'TransactionID' in master_df.columns and 'TransactionID' in st.session_state['predictions_df'].columns:
                    master_df = pd.merge(master_df, st.session_state['predictions_df'][['TransactionID', 'isFraud']], on='TransactionID', how='left')
                    st.session_state['master_df'] = master_df
                    st.session_state['predictions_made'] = True

                # Display the formatted DataFrame
                st.dataframe(st.session_state['predictions_df'])

            except json.JSONDecodeError:
                st.error("Received malformed JSON response.")
        else:
            st.error(f"Failed to make predictions: {response.text}")


# #                   TRIGGER EXPLANATION                   #
with tab2:
    st.subheader("Get Explanation for a Transaction")
    transaction_id_input = st.text_input("Enter TransactionID to get explanation", key='transaction_id')

    if st.button("Explain Transaction", key='explain'):
        if transaction_id_input:
            explain_response = requests.get(
                f"http://localhost:8000/explain/{transaction_id_input.strip()}?model_name={model_selected}"
            )
            if explain_response.status_code == 200:
                explanation_data = explain_response.json()
                st.write("LIME Explanation:", explanation_data)
            else:
                st.error(f"Failed to get explanations: {explain_response.json().get('detail', 'No error details provided.')}")
        else:
            st.error("Please enter a TransactionID.")






#                   DASHBOARD                   #
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import timedelta

# Define a function to filter the DataFrame based on the timeframe
def filter_by_timeframe(df, timeframe):
    now = pd.Timestamp.now()
    if timeframe == 'Last 7 days':
        past_date = now - pd.Timedelta(days=7)
    elif timeframe == 'Last 30 days':
        past_date = now - pd.Timedelta(days=30)
    elif timeframe == 'Last 90 days':
        past_date = now - pd.Timedelta(days=90)
    else:
        past_date = df['DateTime'].min()
    return df[df['DateTime'] >= past_date]


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
    fig = px.bar(fraud_dist, x=column, y='Frequency', title=title,
                 color_discrete_sequence=['#A81D1E'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'categoryorder':'total descending'}
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)




#                   DASHBOARD EXECUTION                     #
# Import dataset
if 'predictions_made' in st.session_state and st.session_state['predictions_made']:
    master_file = st.session_state['master_df']

    # Read the data and convert TransactionDT to datetime
    START_DATE = pd.Timestamp('2017-11-30')
    master_file['DateTime'] = START_DATE + pd.to_timedelta(master_file['TransactionDT'], unit='s')

    # Dropdown to select the time frame
    timeframe = st.selectbox(
        'Select Time Frame',
        options=['Last 7 days', 'Last 30 days', 'Last 90 days', 'All time'],
        index=3  # Default selection
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
    if not fraud_trends.empty:
        fig_fraud_trend = px.line(
            fraud_trends, 
            x='DateTime',  # Ensure this column exists in fraud_trends
            y='Fraud Count',  # Ensure this column exists in fraud_trends
            title='Fraud Trends by Time',
            labels={'DateTime': 'Date', 'Fraud Count': 'Count of Fraudulent Transactions'},
            color_discrete_sequence=['#A81D1E']
        )
        fig_fraud_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )   
        st.plotly_chart(fig_fraud_trend, use_container_width=True)

        # Fraud Amount over Time
        fraud_amount_by_date = filtered_df[filtered_df['isFraud'] == 1].groupby(filtered_df['DateTime'].dt.date)['TransactionAmt'].sum()
        fig_fraud_amount = px.line(
            x=fraud_amount_by_date.index,
            y=fraud_amount_by_date.values,
            title='Fraud Amount Over Time',
            labels={'x': 'Date', 'y': 'Total Fraud Amount'},
            color_discrete_sequence=['#A81D1E']
        )
        fig_fraud_amount.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
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
            fig_product_cd = px.bar(fraud_transactions['ProductCD'].value_counts(), title='ProductCD Distribution in Fraud Transactions', color_discrete_sequence=['#A81D1E'])
            fig_product_cd.update_layout(
                showlegend=False,
                xaxis_title='ProductCD',
                yaxis_title='Amount',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_product_cd, use_container_width=True)

        # Plot in the second column
        with col2:
            # Card Type Usage
            # Histogram of TransactionAmt for fraud and non-fraud transactions
            # Create a bar plot for card type usage
            fig_card_type = px.bar(filtered_df['card4'].value_counts(), title='Card Type Usage', color_discrete_sequence=['#A81D1E'])
            fig_card_type.update_layout(
                showlegend=False,
                xaxis_title='Payment Gateways',
                yaxis_title='Amount',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_card_type, use_container_width=True)

        # Email Domain Analysis
        # Bar plot for email domains in fraudulent transactions
        fig_email_domain = px.bar(fraud_transactions['P_emaildomain'].value_counts(), title='Email Domain Analysis for Fraud Transactions', color_discrete_sequence=['#A81D1E'])
        fig_email_domain.update_layout(
            xaxis_title='Payer Email Domain',
            yaxis_title='Amount',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_email_domain, use_container_width=True)

        # Identity Plots
        filtered_df['id_30'] = filtered_df['id_30'].apply(clean_id_30)
        filtered_df['id_31'] = filtered_df['id_31'].apply(clean_id_31)

        # Filter out fraudulent transactions
        fraud_identity = filtered_df[filtered_df['isFraud'] == 1]

        # Plot the barplots
        col1, col2 = st.columns(2)
        with col1:
            plot_fraud_distribution(fraud_identity, 'id_30', 'Fraud Distribution for OS')
        with col2:
            plot_fraud_distribution(fraud_identity, 'DeviceType', 'Fraud Distribution for Device Types')

        plot_fraud_distribution(fraud_identity, 'id_31', 'Fraud Distribution for Browsers')

    else:
        st.write("No fraudulent transactions to display.")

    