"""
Created on Sat Jul  6 11:43:27 2024

@author: CK
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import os

# Function to detect anomalies
def detect_anomalies(data):
    # Preprocessing
    data['TransactionDateTime'] = pd.to_datetime(data['TransactionDateTime'], format='%d %b %Y %H:%M:%S')
    data['ScannedDateTime'] = pd.to_datetime(data['ScannedDateTime'], errors='coerce')
    data['ConfirmedDateTime'] = pd.to_datetime(data['ConfirmedDateTime'], errors='coerce')

    # Fill missing values if necessary
    data.fillna({
        'Odometer': 0,
        'PumpNumber': 0,
        'TripNo': 0,
        'TrackingCardNo': 0,
        'ScannedDateTime': pd.Timestamp.min,
        'ConfirmedDateTime': pd.Timestamp.min
    }, inplace=True)

    # Feature engineering
    data['Latitude'] = data['GPSCoordinatelatitude']
    data['Longitude'] = data['GPSCoordinateLongitude']
    data['Quantity'] = data['Quantity'].astype(float)
    data['Amount'] = data['Amount'].astype(float)

    # Sort data by VehicleRegistrationNo and TransactionDateTime to compute time differences
    data.sort_values(by=['VehicleRegistrationNo', 'TransactionDateTime'], inplace=True)

    # Compute time difference in minutes between consecutive transactions for each vehicle
    data['TimeDiff'] = data.groupby('VehicleRegistrationNo')['TransactionDateTime'].diff().dt.total_seconds() / 60
    data['TimeDiff'].fillna(data['TimeDiff'].max(), inplace=True)  # Fill NaN values with the maximum time difference

    # Select features for anomaly detection
    features = data[['Quantity', 'Amount', 'Latitude', 'Longitude', 'TimeDiff']]
    features['Hour'] = data['TransactionDateTime'].dt.hour

    # Fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.05)  # Adjust the contamination rate as needed
    data['Anomaly'] = iso_forest.fit_predict(features)

    # Label anomalies
    data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})

    # Filter anomalies
    anomalies = data[data['Anomaly'] == 1]
    
    # Indicate reasons for anomalies
    mean_quantity = data['Quantity'].mean()
    mean_amount = data['Amount'].mean()
    mean_timediff = data['TimeDiff'].mean()
    
    # Indicate reasons for anomalies
    mean_quantity = data['Quantity'].mean()
    mean_amount = data['Amount'].mean()
    mean_timediff = data['TimeDiff'].mean()
    
    anomalies['Reason'] = ''
    anomalies.loc[anomalies['Quantity'] > mean_quantity, 'Reason'] += 'High Quantity; '
    anomalies.loc[anomalies['Amount'] > mean_amount, 'Reason'] += 'High Amount; '
    anomalies.loc[anomalies['TimeDiff'] < mean_timediff, 'Reason'] += 'Short Time Diff; '
    
    # Example for unusual location (you can define criteria based on your specific use case)
    anomalies['Reason'] = anomalies['Reason'].str.rstrip('; ')
    
    return anomalies

# Streamlit app
def main():
    st.title('Fuel Transaction Anomaly Detection')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader('Raw Data')
        st.write(data)

        # Detect anomalies
        anomalies = detect_anomalies(data)

        # Display anomalies
        st.subheader('Detected Anomalies')
        st.write(anomalies)

        # Save the anomalies to a CSV file
        anomalies_file = 'detected_anomalies.csv'
        anomalies.to_csv(anomalies_file, index=False)

        # Provide a download button for the anomalies
        st.download_button(
            label="Download detected anomalies as CSV",
            data=anomalies.to_csv(index=False).encode('utf-8'),
            file_name='detected_anomalies.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
