import numpy as np
import pandas as pd
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, OrdinalEncoder
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

'''                 FUNCTION DEFINITION                 ''' 
def transform_DeviceInfo(df):
    df['DeviceCorp'] = df['DeviceInfo'].str.upper()  # Convert to uppercase to standardize before processing
    # Categorize based on known patterns
    df.loc[df['DeviceInfo'].str.contains('HUAWEI|HONOR', case=False, na=False), 'DeviceCorp'] = 'HUAWEI'
    df.loc[df['DeviceInfo'].str.contains('OS', na=False), 'DeviceCorp'] = 'APPLE'
    df.loc[df['DeviceInfo'].str.contains('Idea|TA', case=False, na=False), 'DeviceCorp'] = 'LENOVO'
    df.loc[df['DeviceInfo'].str.contains('Moto|XT|Edison', case=False, na=False), 'DeviceCorp'] = 'MOTO'
    df.loc[df['DeviceInfo'].str.contains('MI|Mi|Redmi', case=False, na=False), 'DeviceCorp'] = 'MI'
    df.loc[df['DeviceInfo'].str.contains('VS|LG|EGO', case=False, na=False), 'DeviceCorp'] = 'LG'
    df.loc[df['DeviceInfo'].str.contains('ONE TOUCH|ALCATEL', case=False, na=False), 'DeviceCorp'] = 'ALCATEL'
    df.loc[df['DeviceInfo'].str.contains('ONE A', na=False), 'DeviceCorp'] = 'ONEPLUS'
    df.loc[df['DeviceInfo'].str.contains('OPR6', na=False), 'DeviceCorp'] = 'HTC'
    df.loc[df['DeviceInfo'].str.contains('Nexus|Pixel', case=False, na=False), 'DeviceCorp'] = 'GOOGLE'
    df.loc[df['DeviceInfo'].str.contains('STV', na=False), 'DeviceCorp'] = 'BLACKBERRY'
    df.loc[df['DeviceInfo'].str.contains('ASUS', case=False, na=False), 'DeviceCorp'] = 'ASUS'
    df.loc[df['DeviceInfo'].str.contains('BLADE', case=False, na=False), 'DeviceCorp'] = 'ZTE'
    
    # Categorize based on the starting letter or specific substrings
    df.loc[df['DeviceInfo'].str.startswith('Z', na=False), 'DeviceCorp'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.startswith('KF', na=False), 'DeviceCorp'] = 'AMAZON'
    
    for i in ['D', 'E', 'F', 'G']:
        df.loc[df['DeviceInfo'].str.startswith(i, na=False), 'DeviceCorp'] = 'SONY'
    
    # Group the less frequent categories into 'Other'
    df.loc[df['DeviceCorp'].isin(df['DeviceCorp'].value_counts()[df['DeviceCorp'].value_counts() < 100].index), 'DeviceCorp'] = 'OTHER'

    return df

# Add the date features to DataFrame
def add_date_features(df, start_date, us_holidays):
    df['day'] = (df['TransactionDT'] // (3600 * 24) - 1) % 7
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['D1n'] = df['D1'] - df['TransactionDT'] / np.float32(24 * 60 * 60)
    df['DT'] = df['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))
    df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
    df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.isocalendar().week).astype(np.int8)
    df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)
    df['DT_hour'] = df['DT'].dt.hour.astype(np.int8)
    df['DT_day_week'] = df['DT'].dt.dayofweek.astype(np.int8)
    df['DT_day_month'] = df['DT'].dt.day.astype(np.int8)
    df['is_december'] = (df['DT'].dt.month == 12).astype(np.int8)
    df['is_holiday'] = df['DT'].dt.date.isin(us_holidays).astype(np.int8)
    return df




'''                 TRANSFORMER CLASS                 ''' 
class ReduceFloatingPoint(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns  # List of columns to apply convert_column to

    def fit(self, X, y=None):
        # No fitting process needed for this transformer, so just return self
        return self

    def transform(self, X):
        # Apply the convert_column function to specified columns in X
        for col in self.columns:
            if col in X.columns:
                X[col] = self.convert_column(X[col])
        return X

    def convert_column(self, col_data):
        # Check if all values in the column are whole numbers (even if represented as floats)
        if (col_data % 1 == 0).all():
            # The column can be converted to an integer type
            if col_data.min() >= np.iinfo(np.int8).min and col_data.max() <= np.iinfo(np.int8).max:
                return col_data.astype(np.int8)
            elif col_data.min() >= np.iinfo(np.int16).min and col_data.max() <= np.iinfo(np.int16).max:
                return col_data.astype(np.int16)
            elif col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                return col_data.astype(np.int32)
            else:
                return col_data.astype(np.int64)
        else:
            # Convert to the most appropriate floating-point type
            if col_data.min() >= np.finfo(np.float16).min and col_data.max() <= np.finfo(np.float16).max:
                temp_col = col_data.astype(np.float16)
                return temp_col
            return col_data.astype(np.float32)  # default to float32 for floating point numbers
        
        
class DeviceInfoTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for this transformer, return self
        return self

    def transform(self, X):
        # Apply the transformation logic from transform_DeviceInfo
        X_transformed = X.copy()
        X_transformed = transform_DeviceInfo(X_transformed)
        return X_transformed

class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_date_str='2017-11-30'):
        self.start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        self.calendar = calendar()
        self.us_holidays = None

    def fit(self, X, y=None):
        # Ensure that the date range covers all dates in the dataset for which you need holidays
        dates_range = pd.date_range(start='2017-01-01', end='2019-12-31')
        self.us_holidays = self.calendar.holidays(start=dates_range.min(), end=dates_range.max())
        return self

    def transform(self, X):
        if self.us_holidays is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X = X.copy()
        return add_date_features(X, self.start_date, self.us_holidays)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, encoder_path=None, default_frequency=0):
        self.columns = columns
        self.encoder_path = encoder_path
        self.encoders = {}
        self.default_frequency = default_frequency  # Default frequency for unseen categories

    def fit(self, X, y=None):
        if self.encoder_path:
            # Load pre-fitted encoders
            self.encoders = joblib.load(self.encoder_path)
        else:
            # Calculate frequencies including NaNs
            for col in self.columns:
                self.encoders[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Use .get method with default frequency for unseen categories
            X[col] = X[col].map(lambda x: self.encoders[col].get(x, self.default_frequency))
        return X



class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, median_values_path='preprocessing/median_values.pkl', features=None):
        self.median_values_path = median_values_path
        self.features = features
        self.median_values_ = None

    def fit(self, X, y=None):
        self.median_values_ = joblib.load(self.median_values_path)
        if self.features is not None:
            self.median_values_ = {k: self.median_values_[k] for k in self.features}
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            if feature in X.columns and feature in self.median_values_:
                X[feature] = X[feature].fillna(self.median_values_[feature])
        return X

class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_path='preprocessing/minmax_scaler.pkl', include_columns=None, exclude_columns=None):
        self.scaler_path = scaler_path
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns or []
        self.scaler = None

    def fit(self, X, y=None):
        self.scaler = joblib.load(self.scaler_path)
        return self

    def transform(self, X):
        X = X.copy()
        if self.include_columns is not None:
            columns_to_scale = [col for col in self.include_columns if col not in self.exclude_columns and col in X.columns]
        else:
            columns_to_scale = [col for col in X.columns if col not in self.exclude_columns]
        
        if self.scaler is not None and columns_to_scale:
            X[columns_to_scale] = self.scaler.transform(X[columns_to_scale])
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_file=None):
        self.features_file = features_file
        self.selected_features = None

    def fit(self, X, y=None):
        # Check if the features file exists
        if os.path.exists(self.features_file):
            # Load the list of selected features from the file
            self.selected_features = joblib.load(self.features_file)
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"The specified features file was not found: {self.features_file}")
        return self

    def transform(self, X):
        # Ensure that the selected features are in X
        missing_features = [feat for feat in self.selected_features if feat not in X.columns]
        if missing_features:
            raise ValueError(f"The following required features are missing from the input data: {missing_features}")
        # Return only the selected features from X
        return X[self.selected_features]




'''                 PIPELINE                 ''' 
# Define data processing and feature engineering pipeline
def load_selected_features(features_file):
    with open(features_file, 'rb') as file:
        selected_features = pickle.load(file)
    return selected_features

def create_pipeline(master_continuous, master_categorical, exclusions):
    # Construct the pipeline
    pipeline = Pipeline([
        ('type_converter', ReduceFloatingPoint(columns=master_continuous)),
        ('device_info', DeviceInfoTransformer()),
        ('date_features', DateFeaturesTransformer('2017-11-30')),
        ('feature_selector', FeatureSelector(features_file='preprocessing/selected_features.pkl')),
        ('freq_encoder', FrequencyEncoder(columns=master_categorical, encoder_path='preprocessing/frequency_encoders.pkl')),
        ('median_imputer', MedianImputer(median_values_path='preprocessing/median_values.pkl', features=master_continuous)),
        ('min_max_scaler', MinMaxScaler(scaler_path='preprocessing/minmax_scaler.pkl', include_columns=master_continuous, exclude_columns=exclusions)),
    ])
    return pipeline




'''         EXECUTION           '''
def preprocess_data(data):

    # Load selected features
    selected_features = load_selected_features('preprocessing/selected_features.pkl')

    # Define lists of categorical columns
    transaction_categorical_columns = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
        'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ]

    identity_categorical_columns = [
        'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',
        'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
        'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]

    # Combine and filter categorical columns based on selected features
    all_categorical_columns = set(transaction_categorical_columns + identity_categorical_columns + ['DeviceCorp', 'DT', 'is_december', 'is_holiday'])
    master_categorical = [col for col in selected_features if col in all_categorical_columns]
    master_continuous = [col for col in selected_features if col not in all_categorical_columns]

    # Define exclusions
    exclusions = ['isFraud', 'TransactionDT', 'is_december', 'is_holiday', 'day', 'hour', 'DT_M', 'DT_W', 'DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month']

    # Pass the lists as arguments to the pipeline creation function
    pipeline = create_pipeline(master_continuous, master_categorical, exclusions)

    # Preprocess the data using the pipeline
    preprocessed_data = pipeline.fit_transform(data)

    return preprocessed_data


















