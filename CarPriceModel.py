import os
import json

import requests
import kagglehub
import datetime
import threading
import time

import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Car Price Estimator and Forecasting Class
class CarPriceEstimator:
    def __init__(self, df, car_make, car_mileage, car_model, car_variant):
        self.df = df
        self.car_make = car_make
        self.mileage = car_mileage
        self.car_model = car_model
        self.car_variant = car_variant
        self.model = None

    def preprocess_data(self, car_model, car_variant):
        filtered_df = self.df[(self.df['model'] == car_model) & (self.df['variant'] == car_variant)]
        
        if filtered_df.empty:
            # Try filtering by make if model is not found
            filtered_df = self.df[self.df['make'] == self.car_make]
        if filtered_df.empty:
            raise ValueError("No data found for the specified car model or make.")
        
        filtered_df = filtered_df.dropna()
        if filtered_df.empty:
            raise ValueError("No data found for the specified car model or make.")
        
        # Remove rows where the car price is above the 95th percentile or below the 5th percentile
        q_low = filtered_df['car_price'].quantile(0.05)
        q_hi  = filtered_df['car_price'].quantile(0.95)
        filtered_df = filtered_df[(filtered_df['car_price'] < q_hi) & (filtered_df['car_price'] > q_low)]

        for column in filtered_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            filtered_df[column] = le.fit_transform(filtered_df[column])
        X = filtered_df.drop(columns=['car_price', 'make', 'model', 'variant'])
        y = filtered_df['car_price']

        # # Normalize the car prices
        # y = np.log1p(y)

        # # Normalize any numerical columns
        # for column in X.select_dtypes(include=['int64', 'float64']).columns:
        #     X[column] = np.log1p(X[column])
        return X, y
    
    def select_important_features(self, X, y):
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True, threshold='mean')
        X_important = selector.transform(X)
        important_features = X.columns[selector.get_support()]
        print("Important Features for car_price:", list(important_features))
        return pd.DataFrame(X_important, columns=important_features), important_features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X_train, y_train)

        # What is the accuracy of the model?
        print(f"Model Accuracy: {self.model.score(X_test, y_test) * 100:.2f}%")
        accuracy = self.model.score(X_test, y_test)
        
        return self.model, accuracy

    def estimate_price(self, final_data_dict):
        input_features = final_data_dict
        input_df = pd.DataFrame([input_features])
        # Fit transform the input features
        for column in input_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            input_df[column] = le.fit_transform(input_df[column])
        # Ensure the input_df columns in the same order as the model_fit_columns
        model_fit_columns = self.model.feature_names_in_
        input_df = input_df[model_fit_columns]
        # Normalize the input features
        for column in input_df.select_dtypes(include=['int64', 'float64']).columns:
            input_df[column] = np.log1p(input_df[column])
        estimated_price = self.model.predict(input_df)[0]
        # estimated_price = np.expm1(estimated_price)
        # Round estimated price to the nearest 100
        estimated_price = round(estimated_price / 100) * 100
        return estimated_price

    def forecast_prices(self, car_model, car_variant, final_data_dict, estimated_price_2022):
        for key in final_data_dict.keys():
            if key not in self.df.columns:
                self.df[key] = final_data_dict[key]
        price_series = self.df[(self.df['model'] == car_model) & (self.df['variant'] == car_variant)][['year', 'car_price']]
        
        avg_price_per_year = price_series.groupby('year').mean()
        avg_price_per_year = avg_price_per_year.reset_index()
        avg_price_per_year = avg_price_per_year.set_index('year')
        
        avg_price_per_year.index = pd.to_datetime(avg_price_per_year.index, format='%Y')
        avg_price_per_year = avg_price_per_year.resample('Y').mean()
        avg_price_per_year = avg_price_per_year.interpolate(method='linear')
        avg_price_per_year.index = avg_price_per_year.index.year

        # Remove anomalies above 95% quantile and below 5% quantile
        q_low = avg_price_per_year['car_price'].quantile(0.05)
        q_hi  = avg_price_per_year['car_price'].quantile(0.95)
        avg_price_per_year = avg_price_per_year[(avg_price_per_year['car_price'] < q_hi) & (avg_price_per_year['car_price'] > q_low)]

        # Interpolate the car_prices for the missing years in avg_price_per_year
        avg_price_per_year.index = pd.to_datetime(avg_price_per_year.index, format='%Y')
        avg_price_per_year = avg_price_per_year.resample('Y').mean()
        avg_price_per_year = avg_price_per_year.interpolate(method='linear')
        avg_price_per_year.index = avg_price_per_year.index.year

        # Flip the index so that the most recent year is first
        avg_price_per_year = avg_price_per_year.sort_index(ascending=False)

        # Calculate the compound annual decrease rate
        cagr = (avg_price_per_year['car_price'].iloc[-1] / avg_price_per_year['car_price'].iloc[0]) ** (1 / (len(avg_price_per_year) - 1)) - 1 
        cagr = cagr * 100

        # Using this cagr with the price series and the estimated price for 2022, forecast the prices for the next 5 years
        forecasted_prices = []
        for i in range(1, 6):
            forecasted_price = estimated_price_2022 * (1 + cagr/100) ** i
            forecasted_prices.append(forecasted_price)
            print(f"Forecasted price for {car_model} in {2022 + i}: £{forecasted_price:.2f}")
        
        forecasted_prices = pd.Series(forecasted_prices, index=range(2023, 2028))
        forecasted_prices - pd.DataFrame(forecasted_prices, columns=['car_price'])

        return forecasted_prices, cagr
        

class UKUsedCarDataset:
    def __init__(self):
        self.dataset_path = None

    def download_dataset(self):
        """Download the UK used car market dataset from Kaggle."""
        self.dataset_path = kagglehub.dataset_download("guanhaopeng/uk-used-car-market")
        print("Path to dataset files:", self.dataset_path)

        # Read the dataset from the path
    def load_data(self):
        """Load the dataset into a pandas DataFrame."""
        if self.dataset_path:
            csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
            if csv_files:
                file_path = os.path.join(self.dataset_path, csv_files[0])
                return pd.read_csv(file_path)
            else:
                print("No CSV files found in dataset directory.")
        else:
            print("Dataset path not found. Please download the dataset first.")
        return None
        

class DVLAVehicleLookupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DVLA Vehicle Lookup")
        self.root.geometry("400x450")
        self.car_data = car_data
        self.typing_timer = None

        # API Configuration
        self.API_URL = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
        self.API_KEY = "1S1OS5modo6wCKBaGUZOo7nsKEj4mkHf4absR4JP"  

        tk.Label(root, text="Enter Registration Number:").pack(pady=5)
        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)
        self.entry.bind('<KeyRelease>', self.schedule_model_update)

        tk.Label(root, text="Select Model:").pack(pady=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var)
        self.model_dropdown.pack(pady=5)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.update_variant_dropdown)

        tk.Label(root, text="Select Variant:").pack(pady=5)
        self.variant_var = tk.StringVar()
        self.variant_dropdown = ttk.Combobox(root, textvariable=self.variant_var)
        self.variant_dropdown.pack(pady=5)

        tk.Label(root, text="Enter Mileage:").pack(pady=5)
        self.mileage_entry = tk.Entry(root)
        self.mileage_entry.pack(pady=5)

        self.search_btn = tk.Button(root, text="Search", command=self.search_vehicle)
        self.search_btn.pack(pady=10)

        self.result_text = tk.StringVar()
        self.result_label = tk.Label(root, textvariable=self.result_text, justify="left")
        self.result_label.pack(pady=10)

    def schedule_model_update(self, event=None):
        if self.typing_timer:
            self.typing_timer.cancel()
        self.typing_timer = threading.Timer(2, self.update_model_dropdown)
        self.typing_timer.start()

    def update_model_dropdown(self):
        reg_number = self.entry.get().strip().lower()
        headers = {
            "x-api-key": self.API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"registrationNumber": reg_number}
        # convert all characters in 'make' to lowercase
        car_data['make'] = car_data['make'].str.lower()
        make = requests.post(self.API_URL, json=payload, headers=headers).json().get('make', '')
        filtered_models = sorted(self.car_data[self.car_data['make'].str.contains(make.lower(), na=False)]['model'].dropna().unique())
        self.model_dropdown['values'] = filtered_models

    def update_variant_dropdown(self, event=None):
        selected_model = self.model_var.get().strip().lower()
        filtered_variants = sorted(self.car_data[self.car_data['model'].str.lower() == selected_model]['variant'].dropna().unique())
        self.variant_dropdown['values'] = filtered_variants

    def clean_data_dict(self, data_dict, columns, important_features):
        cleaned_data_dict = {}
        final_data_dict = {}

        # Basic cleaning by renaming keys in data_dict to match columns
        data_dict['year'] = data_dict.pop('yearOfManufacture', 'N/A')
        data_dict['feul_type'] = data_dict.pop('fuelType', 'N/A')
        data_dict['engine_vol'] = data_dict.pop('engineCapacity', 'N/A')
        data_dict['miles'] = data_dict.pop('mileage', 'N/A')
        if data_dict['motStatus'] == 'Valid':
            data_dict['full_service'] = 1
        else:
            data_dict['full_service'] = 0
        if data_dict['year'] - datetime.datetime.now().year > 1:
            data_dict['brand_new'] = 1
        else:
            data_dict['brand_new'] = 0
        
        # Divide engine_vol by 1000 and round to 1 decimal place
        data_dict['engine_vol'] = round(data_dict['engine_vol'] / 1000, 1)

        # Make all values in data_dict lowercase
        data_dict = {key: value.lower() if isinstance(value, str) else value for key, value in data_dict.items()}

        # Make all keys in data_dict lowercase
        data_dict = {key.lower(): value for key, value in data_dict.items()}

        for key, value in data_dict.items():
            match, score = process.extractOne(key, columns)
            if score > 90:
                cleaned_data_dict[match] = value
        for variable in cleaned_data_dict.keys():
            if variable not in important_features:
                continue
            final_data_dict[variable] = cleaned_data_dict[variable]
        
        return final_data_dict

    def search_vehicle(self):
        """Fetch vehicle details from the API based on user input."""
        reg_number = self.entry.get().strip().upper()
        model = self.model_var.get().strip()
        mileage = self.mileage_entry.get().strip()
        variant = self.variant_var.get().strip()
        
        if not reg_number:
            messagebox.showerror("Error", "Please enter a registration number")
            return
        
        headers = {
            "x-api-key": self.API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"registrationNumber": reg_number}
        
        try:
            response = requests.post(self.API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                query_result = self.return_result(response.json())
                # Convert any numeric values in query_result to integers
                for key, value in query_result.items():
                    if isinstance(value, str) and value.isdigit():
                        query_result[key] = int(value)

                car_make = query_result.get('make', 'N/A')
                car_mileage = query_result.get('mileage', 'N/A')
                car_model = query_result.get('model', 'N/A')
                car_variant = query_result.get('variant', 'N/A')
                # Check if car_make or car_mileage are missing as these are mandatory for the CarPriceEstimator
                if car_make == 'N/A' or car_mileage == 'N/A' or car_model == 'N/A' or car_variant == 'N/A':
                    messagebox.showerror("Error", "Registration, Model, Variant and Mileage is required for price estimation")
                    return
                
                # Initialize the CarPriceEstimator
                estimator = CarPriceEstimator(car_data, car_make, car_mileage, car_model, car_variant)

                # Preprocess the data
                X, y = estimator.preprocess_data(car_model, car_variant)

                # Select important features
                X_important, important_features = estimator.select_important_features(X, y)

                # Train the model
                model, accuracy = estimator.train_model(X_important, y)

                # See if any of the important features are in the response
                final_data_dict = self.clean_data_dict(query_result, car_data.columns, X_important.columns)
                
                # Estimate the price
                estimated_price_2022 = estimator.estimate_price(final_data_dict)
                print(f"2022 Estimated price for {car_model} with {mileage} miles: £{estimated_price_2022:.2f}")
                
                # Forecast prices
                forecast, cagr = estimator.forecast_prices(car_model, car_variant, final_data_dict, estimated_price_2022)
                estimate_2025 = forecast.loc[2025]
                estimate_2025 = estimate_2025 * (1 - accuracy)
                
                # reduce the forecasted price by the model accuracy
                forecast = forecast * (1 - accuracy)

                # Add the estimated price to the result text and add the forecasted prices
                self.display_result(response.json(), model, mileage, variant, estimate_2025, forecast)

            else:
                messagebox.showerror("Error", f"Failed to fetch details: {response.status_code}\n{response.text}")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {e}")
    
    def return_result(self, data):
        # return the vehicle results as a dictionary
        data['model'] = self.model_var.get().strip()
        data['mileage'] = self.mileage_entry.get().strip()
        data['variant'] = self.variant_var.get().strip()
        return data

    def display_result(self, data, model, mileage, variant, estimated_price, forecast):
        """Display vehicle details in the UI."""
        # Highlight the estimated price big and bold
        SEPARATOR = "----------------------------------------\n"
        self.result_text.set(
            f"Estimated Price: £{estimated_price:.2f}\n"
            f"{SEPARATOR}"
            f"Forecasted Prices: \n"
            f"2026: £{forecast.loc[2026]:.2f}\n"
            f"2027: £{forecast.loc[2027]:.2f}\n"
            f"{SEPARATOR}"
            f"Make: {data.get('make', 'N/A')}\n"
            f"Model: {model if model else 'N/A'}\n"
            f"Variant: {variant if variant else 'N/A'}\n"
            f"Mileage: {mileage if mileage else 'N/A'}\n"
            f"Colour: {data.get('colour', 'N/A')}\n"
            f"Year: {data.get('yearOfManufacture', 'N/A')}\n"
            f"Fuel: {data.get('fuelType', 'N/A')}\n"
            f"Tax: {data.get('taxStatus', 'N/A')} (Due: {data.get('taxDueDate', 'N/A')})\n"
            f"MOT: {data.get('motStatus', 'N/A')}\n"
            f"CO2 Emissions: {data.get('co2Emissions', 'N/A')} g/km\n"
            f"Engine Capacity: {data.get('engineCapacity', 'N/A')} cc\n"
            f"Euro Status: {data.get('euroStatus', 'N/A')}\n"
            f"Real Driving Emissions: {data.get('realDrivingEmissions', 'N/A')}\n"
            f"Wheelplan: {data.get('wheelplan', 'N/A')}\n"
            f"Revenue Weight: {data.get('revenueWeight', 'N/A')} kg\n"
            #f"Type Approval: {data.get('typeApproval', 'N/A')}\n"
            #f"Marked for Export: {'Yes' if data.get('markedForExport', False) else 'No'}\n"
            f"Month of First Registration: {data.get('monthOfFirstRegistration', 'N/A')}\n"
            f"Date of Last V5C Issued: {data.get('dateOfLastV5CIssued', 'N/A')}\n"
            #f"Art End Date: {data.get('artEndDate', 'N/A')}"
        )


if __name__ == "__main__":
    dataset = UKUsedCarDataset()
    dataset.download_dataset()
    car_data = dataset.load_data()
    # Drop some redundant columns
    car_data = car_data.drop(
        columns=[
            'Unnamed: 0', 
            'car_specs', 
            'car_seller', 
            'car_seller_rating', 
            'car_seller_location',
            'car_attention_grabber',
            'car_badges',
            'car_sub_title',
            'reg',
            'ulez',
            'engine_size',
            'car_title', 
            'body_type', 
            'part_warranty',
            'full_dealership', 
            'first_year_road_tax', 
            'finance_available', 
            'discounted',
            'part_service',
            # TODO: get these columns from the API
            'num_owner',
            'transmission',
        ]
    )
    # Drop rows where the year is not a number
    car_data = car_data[car_data['year'].str.isnumeric()]
    car_data['year'] = car_data['year'].astype(int)

    if car_data is not None:
        print("Loaded Data Sample:")
        print(car_data.head())

    root = tk.Tk()
    app = DVLAVehicleLookupApp(root)
    root.mainloop()
    print('done')