import pandas as pd
import pickle
import numpy as np
import json

numeric_features = [
    'lead_time', 'no_of_adults', 'no_of_children',
    'no_of_week_nights', 'no_of_weekend_nights',
    'avg_price_per_room', 'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled', 'no_of_special_requests'
]

class PredictCancel(object):
    def __init__(self):
        # Scalers usados nos atributos finais
        self.scaler = pickle.load(open('parameter/hotelsmart_scaler.pkl', 'rb'))

        # Encoders
        self.market_segment_type = pickle.load(open('parameter/market_segment_type_encoder.pkl', 'rb'))

        # Imputer
        self.imputer_knn = pickle.load(open('../parameter/imputer_knn.pkl', 'rb'))

    def data_cleaning(self, df1):
        num_attributes = df1.select_dtypes(include=['int64', 'float64'])
        cat_attributes = df1.select_dtypes(include=['object', 'category'])

        if num_attributes.isnull().any().any():
            df_imputed = pd.DataFrame(
                self.imputer_knn.transform(num_attributes),
                columns=num_attributes.columns
            )
            df1 = pd.concat([df_imputed, cat_attributes], axis=1)
        return df1

    def feature_engineering(self, df2):
        df2["total_nights"] = df2["no_of_weekend_nights"] + df2["no_of_week_nights"]
        df2["total_guests"] = df2["no_of_adults"] + df2["no_of_children"]
        df2["lead_time_category"] = pd.cut(df2["lead_time"], bins=[-1, 7, 30, 90, 180, 365, 500],
                                        labels=["curto", "baixo", "medio", "alto", "muito_alto", "extremo"])
        df2["is_group_booking"] = (df2["total_guests"] >= 4).astype(int)
        df2["total_prev_interactions"] = df2["no_of_previous_cancellations"] + df2["no_of_previous_bookings_not_canceled"]
        df2["has_children"] = (df2["no_of_children"] > 0).astype(int)
        df2["has_meal_plan"] = (df2["type_of_meal_plan"] != "Not Selected").astype(int)
        df2["is_high_season"] = df2["arrival_month"].isin([1, 7, 8, 12]).astype(int)
        df2["had_prev_cancel"] = (df2["no_of_previous_cancellations"] > 0).astype(int)
        df2["had_prev_booking"] = (df2["no_of_previous_bookings_not_canceled"] > 0).astype(int)
        df2["is_repeated_guest"] = (df2["repeated_guest"] == 1).astype(int)

        return df2

    def data_preparation(self, df3):
        numeric_features = [
        'lead_time', 'no_of_adults', 'no_of_children',
        'no_of_week_nights', 'no_of_weekend_nights',
        'avg_price_per_room', 'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled', 'no_of_special_requests']
        df3[numeric_features] = self.scaler.transform(df3[numeric_features])

        # Encoders
        df3[['market_segment_type']] = self.market_segment_type.transform(df3[['market_segment_type']])

        boruta_columns = ['lead_time',
        'arrival_month',
        'arrival_date',
        'market_segment_type',
        'avg_price_per_room',
        'no_of_special_requests']

        return df3[boruta_columns]
    
    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred
        return original_data.to_json(orient='records')
