import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

PREPROCESSOR_PATH = "model/scaler.pkl"

def preprocess_raw_data(df):
    df['event_date'] = df['event_time'].dt.date
    df['trip_id'] = df['shopper_id'].astype(str) + "_" + df['event_date'].astype(str)

    agg_funcs = {
        'event_type': [
            ('n_views', lambda x: (x == 'view').sum()),
            ('n_add_to_cart', lambda x: (x == 'add_to_cart').sum()),
            ('n_purchases', lambda x: (x == 'purchase').sum())
        ],
        'is_promo': [
            ('n_promo_interactions', 'sum'),
            ('promo_interaction_rate', 'mean')
        ],
        'price': [
            ('avg_price_viewed', 'mean'),
            ('min_price_viewed', 'min'),
            ('max_price_viewed', 'max'),
            ('price_range', lambda x: x.max() - x.min())
        ],
        'quantity': [
            ('total_quantity', 'sum')
        ],
        'device_type': [
            ('dominant_device', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown')
        ],
        'region': [
            ('region', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown')
        ],
        'shopper_id': [('shopper_id', 'first')],
        'event_date': [('trip_date', 'first')]
    }

    trip_df = df.groupby('trip_id').agg(agg_funcs)
    trip_df.columns = ['_'.join(col).strip() for col in trip_df.columns.values]

    trip_df['total_events'] = (
        trip_df['event_type_n_views'] +
        trip_df['event_type_n_add_to_cart'] +
        trip_df['event_type_n_purchases']
    )

    trip_df['view_to_cart_ratio'] = (
        trip_df['event_type_n_add_to_cart'] / (trip_df['event_type_n_views'] + 1e-6)
    )

    trip_df['cart_to_purchase_ratio'] = (
        trip_df['event_type_n_purchases'] / (trip_df['event_type_n_add_to_cart'] + 1e-6)
    )

    trip_df['converted'] = (trip_df['event_type_n_purchases'] > 0).astype(int)

    trip_df = trip_df.reset_index()

    trip_df = trip_df.join(pd.get_dummies(trip_df['device_type_dominant_device'], prefix='device_type', drop_first=True))
    trip_df = trip_df.join(pd.get_dummies(trip_df['region_region'], prefix='region', drop_first=True))

    X = trip_df.drop(columns=[
        'trip_id', 'price_price_range', 'quantity_total_quantity',
        'device_type_dominant_device', 'event_type_n_purchases', 'region_region',
        'shopper_id_shopper_id', 'event_date_trip_date', 'cart_to_purchase_ratio',
        'total_events', 'converted'
    ])
    y = trip_df['converted']

    return X, y

def get_correlated_features(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

def train_test_preprocess(X, y, corr_threshold=0.85):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    corr_cols = get_correlated_features(X_train, corr_threshold)
    X_train.drop(corr_cols, axis=1, inplace=True)
    X_test.drop(corr_cols, axis=1, inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save with pickle
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
