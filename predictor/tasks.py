import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
from .utils import (
    load_encoder, load_model, load_scaler, load_cluster, predict_data,
    safe_compute_properties, property_columns
)
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
import logging
import pandas as pd
from .utils import (
    load_encoder, load_model, load_scaler, load_cluster, predict_data,
    safe_compute_properties, property_columns
)
from pathlib import Path

@shared_task
def run_prediction_task(data_dict, results_file_path):
    try:
        logging.info("Prediction task started.")
        new_df = pd.DataFrame(data_dict)

        # One-hot encode the "Adduct" column
        encoder = load_encoder()
        new_adducts = new_df["Adduct"].values.reshape(-1, 1)
        one_hot = encoder.transform(new_adducts)
        one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(["Adduct"]))
        df_encoded = pd.concat([new_df, one_hot_df], axis=1)

        # Compute properties
        properties_df = df_encoded.apply(
            lambda row: pd.Series(safe_compute_properties(row['Smiles'], row['Adduct'])),
            axis=1
        )
        properties_df.columns = property_columns
        df_encoded = pd.concat([df_encoded, properties_df], axis=1)

        # Load the model and other necessary components inside the task
        model = load_model()  # Load the TensorFlow model (not the TFLite interpreter)
        scaler = load_scaler()  # Load the scaler
        kmeans = load_cluster()  # Load k-means clustering model

        logging.info("Model, scaler, and k-means loaded.")

        # Generate predictions
        predictions_df = predict_data(model, df_encoded, scaler, kmeans)

        # Save the predictions to a CSV file
        predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')

        logging.info(f"Prediction saved to: {results_file_path}")

    except Exception as e:
        logging.error(f"Celery prediction task failed: {e}")