import pandas as pd
import numpy as np
import uuid
import os
import time
import logging
from predictor.tasks import run_prediction_task
import threading
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from pathlib import Path
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.shortcuts import redirect
from .forms import CSVUploadForm
from .utils import load_encoder, compute_molecular_properties, load_model, predict_data, load_scaler, load_cluster, safe_compute_properties
from .utils import property_columns
import tensorflow as tf  # Import TensorFlow
from celery.result import AsyncResult
from django.http import JsonResponse
from pathlib import Path  # Add this import at the top if not already imported




def delete_file_after_delay(file_path, delay):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted after {delay} seconds.")
    threading.Timer(delay, delete_file).start()



def check_task_status(request, task_id):
    result = AsyncResult(str(task_id))
    if result.ready():
        if result.successful():
            return JsonResponse({
                'ready': True,
                'status': 'success',
                'file': f"{settings.MEDIA_URL}predictions_{task_id}.csv"
            })
        elif result.state == 'FAILURE':
            return JsonResponse({
                'ready': True,
                'status': 'error',
                'message': str(result.result)
            })
    return JsonResponse({'ready': False})

# Background prediction function
def run_prediction_in_background(new_df, results_file_path):
    try:
        logging.info("Prediction started.")
        encoder = load_encoder()
        new_adducts = new_df["Adduct"].values.reshape(-1, 1)
        one_hot = encoder.transform(new_adducts)
        one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(["Adduct"]))
        df_encoded = pd.concat([new_df, one_hot_df], axis=1)

        logging.info("One-hot encoding completed.")

        properties_df = df_encoded.apply(
            lambda row: pd.Series(safe_compute_properties(row['Smiles'], row['Adduct'])),
            axis=1
        )
        properties_df.columns = property_columns
        df_encoded = pd.concat([df_encoded, properties_df], axis=1)

        logging.info("Properties computed.")

        # Load the TensorFlow model (not the TFLite interpreter)
        model = load_model()  # This should be a regular TensorFlow model
        scaler = load_scaler()
        kmeans = load_cluster()

        logging.info("Model, scaler, and k-means loaded.")

        # Pass the TensorFlow model to predict_data (not the TFLite interpreter)
        predictions_df = predict_data(model, df_encoded, scaler, kmeans)

        # Save the predictions to a CSV file
        predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')

        logging.info(f"Prediction written to {results_file_path}")

        # delete_file_after_delay(results_file_path, 300)  # Optional: clean up after 5 min
        logging.info("File cleanup scheduled.")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")


def predict(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            # # Validate file size
            # if uploaded_file.size > 2 * 1024 * 1024:  # 2MB limit
            #     form.add_error('file', 'The file size exceeds the limit of 2MB.')
            #     return render(request, 'predictor/upload.html', {'form': form})

            # Save uploaded file to media root
            media_root_path = Path(settings.MEDIA_ROOT)
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            full_path = media_root_path / Path(file_path)

            try:
                new_df = pd.read_csv(full_path)

                # # Validate row count (<= 10)
                # if len(new_df) > 10:
                #     form.add_error('file', 'The uploaded CSV should not have more than 10 rows.')
                #     return render(request, 'predictor/upload.html', {'form': form})

                # Prepare result file path and URL
                unique_filename = f'predictions_{uuid.uuid4().hex}.csv'
                results_file_path = media_root_path / Path(unique_filename)
                results_file_url = settings.MEDIA_URL + unique_filename

                # Run prediction in the background (no Celery in this case)
                run_prediction_in_background(new_df, results_file_path)

                # Return the result page with download link
                return render(request, 'predictor/results_done.html', {
                    'results_file_url': results_file_url,
                    'message': 'Prediction complete. You can download the result below.'
                })

            finally:
                if os.path.exists(full_path):
                    os.remove(full_path)
    else:
        form = CSVUploadForm()
    return render(request, 'predictor/upload.html', {'form': form})
# Other views
def redirect_to_predict(request):
    return redirect('predict', permanent=True)

def about(request):
    return render(request, 'predictor/about.html')

def contact(request):
    return render(request, 'predictor/contact.html')

def upload(request):
    return render(request, 'predictor/upload.html')
