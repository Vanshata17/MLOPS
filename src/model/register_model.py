# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("MLOPS_KEY")
if not dagshub_token:
    raise EnvironmentError("MLOPS_KEY environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "vanshatajaiswal4"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "vanshatajaiswal4"
repo_name = "MLOPS"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
#mlflow.set_tracking_uri('https://dagshub.com/vanshatajaiswal4/MLOPS.mlflow')
#dagshub.init(repo_owner='vanshatajaiswal4', repo_name='MLOPS', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']

        client = mlflow.tracking.MlflowClient()

        # Get the artifact URI from the run and build the model source path
        run = client.get_run(run_id)
        artifact_uri = f"{run.info.artifact_uri}/{model_path}"

        # Create the registered model if it doesn't already exist
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass  # already exists

        # Create a model version directly from the artifact URI (bypasses LoggedModel lookup)
        model_version = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=run_id
        )

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        raise

if __name__ == '__main__':
    main()

