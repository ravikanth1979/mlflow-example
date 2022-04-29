import mlflow
import pandas as pd
import os
import numpy as np

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking import MlflowClient

import mlflow.pyfunc
mlflow.set_tracking_uri("http://ec2-34-243-250-201.eu-west-1.compute.amazonaws.com:5000/")
model_name = "my_experiment"
stage = 'Staging'
np.random.seed(40)

# Read the wine-quality csv file (make sure you're running this from the root of
# MLflow!)
wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data/wine-quality-test.csv")
data = pd.read_csv(wine_path)

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)
data = data.drop(["quality"], axis=1)
print(model.predict(data))







