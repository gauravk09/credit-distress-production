"""Put the chosen run's model on the registry shelf and label it @champion.

- registers run 04_clipped's model under the name 'credit-distress' (-> a version)
- tags that version with its serving threshold, so model + threshold travel together
- points the @champion alias at it, which is what the server will load
"""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

NAME = "credit-distress"
THRESHOLD = "0.52"

# find the run we want by its run name
run = mlflow.search_runs(
    experiment_names=["credit-distress"],
    filter_string="tags.mlflow.runName = '04_clipped'",
).iloc[0]
run_id = run["run_id"]

# register its logged model -> creates a new version
mv = mlflow.register_model(f"runs:/{run_id}/model", NAME)
client.set_model_version_tag(NAME, mv.version, "threshold", THRESHOLD)
client.set_registered_model_alias(NAME, "champion", mv.version)

print(f"registered {NAME} v{mv.version} from run {run_id[:8]}, alias @champion, threshold={THRESHOLD}")
