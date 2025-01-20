import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from azure.identity import ClientSecretCredential

# Initialize ML client
def init_ml_client():
    # Replace with your Azure details
    tenant_id = "c9b787f2-9b78-4285-bd4c-89a5d752d44b"            # Directory (Tenant) ID
    client_id = "2e13dc73-376f-4e9e-b148-f80db33b0fda"            # Application (Client) ID
    client_secret = "4tT8Q~NjXvVdvi2XNHlC2uh0hxO1ELIgtjRsmcEq"    # Client Secret
    subscription_id = "73f10b45-ec04-427e-a5d6-42ed4e79c537"  # Azure Subscription ID
    resource_group = "MLRG01"  # Azure Resource Group
    workspace_name = "AzureMLWS01"  # Azure ML Workspace Name
    credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret)

    return MLClient(credential, subscription_id, resource_group,workspace_name)

# Define the main function
def main(args):
    # Load data
    print("Loading data...")
    ml_client = init_ml_client()
    data_asset = ml_client.data.get(args.data, version="1")
    df = pd.read_csv(data_asset.path)
    # Set up MLflow
    mlflow.sklearn.autolog()

    # Log number of samples and features
    num_samples, num_features = df.shape
    print(f"Number of samples: {num_samples}, Number of features: {num_features}")

    # Split data
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_train_ratio, random_state=42)

    # Preprocess data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    # Train model based on chosen estimator
    if args.estimator == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.estimator == "svm":
        model = SVC(kernel=args.kernel, random_state=42)
    else:
        raise ValueError("Unsupported estimator. Choose either 'random_forest' or 'svm'.")

    with mlflow.start_run(run_name=f"Training_{args.estimator}") as run:
        print(f"Training model: {args.estimator}")
        model.fit(X_train, y_train)

        # Save the model using MLflow
        mlflow.sklearn.save_model(
            sk_model=model,
            path=args.registered_model_name
        )

        # Log the model using MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=args.registered_model_name,
            registered_model_name=args.registered_model_name,
        )

        # Log metrics
        mlflow.log_metric("num_samples", num_samples)
        mlflow.log_metric("num_features", num_features)

        print(f"Model logged and registered with MLflow as: {args.registered_model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model with Azure ML")

    # Add arguments
    parser.add_argument("--data", type=str, required=True, help="Name of the Azure ML data asset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train split ratio")
    parser.add_argument("--estimator", type=str, choices=["random_forest", "svm"], required=True, help="Type of estimator to use")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for SVM (only used if estimator is 'svm')")
    parser.add_argument("--registered_model_name", type=str, required=True, help="Name of the registered model")

    args = parser.parse_args()
    main(args)
