import asyncio
import os
import json
import base64
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,  # Key for persistence: keep worker alive
    "workers_max": 1,
    "name": "simple-inference-server"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "scikit-learn", "numpy", "matplotlib"]
)
def train_and_infer():
    """
    Train a simple machine learning model and perform inference.
    This example uses a RandomForestClassifier for simplicity.
    
    Returns:
        dict: Training and inference results
    """
    import torch
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib.pyplot as plt
    import io
    import base64
    import time
    
    # Check if CUDA is available (for demonstration purposes)
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")
    print(f"Using device: {device}")
    
    # Generate a synthetic classification dataset
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=2, 
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a RandomForestClassifier
    print("Training RandomForestClassifier...")
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.4f} seconds")
    
    # Perform inference on test data
    print("Performing inference...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Create a visualization of feature importances
    plt.figure(figsize=(10, 6))
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), feature_importances[indices], align='center')
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Create a visualization of predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.5)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend()
    plt.tight_layout()
    
    # Save the second plot to a bytes buffer
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=100)
    buf2.seek(0)
    
    # Encode the second image as base64
    img2_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()
    
    # Return results
    return {
        "model_info": {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y))
        },
        "performance": {
            "accuracy": float(accuracy),
            "training_time": training_time,
            "inference_time": inference_time,
            "samples_per_second": len(X_test) / inference_time
        },
        "feature_importance_visualization": img_base64,
        "prediction_visualization": img2_base64,
        "device_info": {
            "device": str(device),
            "cuda_available": is_cuda_available
        },
        # Include a small sample of predictions for demonstration
        "sample_predictions": {
            "actual": y_test[:10].tolist(),
            "predicted": y_pred[:10].tolist(),
            "probabilities": y_pred_proba[:10].tolist()
        }
    }

async def main():
    print("Starting simple inference example...")
    
    # Perform training and inference
    print("\nTraining model and performing inference...")
    results = await train_and_infer()
    
    # Print results
    print("\nInference Results:")
    print(f"Model: {results['model_info']['type']}")
    print(f"Accuracy: {results['performance']['accuracy']:.4f}")
    print(f"Training time: {results['performance']['training_time']:.4f} seconds")
    print(f"Inference time: {results['performance']['inference_time']:.4f} seconds")
    print(f"Samples per second: {results['performance']['samples_per_second']:.2f}")
    print(f"Device: {results['device_info']['device']}")
    print(f"CUDA available: {results['device_info']['cuda_available']}")
    
    # Save visualizations
    print("\nSaving visualizations...")
    
    # Save feature importance visualization
    feature_img_data = base64.b64decode(results["feature_importance_visualization"])
    with open("feature_importance.png", "wb") as f:
        f.write(feature_img_data)
    print("Saved feature importance visualization to feature_importance.png")
    
    # Save prediction visualization
    pred_img_data = base64.b64decode(results["prediction_visualization"])
    with open("predictions_vs_actual.png", "wb") as f:
        f.write(pred_img_data)
    print("Saved predictions visualization to predictions_vs_actual.png")
    
    # Save results to JSON
    print("\nSaving results to JSON...")
    with open("inference_results.json", "w") as f:
        # Create a simplified version of the results for JSON (excluding the large base64 images)
        simplified_results = {
            "model_info": results["model_info"],
            "performance": results["performance"],
            "device_info": results["device_info"],
            "sample_predictions": results["sample_predictions"]
        }
        json.dump(simplified_results, f, indent=2)
    print("Saved results to inference_results.json")
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
