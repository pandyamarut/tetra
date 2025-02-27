# Tetra

Tetra is a framework for executing Python functions remotely on GPU resources. It simplifies the process of running machine learning workloads on remote servers.

## Installation

### Using Poetry (Recommended)

1. Make sure you have Poetry installed. If not, install it following the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

2. Clone the repository:
   ```bash
   git clone https://github.com/pandyamarut/tetra.git
   cd tetra
   ```

3. Install the package and its dependencies:
   ```bash
   poetry install
   ```

### Using Pip

1. Clone the repository:
   ```bash
   git clone https://github.com/pandyamarut/tetra.git
   cd tetra
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Configuration

To use the remote execution features with RunPod, you need to set up your RunPod API key:

1. Sign up for a RunPod account at [https://www.runpod.io/](https://www.runpod.io/)
2. Get your API key from the RunPod dashboard
3. Set the API key as an environment variable:
   ```bash
   export RUNPOD_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Example

The `examples/example.py` file demonstrates how to use Tetra to run a machine learning model on a remote GPU:

```python
import asyncio
import os
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "your-template-id",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,
    "workers_max": 1,
    "name": "simple-model-server"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["scikit-learn", "numpy", "torch"]
)
def initialize_model():
    # Function code runs on the remote server
    # ...

@remote(
    resource_config=gpu_config,
    resource_type="serverless"
)
def predict(features):
    # Make predictions using the model
    # ...

async def main():
    # Initialize the model
    init_result = await initialize_model()
    
    # Make predictions
    prediction = await predict([2.5, 3.5])
    print(prediction)

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Example

To run the example:

```bash
# Make sure your RUNPOD_API_KEY is set
export RUNPOD_API_KEY="your-api-key-here"

# Run the example
python examples/example.py
PYTHONPATH=/Users/rachfop/marut/tetra python examples/inference_example.py
PYTHONPATH=/Users/rachfop/marut/tetra python examples/tensor_reshape.py
PYTHONPATH=/Users/rachfop/marut/tetra python examples/image_classification.py
```

## License

MIT
