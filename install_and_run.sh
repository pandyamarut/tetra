#!/bin/bash

# This script installs the Tetra package and runs the example

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install the package using Poetry
echo "Installing Tetra using Poetry..."
poetry install

# Check if RUNPOD_API_KEY is set
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Warning: RUNPOD_API_KEY environment variable is not set."
    echo "You need to set it to run the example with remote execution."
    echo "Example: export RUNPOD_API_KEY=your-api-key-here"
    exit 1
fi

# Run the example
echo "Running the example..."
poetry run python examples/example.py
