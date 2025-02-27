import asyncio
import os
import base64
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",
    "gpu_ids": "AMPERE_16",
    "workers_min": 1,
    "workers_max": 1,
    "name": "simple-tensor-reshape"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "matplotlib"]
)
def simple_reshape_demo():
    """
    A simple demonstration of tensor reshaping operations with visualization.
    """
    import torch
    import matplotlib.pyplot as plt
    import io
    import base64
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a sequence tensor from 1 to 12
    tensor = torch.arange(1, 13).to(device)
    print(f"Original 1D tensor: {tensor}")
    
    # Reshape to 3x4 matrix
    tensor_3x4 = tensor.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{tensor_3x4}")
    
    # Reshape to 2x6 matrix
    tensor_2x6 = tensor.reshape(2, 6)
    print(f"Reshaped to 2x6:\n{tensor_2x6}")
    
    # Reshape to 4x3 matrix
    tensor_4x3 = tensor.reshape(4, 3)
    print(f"Reshaped to 4x3:\n{tensor_4x3}")
    
    # Create a visualization
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    
    # Plot the 3x4 tensor
    axs[0].imshow(tensor_3x4.cpu().numpy(), cmap='Blues')
    axs[0].set_title('Tensor reshaped to 3x4')
    for i in range(3):
        for j in range(4):
            axs[0].text(j, i, tensor_3x4[i, j].item(),
                      ha="center", va="center", color="white" if tensor_3x4[i, j] > 6 else "black")
    
    # Plot the 2x6 tensor
    axs[1].imshow(tensor_2x6.cpu().numpy(), cmap='Blues')
    axs[1].set_title('Tensor reshaped to 2x6')
    for i in range(2):
        for j in range(6):
            axs[1].text(j, i, tensor_2x6[i, j].item(),
                      ha="center", va="center", color="white" if tensor_2x6[i, j] > 6 else "black")
    
    # Plot the 4x3 tensor
    axs[2].imshow(tensor_4x3.cpu().numpy(), cmap='Blues')
    axs[2].set_title('Tensor reshaped to 4x3')
    for i in range(4):
        for j in range(3):
            axs[2].text(j, i, tensor_4x3[i, j].item(),
                      ha="center", va="center", color="white" if tensor_4x3[i, j] > 6 else "black")
    
    fig.suptitle('Tensor Reshape Demonstration', fontsize=16)
    plt.figtext(0.5, 0.01, 
                'Notice how the values maintain their sequence when reshaped.\n'
                'The elements are filled in row-major order by default.',
                ha='center', fontsize=12)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        "visualization": img_base64,
        "device_info": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available()
        },
        "shapes": {
            "original": list(tensor.shape),
            "shape_3x4": list(tensor_3x4.shape),
            "shape_2x6": list(tensor_2x6.shape),
            "shape_4x3": list(tensor_4x3.shape)
        }
    }

async def main():
    print("Running simple tensor reshape demonstration on remote GPU...")
    
    # Run the reshape demo
    results = await simple_reshape_demo()
    
    # Print device information
    print(f"\nRan on device: {results['device_info']['device']}")
    print(f"CUDA available: {results['device_info']['cuda_available']}")
    
    # Save the visualization
    print("\nSaving visualization...")
    img_data = base64.b64decode(results["visualization"])
    with open("tensor_reshape_demo.png", "wb") as f:
        f.write(img_data)
    print("Saved visualization to tensor_reshape_demo.png")
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
