import asyncio
import os
import numpy as np
import base64
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,  # Key for persistence: keep worker alive
    "workers_max": 1,
    "name": "tensor-operations"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "numpy"]
)
def demonstrate_reshape_operations():
    """
    Demonstrate various tensor reshape operations using PyTorch and NumPy.
    
    Reshaping a tensor changes its dimensions while preserving the number and order of elements.
    This means if you were to iterate over the tensor from major to minor dimension,
    the iteration order would remain the same.
    """
    import torch
    import numpy as np
    
    results = {}
    
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")
    print(f"Using device: {device}")
    
    # Example 1: Basic reshaping with PyTorch
    print("\nExample 1: Basic reshaping with PyTorch")
    # Create a sequence tensor from 1 to 12
    tensor_1d = torch.arange(1, 13).to(device)
    print(f"Original 1D tensor: {tensor_1d}")
    print(f"Shape: {tensor_1d.shape}")
    
    # Reshape to 3x4 matrix
    tensor_2d = tensor_1d.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{tensor_2d}")
    print(f"Shape: {tensor_2d.shape}")
    
    # Reshape to 4x3 matrix
    tensor_2d_alt = tensor_1d.reshape(4, 3)
    print(f"Reshaped to 4x3:\n{tensor_2d_alt}")
    print(f"Shape: {tensor_2d_alt.shape}")
    
    # Reshape to 2x2x3 tensor
    tensor_3d = tensor_1d.reshape(2, 2, 3)
    print(f"Reshaped to 2x2x3:\n{tensor_3d}")
    print(f"Shape: {tensor_3d.shape}")
    
    # Store results
    results["example1"] = {
        "original": tensor_1d.cpu().numpy().tolist(),
        "reshaped_3x4": tensor_2d.cpu().numpy().tolist(),
        "reshaped_4x3": tensor_2d_alt.cpu().numpy().tolist(),
        "reshaped_2x2x3": tensor_3d.cpu().numpy().tolist()
    }
    
    # Example 2: Using -1 dimension for automatic calculation
    print("\nExample 2: Using -1 dimension for automatic calculation")
    # Create a 3x4 tensor
    tensor_orig = torch.arange(1, 13).reshape(3, 4).to(device)
    print(f"Original tensor:\n{tensor_orig}")
    
    # Reshape using -1 to automatically determine one dimension
    tensor_auto = tensor_orig.reshape(-1, 6)  # Should be 2x6
    print(f"Reshaped to ?x6 (auto-calculated):\n{tensor_auto}")
    print(f"Shape: {tensor_auto.shape}")
    
    tensor_auto2 = tensor_orig.reshape(6, -1)  # Should be 6x2
    print(f"Reshaped to 6x? (auto-calculated):\n{tensor_auto2}")
    print(f"Shape: {tensor_auto2.shape}")
    
    # Store results
    results["example2"] = {
        "original": tensor_orig.cpu().numpy().tolist(),
        "reshaped_2x6": tensor_auto.cpu().numpy().tolist(),
        "reshaped_6x2": tensor_auto2.cpu().numpy().tolist()
    }
    
    # Example 3: Comparing reshape vs. transpose
    print("\nExample 3: Comparing reshape vs. transpose")
    # Create a 3x4 tensor
    tensor_orig = torch.arange(1, 13).reshape(3, 4).to(device)
    print(f"Original tensor (3x4):\n{tensor_orig}")
    
    # Reshape to 4x3
    tensor_reshaped = tensor_orig.reshape(4, 3)
    print(f"Reshaped to 4x3:\n{tensor_reshaped}")
    
    # Transpose
    tensor_transposed = tensor_orig.transpose(0, 1)  # or tensor_orig.T
    print(f"Transposed (4x3):\n{tensor_transposed}")
    
    # Demonstrate the difference
    print("\nDifference between reshape and transpose:")
    print("Reshape preserves the order of elements when viewed as a 1D array:")
    print(f"Original flattened: {tensor_orig.flatten()}")
    print(f"Reshaped flattened: {tensor_reshaped.flatten()}")
    print(f"Transposed flattened: {tensor_transposed.flatten()}")
    
    # Store results
    results["example3"] = {
        "original": tensor_orig.cpu().numpy().tolist(),
        "reshaped": tensor_reshaped.cpu().numpy().tolist(),
        "transposed": tensor_transposed.cpu().numpy().tolist(),
        "original_flattened": tensor_orig.flatten().cpu().numpy().tolist(),
        "reshaped_flattened": tensor_reshaped.flatten().cpu().numpy().tolist(),
        "transposed_flattened": tensor_transposed.flatten().cpu().numpy().tolist()
    }
    
    # Example 4: Reshaping with NumPy
    print("\nExample 4: Reshaping with NumPy")
    # Create a NumPy array
    np_array = np.arange(1, 13)
    print(f"Original NumPy array: {np_array}")
    
    # Reshape to 3x4
    np_reshaped = np_array.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{np_reshaped}")
    
    # Reshape with order='F' (Fortran-style, column-major)
    np_reshaped_f = np_array.reshape(3, 4, order='F')
    print(f"Reshaped to 3x4 with column-major order:\n{np_reshaped_f}")
    
    # Store results
    results["example4"] = {
        "original": np_array.tolist(),
        "reshaped_row_major": np_reshaped.tolist(),
        "reshaped_column_major": np_reshaped_f.tolist()
    }
    
    # Example 5: Practical application - Batch processing in deep learning
    print("\nExample 5: Practical application - Batch processing in deep learning")
    # Create a batch of 6 images, each 2x2 with 3 channels (RGB)
    batch = torch.arange(1, 73).reshape(6, 3, 2, 2).to(device)
    print(f"Batch shape: {batch.shape}")
    print("First image in the batch:")
    print(batch[0])
    
    # Reshape to process all pixels at once
    flattened_batch = batch.reshape(6, -1)  # Each image becomes a single row
    print(f"Reshaped for batch processing: {flattened_batch.shape}")
    print("First image flattened:")
    print(flattened_batch[0])
    
    # Reshape back to original shape
    restored_batch = flattened_batch.reshape(6, 3, 2, 2)
    print(f"Restored shape: {restored_batch.shape}")
    print("First image restored:")
    print(restored_batch[0])
    
    # Verify that the restored batch is identical to the original
    is_identical = torch.all(batch == restored_batch).item()
    print(f"Restored batch identical to original: {is_identical}")
    
    # Store results
    results["example5"] = {
        "batch_shape": list(batch.shape),
        "flattened_shape": list(flattened_batch.shape),
        "restored_shape": list(restored_batch.shape),
        "is_identical": is_identical
    }
    
    return {
        "results": results,
        "device": str(device),
        "cuda_available": is_cuda_available
    }

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "numpy", "matplotlib"]
)
def visualize_reshape_effects(shape1=(2, 6), shape2=(3, 4)):
    """
    Visualize how reshaping affects the arrangement of elements in a tensor.
    
    Args:
        shape1: First shape to demonstrate
        shape2: Second shape to demonstrate
        
    Returns:
        dict: Base64 encoded images showing the reshaping visualization
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom colormap for better visualization
    colors = [(0.8, 0.8, 1), (0.2, 0.2, 0.8)]  # Light blue to dark blue
    cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=12)
    
    # Create a sequence tensor from 1 to 12
    tensor = torch.arange(1, 13)
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reshape and plot the first shape
    tensor_shape1 = tensor.reshape(shape1)
    im1 = ax1.imshow(tensor_shape1.numpy(), cmap=cmap)
    ax1.set_title(f'Tensor reshaped to {shape1}')
    # Add text annotations with the original values
    for i in range(shape1[0]):
        for j in range(shape1[1]):
            text = ax1.text(j, i, tensor_shape1[i, j].item(),
                           ha="center", va="center", color="white" if tensor_shape1[i, j] > 6 else "black")
    
    # Reshape and plot the second shape
    tensor_shape2 = tensor.reshape(shape2)
    im2 = ax2.imshow(tensor_shape2.numpy(), cmap=cmap)
    ax2.set_title(f'Same tensor reshaped to {shape2}')
    # Add text annotations with the original values
    for i in range(shape2[0]):
        for j in range(shape2[1]):
            text = ax2.text(j, i, tensor_shape2[i, j].item(),
                           ha="center", va="center", color="white" if tensor_shape2[i, j] > 6 else "black")
    
    # Add a colorbar
    fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
    
    # Add a title explaining the concept
    fig.suptitle('Reshaping Preserves Element Order', fontsize=16)
    plt.figtext(0.5, 0.01, 
                'Note how the values maintain their sequence when reshaped.\n'
                'The elements are filled in row-major order (C-style) by default.',
                ha='center', fontsize=12)
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Create a second visualization showing the difference between reshape and transpose
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create a 3x4 tensor
    tensor_orig = torch.arange(1, 13).reshape(3, 4)
    
    # Plot original
    im1 = ax1.imshow(tensor_orig.numpy(), cmap=cmap)
    ax1.set_title('Original 3x4 Tensor')
    for i in range(3):
        for j in range(4):
            text = ax1.text(j, i, tensor_orig[i, j].item(),
                           ha="center", va="center", color="white" if tensor_orig[i, j] > 6 else "black")
    
    # Plot transposed
    tensor_transposed = tensor_orig.transpose(0, 1)
    im2 = ax2.imshow(tensor_transposed.numpy(), cmap=cmap)
    ax2.set_title('Transposed Tensor (4x3)')
    for i in range(4):
        for j in range(3):
            text = ax2.text(j, i, tensor_transposed[i, j].item(),
                           ha="center", va="center", color="white" if tensor_transposed[i, j] > 6 else "black")
    
    # Add a colorbar
    fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
    
    # Add a title explaining the concept
    fig.suptitle('Reshape vs. Transpose', fontsize=16)
    plt.figtext(0.5, 0.01, 
                'Transpose: Elements maintain their position in the grid but axes are swapped.\n'
                'Reshape: Elements are rearranged to fit the new shape while preserving sequence.',
                ha='center', fontsize=12)
    
    # Save the second plot to a bytes buffer
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100)
    buf2.seek(0)
    
    # Encode the second image as base64
    img2_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        "reshape_visualization": img_base64,
        "transpose_visualization": img2_base64,
        "shape1": shape1,
        "shape2": shape2
    }

async def main():
    # Step 1: Demonstrate various reshape operations
    print("Demonstrating tensor reshape operations...")
    reshape_results = await demonstrate_reshape_operations()
    print("\nReshape operations completed on device:", reshape_results["device"])
    print(f"CUDA available: {reshape_results['cuda_available']}")
    
    # Step 2: Visualize reshape effects
    print("\nGenerating visualizations...")
    visualization_results = await visualize_reshape_effects(shape1=(2, 6), shape2=(3, 4))
    
    # Step 3: Save the visualizations
    print("\nSaving visualizations...")
    
    # Save the reshape visualization
    reshape_img_data = base64.b64decode(visualization_results["reshape_visualization"])
    with open("reshape_visualization.png", "wb") as f:
        f.write(reshape_img_data)
    print("Saved reshape visualization to reshape_visualization.png")
    
    # Save the transpose vs reshape visualization
    transpose_img_data = base64.b64decode(visualization_results["transpose_visualization"])
    with open("transpose_vs_reshape.png", "wb") as f:
        f.write(transpose_img_data)
    print("Saved transpose vs reshape visualization to transpose_vs_reshape.png")
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
