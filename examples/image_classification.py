import asyncio
import os
import base64
from io import BytesIO
from PIL import Image
import requests
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,  # Key for persistence: keep worker alive
    "workers_max": 1,
    "name": "image-classifier"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "torchvision", "pillow", "requests"]
)
def initialize_model():
    """Initialize a pre-trained ResNet model for image classification."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    import json
    import os
    
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print(f"Is CUDA available: {is_cuda_available}, Device count: {device_count}")
    
    # Set device
    device = torch.device("cuda" if is_cuda_available else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Save model path (not actually saving the model, just a placeholder)
    model_info = {
        "model_type": "resnet50",
        "device": str(device),
        "cuda_available": is_cuda_available,
        "device_count": device_count
    }
    
    # Load ImageNet class labels
    labels_path = "/tmp/imagenet_labels.json"
    if not os.path.exists(labels_path):
        # If labels don't exist, create a simplified version (top 10 classes)
        # In a real scenario, you'd download the full ImageNet labels
        sample_labels = {
            "0": "tench",
            "1": "goldfish",
            "2": "great white shark",
            "3": "tiger shark",
            "4": "hammerhead shark",
            "5": "electric ray",
            "6": "stingray",
            "7": "rooster",
            "8": "hen",
            "9": "ostrich"
        }
        with open(labels_path, 'w') as f:
            json.dump(sample_labels, f)
    
    return {
        "status": "ready", 
        "model_info": model_info,
        "labels_path": labels_path
    }

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "torchvision", "pillow", "requests"]
)
def classify_image(image_url=None, image_base64=None):
    """Classify an image using the pre-trained model.
    
    Args:
        image_url (str, optional): URL of the image to classify.
        image_base64 (str, optional): Base64-encoded image data.
        
    Returns:
        dict: Classification results with top 5 predictions.
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import json
    import requests
    from io import BytesIO
    import base64
    
    # Check inputs
    if not image_url and not image_base64:
        return {"error": "Either image_url or image_base64 must be provided"}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load the image
    if image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Load ImageNet class labels
    with open("/tmp/imagenet_labels.json", 'r') as f:
        class_idx = json.load(f)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Convert to Python lists
        top5_prob = top5_prob.cpu().numpy().tolist()
        top5_idx = top5_idx.cpu().numpy().tolist()
        
        # Map indices to class labels
        top5_labels = [class_idx.get(str(idx), f"Class {idx}") for idx in top5_idx]
        
        # Create result dictionary
        results = [
            {"label": label, "probability": float(prob)}
            for label, prob in zip(top5_labels, top5_prob)
        ]
    
    return {
        "predictions": results,
        "device": str(device)
    }

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "torchvision", "pillow", "requests", "numpy"]
)
def process_image(image_url, operation="enhance"):
    """Process an image with various operations.
    
    Args:
        image_url (str): URL of the image to process.
        operation (str): Operation to perform (enhance, grayscale, blur).
        
    Returns:
        str: Base64-encoded processed image.
    """
    import torch
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F
    from PIL import Image, ImageFilter
    import requests
    from io import BytesIO
    import base64
    import numpy as np
    
    # Load the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Process the image based on the operation
    if operation == "enhance":
        # Enhance the image (increase contrast and brightness)
        img_tensor = transforms.ToTensor()(img)
        # Simple enhancement: adjust brightness and contrast
        enhanced = torch.clamp(1.2 * img_tensor, 0, 1)
        img = transforms.ToPILImage()(enhanced)
    
    elif operation == "grayscale":
        # Convert to grayscale
        img = F.to_grayscale(img, num_output_channels=3)
    
    elif operation == "blur":
        # Apply Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
    
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    # Convert the processed image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "processed_image": img_base64,
        "operation": operation,
        "original_size": img.size
    }

async def main():
    # Set the image URL
    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    
    # Step 1: Initialize the model
    print("Initializing model...")
    init_result = await initialize_model()
    print(f"Initialization result: {init_result}")
    
    # Step 2: Classify the image
    print("\nClassifying image...")
    classification_result = await classify_image(image_url=image_url)
    print("Classification results:")
    for pred in classification_result.get("predictions", []):
        print(f"  {pred['label']}: {pred['probability']:.4f}")
    
    # Step 3: Process the image
    print("\nProcessing image...")
    processing_result = await process_image(image_url, operation="enhance")
    
    # Step 4: Save the processed image
    if "processed_image" in processing_result:
        img_data = base64.b64decode(processing_result["processed_image"])
        with open("processed_image.png", "wb") as f:
            f.write(img_data)
        print(f"Processed image saved to processed_image.png")
        print(f"Operation: {processing_result['operation']}")
        print(f"Original size: {processing_result['original_size']}")

if __name__ == "__main__":
    asyncio.run(main())
