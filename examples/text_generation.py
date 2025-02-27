import asyncio
import os
import json
from tetra import remote, get_global_client

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID if needed
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,
    "workers_max": 1,
    "name": "text-generation-server"
}

@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=[
        "torch", 
        "transformers", 
        "accelerate", 
        "sentencepiece", 
        "protobuf"
    ]
)
def generate_text(prompts, model_name="gpt2", max_length=100, temperature=0.7, num_return_sequences=1):
    """
    Generate text using a pre-trained language model.
    
    Args:
        prompts (list): List of text prompts to generate from
        model_name (str): Name of the pre-trained model to use
        max_length (int): Maximum length of generated text
        temperature (float): Temperature for sampling (higher = more random)
        num_return_sequences (int): Number of sequences to generate per prompt
        
    Returns:
        dict: Generated text and model information
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    loading_time = time.time() - start_time
    print(f"Model loaded in {loading_time:.2f} seconds")
    
    # Generate text for each prompt
    all_results = []
    total_tokens = 0
    total_generation_time = 0
    
    for prompt in prompts:
        print(f"Generating text for prompt: {prompt[:50]}...")
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = len(inputs["input_ids"][0])
        
        # Generate text
        generation_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - generation_start
        total_generation_time += generation_time
        
        # Decode the generated text
        generated_texts = []
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
            total_tokens += len(output) - input_length
        
        # Store results for this prompt
        prompt_result = {
            "prompt": prompt,
            "generated_texts": generated_texts,
            "generation_time": generation_time,
            "tokens_generated": sum(len(output) - input_length for output in outputs)
        }
        all_results.append(prompt_result)
    
    # Calculate performance metrics
    avg_generation_time = total_generation_time / len(prompts)
    tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
    
    return {
        "model_info": {
            "name": model_name,
            "loading_time": loading_time,
            "device": str(device)
        },
        "generation_results": all_results,
        "performance": {
            "average_generation_time": avg_generation_time,
            "tokens_per_second": tokens_per_second,
            "total_tokens_generated": total_tokens,
            "total_generation_time": total_generation_time
        }
    }

async def main():
    print("Starting text generation example...")
    
    # Define prompts for text generation
    prompts = [
        "Once upon a time in a distant galaxy,",
        "The artificial intelligence revolution began when",
        "The most effective way to solve climate change is",
        "In the year 2050, transportation will look like"
    ]
    
    # Generate text using different models
    models_to_try = ["gpt2", "distilgpt2"]
    
    for model_name in models_to_try:
        print(f"\nGenerating text with model: {model_name}")
        
        results = await generate_text(
            prompts=prompts,
            model_name=model_name,
            max_length=150,
            temperature=0.8,
            num_return_sequences=2
        )
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Model loaded in {results['model_info']['loading_time']:.2f} seconds on {results['model_info']['device']}")
        print(f"Average generation time: {results['performance']['average_generation_time']:.2f} seconds")
        print(f"Tokens per second: {results['performance']['tokens_per_second']:.2f}")
        print(f"Total tokens generated: {results['performance']['total_tokens_generated']}")
        
        # Print a sample of generated text for each prompt
        print("\nGenerated Text Samples:")
        for i, result in enumerate(results['generation_results']):
            print(f"\nPrompt {i+1}: {result['prompt']}")
            for j, text in enumerate(result['generated_texts'][:1]):  # Show only the first generation per prompt
                print(f"  Generation {j+1}: {text[:100]}...")
        
        # Save results to JSON
        output_file = f"{model_name}_generation_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
