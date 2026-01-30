import runpod
import sys
import os
import base64
from io import BytesIO

# Add notebook directory to sys.path to import inference module
sys.path.append("notebook")

# Import Inference class (assuming it exists based on requirements)
try:
    from inference import Inference
except ImportError:
    print("Warning: Could not import Inference from notebook.inference. Make sure the project structure is correct.")
    # Placeholder for static check context
    Inference = None

# Global model initialization to load weights during container cold start
inference_model = None
CONFIG_PATH = "configs/inference-s3o-40k.yaml" # Assuming a default config path, adjust if needed

def init_model():
    global inference_model
    if inference_model is None:
        if Inference:
             # Assuming config_path is needed. 
             # If exact path isn't known, this might need adjustment by user.
             # Using a plausible default or requiring it in env.
             config_path = os.environ.get("INFERENCE_CONFIG_PATH", "configs/inference-s3o-40k.yaml")
             print(f"Initializing Inference model with config: {config_path}")
             inference_model = Inference(config_path, compile=False)
        else:
             print("Inference class not available.")

def handler(job):
    """
    Handler function for RunPod serverless worker.
    """
    job_input = job.get("input", {})
    
    # 1. Parse Input
    image_input = job_input.get("image")
    prompt_points = job_input.get("points") # list of [x, y] or similar
    prompt_mask = job_input.get("mask")
    seed = job_input.get("seed", 42)

    if not image_input:
        return {"error": "No image provided"}

    # 2. Run Inference
    try:
        # Assuming inference logic acts like the demo snippet
        # output = inference(image, mask, seed=42)
        
        # NOTE: logic to load_image or handle base64 would go here if not handled by Inference class.
        # For this implementation, we pass inputs directly assuming Inference handles or we need minimal prep.
        # If Inference expects PIL image:
        # from PIL import Image
        # import requests
        # if image_input.startswith("http"):
        #     image = Image.open(requests.get(image_input, stream=True).raw)
        # else:
        #     image = Image.open(BytesIO(base64.b64decode(image_input)))
        
        # Using the instantiated global model
        if inference_model:
            # Note: The prompt implies inference(image, mask, seed)
            # We pass what we have. Adjust argument mapping as per actual Inference signature.
            output = inference_model(image_input, prompt_mask, seed=seed)
            
            # 3. Process Output
            # output["gs"].save_ply(f"splat.ply")
            output_path = "splat.ply"
            if "gs" in output:
                output["gs"].save_ply(output_path)
            
            # Return file content as base64
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    ply_content = f.read()
                    ply_base64 = base64.b64encode(ply_content).decode("utf-8")
                return {"ply_base64": ply_base64}
            else:
                return {"error": "Output file not generated"}
        else:
             return {"error": "Model not initialized"}

    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}

# Initialize model on startup
init_model()

# Start RunPod worker
runpod.serverless.start({"handler": handler})
