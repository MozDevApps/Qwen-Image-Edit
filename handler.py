import runpod
import json
import subprocess
import time
import os
import requests
import base64
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configuration
COMFYUI_URL = "http://127.0.0.1:8188"
MAX_WAIT_TIME = 300  # Reduced to 5 minutes
POLL_INTERVAL = 1  # Check more frequently
STARTUP_TIMEOUT = 120  # 2 minutes for startup

def start_comfyui():
    """Start ComfyUI server in background"""
    print("Starting ComfyUI server...")
    print(f"Startup timeout: {STARTUP_TIMEOUT}s")
    
    # Start ComfyUI process
    process = subprocess.Popen([
        "python3", "main.py",
        "--listen", "0.0.0.0",
        "--port", "8188",
        "--preview-method", "none",  # Disable preview to save memory
        "--highvram"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for server to be ready
    max_attempts = STARTUP_TIMEOUT // 2
    for i in range(max_attempts):
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if response.status_code == 200:
                print("✓ ComfyUI server is ready")
                print(f"  Started in {i * 2} seconds")
                return True
        except Exception as e:
            if i % 5 == 0:
                print(f"  Waiting for ComfyUI... ({i * 2}s/{STARTUP_TIMEOUT}s)")
            time.sleep(2)
    
    # Print error logs if startup failed
    print("✗ Failed to start ComfyUI server")
    print("STDOUT:", process.stdout.read() if process.stdout else "N/A")
    print("STDERR:", process.stderr.read() if process.stderr else "N/A")
    return False

def save_base64_image(base64_str: str, filename: Optional[str] = None) -> str:
    """Save base64 image to ComfyUI input folder"""
    if not filename:
        filename = f"input_{uuid.uuid4().hex[:8]}.png"
    
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    input_path = Path(f"/workspace/ComfyUI/input/{filename}")
    input_path.write_bytes(img_bytes)
    
    return filename

def load_workflow_template() -> Dict:
    """Load workflow template from file"""
    workflow_path = Path("/workspace/ComfyUI/qwen_edit_workflow.json")
    
    if workflow_path.exists():
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    # Fallback to simple workflow
    return {
        "1": {
            "inputs": {
                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        "2": {
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image"
            },
            "class_type": "CLIPLoader"
        },
        "3": {
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "4": {
            "inputs": {
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0,
                "model": ["1", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "5": {
            "inputs": {
                "text": "Edit the image",
                "clip": ["2", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "6": {
            "inputs": {
                "text": "",
                "clip": ["2", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "8": {
            "inputs": {
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0]
            },
            "class_type": "KSampler"
        },
        "9": {
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0]
            },
            "class_type": "VAEDecode"
        },
        "10": {
            "inputs": {
                "filename_prefix": "qwen_output",
                "images": ["9", 0]
            },
            "class_type": "SaveImage"
        }
    }

def update_workflow_with_params(workflow: Dict, params: Dict) -> Dict:
    """Update workflow with user parameters"""
    
    # Update prompt if provided
    if 'prompt' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'CLIPTextEncode':
                if 'text' in node['inputs'] and node['inputs']['text'] != "":
                    node['inputs']['text'] = params['prompt']
                    break
    
    # Update seed if provided
    if 'seed' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['seed'] = params['seed']
    
    # Update steps if provided
    if 'steps' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['steps'] = params['steps']
    
    # Update denoise strength if provided
    if 'denoise' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['denoise'] = params['denoise']
    
    # Update CFG if provided
    if 'cfg' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['cfg'] = params['cfg']
    
    # Update image if provided
    if 'input_image_filename' in params:
        for node_id, node in workflow.items():
            if node.get('class_type') == 'LoadImage':
                node['inputs']['image'] = params['input_image_filename']
                break
    
    return workflow

def get_image_as_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

def wait_for_completion(prompt_id: str) -> Dict[str, Any]:
    """Wait for workflow completion and return results"""
    start_time = time.time()
    last_log_time = start_time
    
    print(f"Waiting for completion (max {MAX_WAIT_TIME}s)...")
    
    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            elapsed = time.time() - start_time
            
            # Log progress every 10 seconds
            if elapsed - (last_log_time - start_time) >= 10:
                print(f"  Still processing... ({int(elapsed)}s elapsed)")
                last_log_time = time.time()
            
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            
            if response.status_code == 200:
                history = response.json()
                
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    status = prompt_data.get('status', {})
                    
                    # Check if completed
                    if status.get('completed', False):
                        exec_time = time.time() - start_time
                        print(f"✓ Completed in {exec_time:.2f}s")
                        return {
                            'status': 'success',
                            'prompt_id': prompt_id,
                            'outputs': prompt_data.get('outputs', {}),
                            'execution_time': exec_time
                        }
                    
                    # Check for errors
                    if 'messages' in status:
                        for msg in status['messages']:
                            if isinstance(msg, (list, tuple)) and len(msg) > 0:
                                if msg[0] == 'execution_error':
                                    error_detail = msg[1] if len(msg) > 1 else 'Unknown error'
                                    return {
                                        'status': 'error',
                                        'error': f'Execution error: {error_detail}'
                                    }
            
            time.sleep(POLL_INTERVAL)
            
        except requests.Timeout:
            print("  Request timeout, retrying...")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"  Error checking status: {str(e)}")
            time.sleep(POLL_INTERVAL)
    
    return {
        'status': 'timeout',
        'error': f'Workflow execution exceeded {MAX_WAIT_TIME} seconds. Try using fewer steps or simpler prompts.'
    }

def handler(job):
    """
    Main handler function for RunPod
    This is called by RunPod with the job data
    """
    try:
        # Log for debugging
        print("=" * 70)
        print("New job received")
        print("=" * 70)
        
        # Get input data
        job_input = job.get("input", {})
        
        if not job_input:
            return {
                "error": "No input provided. Please provide 'input' with 'prompt' field."
            }
        
        print(f"Job input keys: {list(job_input.keys())}")
        
        # Load workflow template
        workflow = load_workflow_template()
        
        # Handle image input
        input_image_filename = None
        if 'image' in job_input:
            try:
                input_image_filename = save_base64_image(
                    job_input['image'],
                    job_input.get('image_filename', None)
                )
                print(f"✓ Saved input image: {input_image_filename}")
            except Exception as e:
                return {
                    'error': f'Failed to process input image: {str(e)}'
                }
        
        # Prepare parameters with defaults
        params = {
            'prompt': job_input.get('prompt', 'A beautiful landscape with mountains and lakes'),
            'seed': job_input.get('seed', int(time.time())),
            'steps': job_input.get('steps', 4),
            'denoise': job_input.get('denoise', 1.0 if not input_image_filename else 0.75),
            'cfg': job_input.get('cfg', 1.0)
        }
        
        if input_image_filename:
            params['input_image_filename'] = input_image_filename
        
        # Update workflow with parameters
        workflow = update_workflow_with_params(workflow, params)
        
        # Queue the workflow
        print(f"Queueing workflow...")
        print(f"  Prompt: {params['prompt']}")
        print(f"  Steps: {params['steps']}")
        print(f"  Seed: {params['seed']}")
        
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {
                "error": f"Failed to queue workflow: {response.text}"
            }
        
        result = response.json()
        prompt_id = result.get("prompt_id")
        
        if not prompt_id:
            return {
                "error": "No prompt_id returned from ComfyUI"
            }
        
        print(f"✓ Workflow queued: {prompt_id}")
        
        # Wait for completion
        completion_result = wait_for_completion(prompt_id)
        
        if completion_result['status'] != 'success':
            return completion_result
        
        # Process output images
        output_images = []
        outputs = completion_result.get('outputs', {})
        
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img in node_output['images']:
                    filename = img['filename']
                    subfolder = img.get('subfolder', '')
                    
                    if subfolder:
                        img_path = f"/workspace/ComfyUI/output/{subfolder}/{filename}"
                    else:
                        img_path = f"/workspace/ComfyUI/output/{filename}"
                    
                    img_base64 = get_image_as_base64(img_path)
                    
                    if img_base64:
                        output_images.append({
                            'filename': filename,
                            'image': img_base64
                        })
        
        print(f"✓ Generated {len(output_images)} image(s)")
        print("=" * 70)
        
        return {
            'status': 'success',
            'images': output_images,
            'execution_time': round(completion_result.get('execution_time', 0), 2),
            'parameters': params
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}")
        return {
            'error': str(e),
            'traceback': error_trace
        }

# Initialize ComfyUI on container start
print("=" * 70)
print("ComfyUI Qwen-Image-Edit-2509 Initializing")
print("=" * 70)

if not start_comfyui():
    print("FATAL: Failed to start ComfyUI server")
    exit(1)

print("\n" + "=" * 70)
print("RunPod Serverless Handler Ready ✓")
print("=" * 70)
print()

# Start RunPod serverless with the handler function
runpod.serverless.start({"handler": handler})        except Exception as e:
            if i % 10 == 0:
                print(f"Waiting for ComfyUI... ({i}/{max_attempts})")
            time.sleep(2)

    print("✗ Failed to start ComfyUI server")
    return False


def upload_images_to_comfyui(images: List[Dict[str, str]]) -> List[str]:
    """Upload base64 encoded images to ComfyUI input folder"""
    uploaded_files = []

    for idx, img_data in enumerate(images):
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(img_data['data'])
            filename = img_data.get('filename', f'input_image_{idx}.png')

            # Save to ComfyUI input directory
            input_path = Path(f"/workspace/ComfyUI/input/{filename}")
            input_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            input_path.write_bytes(img_bytes)

            uploaded_files.append(filename)
            print(f"✓ Uploaded image: {filename}")

        except Exception as e:
            print(f"✗ Failed to upload image {idx}: {str(e)}")

    return uploaded_files


def get_image_as_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None


def wait_for_completion(prompt_id: str) -> Dict[str, Any]:
    """Wait for workflow completion and return results"""
    start_time = time.time()

    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            # Check history
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")

            if response.status_code == 200:
                history = response.json()

                if prompt_id in history:
                    prompt_data = history[prompt_id]

                    # Check if completed
                    if prompt_data.get('status', {}).get('completed', False):
                        return {
                            'status': 'success',
                            'prompt_id': prompt_id,
                            'outputs': prompt_data.get('outputs', {}),
                            'execution_time': time.time() - start_time
                        }

                    # Check for errors
                    status_messages = prompt_data.get('status', {}).get('messages', [])
                    for msg in status_messages:
                        if msg[0] == 'execution_error':
                            return {
                                'status': 'error',
                                'error': msg[1]
                            }

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(POLL_INTERVAL)

    return {
        'status': 'timeout',
        'error': f'Workflow execution exceeded {MAX_WAIT_TIME} seconds'
    }


def process_workflow(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process a ComfyUI workflow from job input"""
    try:
        job_input = job.get("input", {})

        # Get workflow (either as dict or load from file)
        workflow = job_input.get("workflow")
        if not workflow:
            # Try loading default Qwen workflow
            with open('/workspace/ComfyUI/qwen_edit_workflow.json', 'r') as f:
                workflow = json.load(f)

        # Handle image inputs if provided
        images = job_input.get("images", [])
        if images:
            uploaded_files = upload_images_to_comfyui(images)
            print(f"Uploaded {len(uploaded_files)} images")

        # Update workflow with custom prompt if provided
        prompt = job_input.get("prompt")
        if prompt and isinstance(workflow, dict):
            # Find CLIPTextEncode nodes and update prompt
            for node_id, node in workflow.items():
                if node.get("class_type") == "CLIPTextEncode":
                    node["inputs"]["text"] = prompt

        # Queue the workflow
        print("Queueing workflow...")
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow}
        )

        if response.status_code != 200:
            return {
                "status": "error",
                "error": f"Failed to queue workflow: {response.text}"
            }

        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"✓ Workflow queued with ID: {prompt_id}")

        # Wait for completion
        completion_result = wait_for_completion(prompt_id)

        if completion_result['status'] != 'success':
            return completion_result

        # Process output images
        output_images = []
        outputs = completion_result.get('outputs', {})

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img in node_output['images']:
                    filename = img['filename']
                    subfolder = img.get('subfolder', '')

                    # Build full path
                    if subfolder:
                        img_path = f"/workspace/ComfyUI/output/{subfolder}/{filename}"
                    else:
                        img_path = f"/workspace/ComfyUI/output/{filename}"

                    # Encode to base64
                    img_base64 = get_image_as_base64(img_path)

                    if img_base64:
                        output_images.append({
                            'filename': filename,
                            'data': img_base64
                        })

        return {
            'status': 'success',
            'prompt_id': prompt_id,
            'images': output_images,
            'execution_time': completion_result.get('execution_time', 0)
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# Initialize ComfyUI on container start
print("=" * 60)
print("Initializing ComfyUI with Qwen-Image-Edit-2509")
print("=" * 60)

if not start_comfyui():
    print("Failed to start ComfyUI server")
    exit(1)

print("\n" + "=" * 60)
print("RunPod Serverless Handler Ready")
print("=" * 60)

# Start RunPod serverless handler
runpod.serverless.start({"handler": process_workflow})
