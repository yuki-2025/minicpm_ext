import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download 
from huggingface_hub import login
login(token=' ')
# Set the model repo name
repo_id = "openbmb/MiniCPM-o-2_6"
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # ID of the model repository
 # Set the save path
save_dir = "/ai-video-sh/chang.yin/new/models/MiniCPM-o-2_6"
# save_dir = "models/DeepSeek-R1-Distill-Qwen-32B"
# deepseek r1, qwen2.5, llama-3.2-3b, mistral

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Download all files to the specified folder
snapshot_download(
    repo_id=repo_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False  # Ensure that full files are copied rather than using symlinks
)

print(f"All files have been downloaded to: {os.path.abspath(save_dir)}")
 
