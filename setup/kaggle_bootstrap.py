"""Paste into the first cell of a Kaggle notebook (Settings -> Accelerator -> GPU T4 x2).

IMPORTANT T4 CAVEATS (Kaggle's default GPU is Turing, sm_75):
  * bf16 is NOT supported -> use fp16 everywhere
  * flash-attention 2 requires Ampere+ -> will not install/run
  * SGLang targets newer architectures and now requires CUDA 13; it may not
    work on T4 at all. Use Kaggle for TRAINING only and serve via SGLang on
    Modal, or fall back to vLLM for rollouts here.

Kaggle free tier: ~30 GPU-hours/week. Sessions die after 12h and on idle.
Checkpoint to /kaggle/working every N steps or you will lose the run.
"""

BOOTSTRAP = r"""
!pip install -q --upgrade pip
!pip install -q trl peft accelerate datasets auditkit
!pip install -q aligntune

import torch, subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())   # False on T4 -> use fp16
"""

if __name__ == "__main__":
    print(BOOTSTRAP)
