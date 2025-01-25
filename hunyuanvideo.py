import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_enhance_for_hunyuanvideo, set_enhance_weight

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 为固定大小输入优化 CUDA 内核
torch.cuda.empty_cache()  # 清理 GPU 缓存

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit= True )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="FastVideo/hunyuan_diffusers",
    help="Path to the local model or Hugging Face model ID"
)
args = parser.parse_args()

model_id = args.model_path

# 加载 transformer
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
    quantization_config=quant_config,
)

# 创建 pipeline
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    revision="refs/pr/18",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.batch_size = 1
pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing(slice_size="max")

# ============ Enhance-A-Video ============
# comment the following if you want to use the original model
# inject_enhance_for_hunyuanvideo(pipe.transformer)
# enhance_weight can be adjusted for better visual quality
#set_enhance_weight(4)
# enable_enhance()
# ============ Enhance-A-Video ============

prompt = "A determined baseball player in a white and blue jersey grips his bat in the dugout. Sunlight casts dramatic shadows across his focused face. The blurred stadium background pulses with competitive energy."

output = pipe(
    prompt=prompt,
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
).frames[0]

# 导出视频
export_to_video(output, "output.mp4", fps=15)
