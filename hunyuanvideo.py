import argparse
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_enhance_for_hunyuanvideo, set_enhance_weight

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", 
    type=str, 
    default="FastVideo/hunyuan_diffusers",
    help="Path to the local model or Hugging Face model ID"
)
args = parser.parse_args()

model_id = args.model_path
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision="refs/pr/18"
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, revision="refs/pr/18", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
# pipe.vae.enable_tiling()

# ============ Enhance-A-Video ============
# comment the following if you want to use the original model
inject_enhance_for_hunyuanvideo(pipe.transformer)
# enhance_weight can be adjusted for better visual quality
set_enhance_weight(4)
enable_enhance()
# ============ Enhance-A-Video ============

prompt = "A focused baseball player stands in the dugout, gripping his bat with determination, wearing a classic white jersey with blue pinstripes and a matching cap. The sunlight casts dramatic shadows across his face, highlighting his intense gaze as he prepares for the game. His hands, wrapped in black batting gloves, firmly hold the bat, showcasing his readiness and anticipation. The background reveals the bustling stadium, with blurred fans and vibrant green field, creating an atmosphere of excitement and competition. As he adjusts his stance, the player's concentration and passion for the sport are palpable, embodying the spirit of baseball."

output = pipe(
    prompt=prompt,
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(output, "output.mp4", fps=15)
