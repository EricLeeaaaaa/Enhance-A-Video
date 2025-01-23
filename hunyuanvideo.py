import argparse
import json
import os
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from enhance_a_video import enable_enhance, inject_enhance_for_hunyuanvideo, set_enhance_weight


def create_pipeline(model_id, torch_dtype=torch.bfloat16, revision="refs/pr/18"):
    """创建 HunyuanVideo pipeline"""
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch_dtype, revision=revision
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id, transformer=transformer, revision=revision, torch_dtype=torch_dtype
    )
    return pipe


def load_lora(pipe, lora_checkpoint_dir):
    """加载 LoRA 权重"""
    print(f"Loading LoRA weights from {lora_checkpoint_dir}")
    config_path = os.path.join(lora_checkpoint_dir, "lora_config.json")
    with open(config_path, "r") as f:
        lora_config_dict = json.load(f)
    rank = lora_config_dict["lora_params"]["lora_rank"]
    lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
    lora_scaling = lora_alpha / rank
    pipe.load_lora_weights(lora_checkpoint_dir, adapter_name="default")
    pipe.set_adapters(["default"], [lora_scaling])
    print(f"Successfully Loaded LoRA weights from {lora_checkpoint_dir}")
    return pipe


def setup_pipeline(pipe):
    """设置 pipeline 的基本配置"""
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def setup_enhance(pipe, enhance_weight=4):
    """设置 Enhance-A-Video"""
    inject_enhance_for_hunyuanvideo(pipe.transformer)
    set_enhance_weight(enhance_weight)
    enable_enhance()
    return pipe


def generate_video(pipe, prompt, height=720, width=1280, num_frames=129, 
                  num_inference_steps=50, seed=42, fps=15):
    """生成视频"""
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    
    export_to_video(output, "output.mp4", fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A focused baseball player stands in the dugout, gripping his bat with determination, wearing a classic white jersey with blue pinstripes and a matching cap. The sunlight casts dramatic shadows across his face, highlighting his intense gaze as he prepares for the game. His hands, wrapped in black batting gloves, firmly hold the bat, showcasing his readiness and anticipation. The background reveals the bustling stadium, with blurred fans and vibrant green field, creating an atmosphere of excitement and competition. As he adjusts his stance, the player's concentration and passion for the sport are palpable, embodying the spirit of baseball.",
        help="Prompt for video generation",
    )
    args = parser.parse_args()

    # 创建并设置 pipeline
    model_id = "tencent/HunyuanVideo"
    pipe = create_pipeline(model_id)
    pipe = setup_pipeline(pipe)
    
    # 加载 LoRA（如果指定）
    if args.lora_checkpoint_dir is not None:
        pipe = load_lora(pipe, args.lora_checkpoint_dir)
    
    # 设置 Enhance-A-Video
    pipe = setup_enhance(pipe)
    
    # 生成视频
    generate_video(pipe, args.prompt)


if __name__ == "__main__":
    main()
