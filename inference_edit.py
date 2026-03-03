import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline,
    ModelConfig,
    MyQwenImagePipeline
)


def load_qwen_image_edit_pipeline() -> MyQwenImagePipeline:
    """
    加载带LoRA的Qwen-Image-Edit图像编辑流水线
    适配bfloat16精度，基于CUDA加速
    """
    # 模型配置：拆分Transformer/TextEncoder/VAE权重路径
    model_configs = [
        ModelConfig(
            model_id="Qwen/Qwen-Image-Edit",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="text_encoder/model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors"
        ),
    ]

    # 处理器配置
    processor_config = ModelConfig(
        model_id="Qwen/Qwen-Image-Edit",
        origin_file_pattern="processor/"
    )

    # 初始化流水线
    pipeline = MyQwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=None,
        processor_config=processor_config,
    )

    # 加载LoRA权重（去摩尔纹任务微调）
    lora_weight_path = "替换为模型地址"  # LoRA权重路径
    pipeline.load_lora(pipeline.dit, lora_weight_path)

    return pipeline


def infer_qwen_image_edit(
        pipeline: MyQwenImagePipeline,
        image_path: str,
        prompt: str,
        batch_size: int = 1,
        seed: int = 0,
        num_inference_steps: int = 40
) -> None:
    """
    执行Qwen-Image-Edit图像编辑推理（去摩尔纹任务）

    Args:
        pipeline: 初始化后的Qwen-Image-Edit流水线
        image_path: 输入图像路径（含摩尔纹的屏幕拍摄图）
        prompt: 编辑提示词（如"去除屏幕摩尔纹，保留原始纹理和色彩"）
        batch_size: 推理批次大小
        seed: 随机种子（保证结果可复现）
        num_inference_steps: 扩散推理步数
    """
    # 加载输入图像
    input_image = Image.open(image_path).convert("RGB")  # 确保RGB格式，避免通道问题
    img_width, img_height = input_image.size

    # 执行推理
    output_images = pipeline(
        prompt=prompt,
        edit_image=input_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=img_height,
        width=img_width,
        batch_size=batch_size
    )

    # 保存输出图像
    if batch_size > 1:
        for idx, img in enumerate(output_images):
            img.save(f"infer_result_{idx}.jpg")
    else:
        output_images.save("infer_result.jpg")


if __name__ == "__main__":
    # 1. 初始化流水线（仅需执行一次）
    qwen_pipeline = load_qwen_image_edit_pipeline()

    # 2. 配置推理参数（去摩尔纹任务）
    INPUT_IMAGE_PATH = "替换为图片地址"
    EDIT_PROMPT = "替换为prompt"
    # 3. 执行推理
    infer_qwen_image_edit(
        pipeline=qwen_pipeline,
        image_path=INPUT_IMAGE_PATH,
        prompt=EDIT_PROMPT,
        batch_size=1,
        seed=0,
        num_inference_steps=40
    )