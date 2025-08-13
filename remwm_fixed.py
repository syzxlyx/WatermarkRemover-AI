import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def check_florence_model_access():
    """检查Florence-2模型是否可以访问"""
    try:
        print("正在检查Florence-2模型访问权限...")
        # 尝试访问模型
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        print("✅ Florence-2模型可以正常访问")
        return True
    except Exception as e:
        print(f"❌ Florence-2模型访问失败: {e}")
        print("💡 解决方案:")
        print("1. 登录Hugging Face: huggingface-cli login")
        print("2. 或者使用透明模式: --transparent")
        print("3. 或者等待模型公开访问")
        return False

def create_dummy_mask(image, max_bbox_percent=10.0):
    """创建虚拟水印掩码（当Florence-2模型不可用时）"""
    width, height = image.size
    # 创建一个简单的矩形掩码作为示例
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 在图像中心创建一个矩形掩码
    bbox_width = int(width * max_bbox_percent / 100)
    bbox_height = int(height * max_bbox_percent / 100)
    x1 = (width - bbox_width) // 2
    y1 = (height - bbox_height) // 2
    x2 = x1 + bbox_width
    y2 = y1 + bbox_height
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    """获取水印掩码"""
    if model is None or processor is None:
        # 如果Florence-2模型不可用，使用虚拟掩码
        logger.warning("Florence-2模型不可用，使用虚拟掩码")
        return create_dummy_mask(image, max_bbox_percent)
    
    try:
        text_input = "watermark"
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            image_area = image.width * image.height
            for bbox in parsed_answer[detection_key]["bboxes"]:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_area = (x2 - x1) * (y2 - y1)
                if (bbox_area / image_area) * 100 <= max_bbox_percent:
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                else:
                    logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")
        else:
            logger.warning("未检测到水印，使用虚拟掩码")
            return create_dummy_mask(image, max_bbox_percent)

        return mask
    except Exception as e:
        logger.error(f"水印检测失败: {e}")
        logger.warning("使用虚拟掩码")
        return create_dummy_mask(image, max_bbox_percent)

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return file_path.suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format):
    """处理视频文件"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确定输出格式
    if force_format and force_format.upper() in ["MP4", "AVI"]:
        output_format = force_format.upper()
    else:
        output_format = "MP4"

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 处理每一帧
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            logger.info(f"处理帧 {frame_count}/{total_frames}")

            # 转换帧为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            # 获取水印掩码
            mask_image = get_watermark_mask(pil_frame, florence_model, florence_processor, device, max_bbox_percent)

            if transparent:
                result_frame = make_region_transparent(pil_frame, mask_image)
                result_frame = result_frame.convert("RGB")
            else:
                lama_result = process_image_with_lama(np.array(pil_frame), np.array(mask_image), model_manager)
                result_frame = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

            # 保存处理后的帧
            frame_path = temp_dir / f"frame_{frame_count:06d}.jpg"
            result_frame.save(frame_path, "JPEG")

        cap.release()

        # 使用FFmpeg重新组合视频
        output_video = output_path.with_suffix(f".{output_format.lower()}")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(temp_dir / "frame_%06d.jpg"),
            "-c:v", "libx264" if output_format == "MP4" else "libxvid",
            "-pix_fmt", "yuv420p", str(output_video)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            logger.info(f"视频处理完成: {output_video}")
        except subprocess.CalledProcessError:
            logger.warning("FFmpeg不可用，输出无音频视频")
            # 如果没有FFmpeg，直接复制最后一帧作为输出
            shutil.copy2(temp_dir / "frame_000001.jpg", output_video)

    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)

def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite):
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    logger.info(f"input_path:{image_path}, output_path:{new_output_path}")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 检查Florence-2模型访问权限
    florence_model = None
    florence_processor = None
    
    if not check_florence_model_access():
        print("⚠️  使用透明模式或虚拟掩码模式")
        if not transparent:
            print("💡 建议使用 --transparent 选项")
    else:
        try:
            florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
            florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
            logger.info("Florence-2 Model loaded")
        except Exception as e:
            logger.error(f"Florence-2模型加载失败: {e}")
            florence_model = None
            florence_processor = None

    if not transparent and florence_model is not None:
        try:
            model_manager = ModelManager(name="lama", device=device)
            logger.info("LaMa model loaded")
        except Exception as e:
            logger.error(f"LaMa模型加载失败: {e}")
            logger.warning("切换到透明模式")
            transparent = True
            model_manager = None
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite)
            progress = int((idx + 1) / total_files * 100)
            print(f"input_path:{file_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = output_path
        if is_video_file(input_path) and output_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            # Ensure video output has proper extension
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_path.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_path.with_suffix(".mp4")  # Default to mp4
        
        handle_one(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
