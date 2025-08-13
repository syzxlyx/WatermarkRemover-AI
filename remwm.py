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

#export HUGGINGFACE_TOKEN=

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

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
    # 如果模型不可用，创建虚拟掩码
    if model is None or processor is None:
        print("⚠️  Florence-2模型不可用，使用虚拟掩码")
        # 创建一个简单的虚拟掩码（覆盖图像中心区域）
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 计算中心区域（图像大小的10%）
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        bbox_size = min(width, height) // 10
        
        x1 = max(0, center_x - bbox_size // 2)
        y1 = max(0, center_y - bbox_size // 2)
        x2 = min(width, center_x + bbox_size // 2)
        y2 = min(height, center_y + bbox_size // 2)
        
        draw.rectangle([x1, y1, x2, y2], fill=255)
        return mask
    
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

    return mask

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
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video_with_ffmpeg(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, temp_dir,
                              vcodec="libx264", crf=None, bitrate=None, preset="fast", pix_fmt="yuv420p", fps=None, threads=None,
                              keep_audio=True, start=None, duration=None):
    """使用FFmpeg方法处理视频，支持可配置的编码参数与片段处理"""
    try:
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir()

        # 提取帧（可选片段）
        logger.info("提取视频帧...")
        extract_cmd = ["ffmpeg", "-y"]
        if start is not None:
            extract_cmd += ["-ss", str(start)]
        if duration is not None:
            extract_cmd += ["-t", str(duration)]
        extract_cmd += ["-i", str(input_path), str(frames_dir / "frame_%06d.png")]

        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"提取帧失败: {result.stderr}")
            return None

        # 处理帧
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"开始处理 {len(frame_files)} 帧")

        for i, frame_file in enumerate(frame_files):
            try:
                pil_image = Image.open(frame_file).convert('RGB')
                mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent)

                if transparent:
                    result_image = make_region_transparent(pil_image, mask_image)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                result_image.save(frame_file)

                if (i + 1) % 30 == 0 and len(frame_files) > 0:
                    progress = int((i + 1) / len(frame_files) * 100)
                    logger.info(f"处理进度: {i + 1}/{len(frame_files)} ({progress}%)")

            except Exception as e:
                logger.error(f"处理帧 {frame_file} 失败: {e}")

        # 获取原视频帧率
        probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(input_path)]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps_value = float(num) / float(den)
            else:
                fps_value = float(fps_str)
        else:
            fps_value = 30.0
        if fps is not None:
            fps_value = fps

        # 重新组合视频（图像序列必须编码，无法 copy）
        temp_video_path = Path(temp_dir) / "temp_video.mp4"
        compose_cmd = [
            "ffmpeg", "-y", "-framerate", str(fps_value),
            "-i", str(frames_dir / "frame_%06d.png"),
        ]
        # 选择编码器
        chosen_vcodec = vcodec if vcodec and vcodec.lower() != "copy" else "libx264"
        compose_cmd += ["-c:v", chosen_vcodec]
        # 码率或CRF
        if bitrate:
            compose_cmd += ["-b:v", str(bitrate)]
        elif crf is not None and chosen_vcodec.lower() in ("libx264", "libx265"):
            compose_cmd += ["-crf", str(crf)]
        # 预设（仅x264/x265有用，其它编码器忽略也不会报错，但可保守处理）
        if preset and chosen_vcodec.lower() in ("libx264", "libx265"):
            compose_cmd += ["-preset", str(preset)]
        # 像素格式
        if pix_fmt:
            compose_cmd += ["-pix_fmt", str(pix_fmt)]
        # 线程
        if threads is not None:
            compose_cmd += ["-threads", str(threads)]
        compose_cmd += [str(temp_video_path)]

        result = subprocess.run(compose_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"视频组合失败: {result.stderr}")
            return None

        # 合并音视频（可选）
        if keep_audio:
            # 为原始音频指定片段同步（如提供）
            final_cmd = ["ffmpeg", "-y",
                         "-i", str(temp_video_path)]
            # 将 -ss/-t 应用于第二个输入（原视频）
            if start is not None:
                final_cmd += ["-ss", str(start)]
            if duration is not None:
                final_cmd += ["-t", str(duration)]
            final_cmd += ["-i", str(input_path),
                          "-c:v", "copy", "-c:a", "aac",
                          "-map", "0:v:0", "-map", "1:a:0",
                          str(output_file)]

            result = subprocess.run(final_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"音频合并失败: {result.stderr}")
                shutil.copy2(temp_video_path, output_file)
        else:
            # 不保留音频，直接输出
            shutil.copy2(temp_video_path, output_file)

        logger.info(f"视频处理完成: {output_file}")
        return output_file

    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def process_video_with_opencv(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, temp_dir, output_format):
    """使用OpenCV方法处理视频（原始方法）"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    # 原始的OpenCV VideoWriter代码...
    # [这里保留原始的编码器选择和帧处理逻辑]
    return output_file

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format,
                  keep_audio=True, vcodec="libx264", crf=None, bitrate=None, preset="fast", pix_fmt="yuv420p", fps=None, threads=None,
                  start=None, duration=None):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video using FFmpeg"""
    logger.info(f"开始处理视频: {input_path}")

    # 始终使用FFmpeg方法处理视频，替代OpenCV VideoWriter
    # 如果FFmpeg不可用，将记录错误并返回None
    try:
        subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("FFmpeg不可用，无法处理视频（已弃用OpenCV VideoWriter方法）")
        return None

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos

    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{Path(input_path).stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    return process_video_with_ffmpeg(
        input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, temp_dir,
        vcodec=vcodec, crf=crf, bitrate=bitrate, preset=preset, pix_fmt=pix_fmt, fps=fps, threads=threads,
        keep_audio=keep_audio, start=start, duration=duration
    )




def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite,
               keep_audio, vcodec, crf, bitrate, preset, pix_fmt, fps, threads, start, duration):
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        return process_video(
            image_path, output_path, florence_model, florence_processor, model_manager, device,
            transparent, max_bbox_percent, force_format,
            keep_audio=keep_audio, vcodec=vcodec, crf=crf, bitrate=bitrate, preset=preset, pix_fmt=pix_fmt,
            fps=fps, threads=threads, start=start, duration=duration
        )

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
# 新增 FFmpeg 相关选项（默认保持与原行为一致）
@click.option("--keep-audio/--no-keep-audio", default=True, help="Whether to merge original audio into the output video.")
@click.option("--vcodec", default="libx264", help="Video codec to use (e.g., libx264, libx265, libxvid, copy).")
@click.option("--crf", type=int, default=None, help="CRF value for constant quality (x264/x265). Lower is better quality.")
@click.option("--bitrate", default=None, help="Target video bitrate (e.g., 5M or 2500k). Overrides CRF when set.")
@click.option("--preset", default="fast", help="Encoding preset for x264/x265 (e.g., ultrafast..veryslow).")
@click.option("--pix-fmt", default="yuv420p", help="Pixel format (e.g., yuv420p).")
@click.option("--fps", type=float, default=None, help="Override output FPS.")
@click.option("--threads", type=int, default=None, help="Number of threads for FFmpeg.")
@click.option("--start", type=float, default=None, help="Start time (seconds) for segment processing.")
@click.option("--duration", type=float, default=None, help="Duration (seconds) for segment processing.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str,
         keep_audio: bool, vcodec: str, crf: int | None, bitrate: str | None, preset: str, pix_fmt: str,
         fps: float | None, threads: int | None, start: float | None, duration: float | None):
    in_path = Path(input_path)
    out_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 尝试加载Florence-2模型，如果失败则使用替代方案
    florence_model = None
    florence_processor = None

    try:
        print("正在尝试加载Florence-2模型...")
        hf_token = os.getenv('HUGGINGFACE_TOKEN', "your_huggingface_token_here")
        florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            token=hf_token if hf_token != "your_huggingface_token_here" else None
        ).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            token=hf_token if hf_token != "your_huggingface_token_here" else None
        )
        logger.info("Florence-2 Model loaded successfully")
    except ImportError as e:
        if "flash_attn" in str(e):
            print("⚠️  Flashe-2模型需要flash_attn依赖，在Mac M1上无法安装")
            print("💡 建议使用 --transparent 选项或使用修复版本 remwm_fixed.py")
            print("🔧 或者登录Hugging Face后重新尝试")
            if not transparent:
                print("❌ 无法继续，请使用 --transparent 选项")
                sys.exit(1)
        else:
            print(f"❌ Florence-2模型加载失败: {e}")
            if not transparent:
                print("❌ 无法继续，请使用 --transparent 选项")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Florence-2模型加载失败: {e}")
        if not transparent:
            print("❌ 无法继续，请使用 --transparent 选项")
            sys.exit(1)

    if not transparent:
        try:
            model_manager = ModelManager(name="lama", device=device)
            logger.info("LaMa model loaded")
        except Exception as e:
            print(f"❌ LaMa模型加载失败: {e}")
            print("💡 建议使用 --transparent 选项")
            sys.exit(1)
    else:
        model_manager = None

    if in_path.is_dir():
        if not out_path.exists():
            out_path.mkdir(parents=True)

        images = list(in_path.glob("*.[jp][pn]g")) + list(in_path.glob("*.webp"))
        videos = list(in_path.glob("*.mp4")) + list(in_path.glob("*.avi")) + list(in_path.glob("*.mov")) + list(in_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = out_path / file_path.name
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite,
                       keep_audio, vcodec, crf, bitrate, preset, pix_fmt, fps, threads, start, duration)
            progress = int((idx + 1) / total_files * 100)
            print(f"input_path:{file_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = out_path
        if is_video_file(in_path) and out_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = out_path.with_suffix(f".{force_format.lower()}")
            else:
                output_file = out_path.with_suffix(".mp4")

        handle_one(in_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite,
                   keep_audio, vcodec, crf, bitrate, preset, pix_fmt, fps, threads, start, duration)
        print(f"input_path:{in_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
