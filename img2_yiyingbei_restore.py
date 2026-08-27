from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI


DEFAULT_BASE_URL = "https://api.v36.cm/v1"
DEFAULT_MODEL = "gpt-image-2-vip"
SOURCE_IMAGE = Path(r"C:\Users\yezhuohai\Pictures\乙瑛碑_分块\乙瑛碑_6.png")
STYLE_IMAGE = Path(
    r"D:\weixin_chat__record\xwechat_files\wxid_hwi8bvfkos3722_7cc2\temp\RWTemp\2026-06\af3fbd886ff12c50608e9d0752b9d890\eb2249c4f32866a3e0b83b278bd939d3.png"
)
DEFAULT_OUTPUT = Path(__file__).with_name("乙瑛碑_6_img2_修复.png")

PROMPT = """
任务：根据两张输入图生成一张修复后的碑帖文字图。

图1是内容来源。请从图1中提取可辨认的汉字、字形结构、行列位置和整体排布；只修复图1中已有或可由残存笔画明确推断的文字，不要新增无关文字。

图2是风格参考。请把图1中的文字修复成图2的视觉样式：纯黑背景，白色拓片字迹，高对比，字形清晰，保留汉隶碑刻的古拙笔意、蚕头雁尾、方折和拓片边缘质感。

处理要求：
1. 去除图1中的裂纹、污渍、纸张纹理、扫描噪声、灰底和杂散白斑。
2. 修补断裂、缺损、被遮挡的笔画，使字迹完整但仍像碑帖拓片，不要变成现代电脑字体。
3. 尽量保持图1原有字符的相对位置、行距、字距和横向裁切比例。
4. 输出只包含黑底白字的碑帖修复结果，不要添加说明文字、拼音、边框、印章、水印或装饰。
5. 如果某个字残缺严重无法确定，请按残存笔画风格做谨慎修复，不要臆造明显不同的字。
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use img2 through an OpenAI-compatible relay to restore image 1 in the style of image 2."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_IMAGE, help="内容来源图片，默认是乙瑛碑_6.png")
    parser.add_argument("--style", type=Path, default=STYLE_IMAGE, help="风格参考图片，默认是黑底白字样式图")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 PNG 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="图像模型，默认 img2；如中转使用官方名可改成 gpt-image-2")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI 兼容中转基址，默认 https://api.v36.cm/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("V36_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="API Key；默认读取 V36_API_KEY，其次读取 OPENAI_API_KEY",
    )
    parser.add_argument("--size", default="1536x1024", help="输出尺寸，默认 1536x1024；也可用 auto")
    parser.add_argument(
        "--quality",
        default="high",
        choices=["auto", "low", "medium", "high"],
        help="输出质量，默认 high",
    )
    return parser.parse_args()


def validate_image(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    if not path.is_file():
        raise ValueError(f"{label} 不是文件：{path}")


def main() -> None:
    args = parse_args()
    validate_image(args.source, "内容来源图片")
    validate_image(args.style, "风格参考图片")
    if not args.api_key:
        raise RuntimeError("缺少 API Key：请设置 V36_API_KEY 或 OPENAI_API_KEY，或使用 --api-key 传入。")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 这里会请求 {base_url}/images/edits。两张参考图需要 edits 接口；
    # {base_url}/images/generations 是纯文字生图接口，标准格式不能上传这两张图片。
    with args.source.open("rb") as source_file, args.style.open("rb") as style_file:
        result = client.images.edit(
            model=args.model,
            image=[source_file, style_file],
            prompt=PROMPT,
            size=args.size,
            quality=args.quality,
            output_format="png",
            background="opaque",
            input_fidelity="high",
        )

    image_base64 = result.data[0].b64_json
    if not image_base64:
        raise RuntimeError("接口没有返回 b64_json 图片数据。")

    args.output.write_bytes(base64.b64decode(image_base64))
    print(f"已保存：{args.output}")


if __name__ == "__main__":
    main()
