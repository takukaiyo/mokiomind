# -*- coding: utf-8 -*-
"""Download VLM models for the Yiyingbei splitter.

Examples:
  python download_florence2.py
  python download_florence2.py --model grounding-dino
  python download_florence2.py --local-dir D:\models\Florence-2-base
  python download_florence2.py --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VLM model files.")
    parser.add_argument("--model", choices=("florence2", "grounding-dino"), default="grounding-dino")
    parser.add_argument("--repo-id")
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com",
    )
    parser.add_argument("--resume", action="store_true", help="Resume partial downloads when supported.")
    parser.add_argument("--patch-compat", action="store_true", help="Patch local files for this Windows environment.")
    return parser.parse_args()


def patch_compat(local_dir: Path) -> None:
    config_path = local_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.setdefault("text_config", {})["forced_bos_token_id"] = config.get("bos_token_id", 0)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    configuration_path = local_dir / "configuration_florence2.py"
    if configuration_path.exists():
        text = configuration_path.read_text(encoding="utf-8-sig")
        text = text.replace(
            'if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):',
            'if getattr(self, "forced_bos_token_id", None) is None and kwargs.get("force_bos_token_to_be_generated", False):',
        )
        configuration_path.write_text(text, encoding="utf-8")

    modeling_path = local_dir / "modeling_florence2.py"
    if modeling_path.exists():
        text = modeling_path.read_text(encoding="utf-8-sig")
        marker = "Windows/local compatibility: do not require flash_attn"
        if marker not in text:
            text = text.replace(
                "from .configuration_florence2 import Florence2LanguageConfig\n",
                "from .configuration_florence2 import Florence2LanguageConfig\n\n"
                "# Windows/local compatibility: do not require flash_attn; eager attention is used.\n"
                "is_flash_attn_2_available = lambda: False\n"
                "is_flash_attn_greater_or_equal_2_10 = lambda: False\n",
            )
        text = text.replace(
            "    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa",
            "    pass  # flash_attn disabled on this Windows environment",
        )
        text = text.replace(
            "    from flash_attn import flash_attn_func, flash_attn_varlen_func",
            "    pass  # flash_attn disabled on this Windows environment",
        )
        modeling_path.write_text(text, encoding="utf-8")

    print(f"Patched compatibility files in {local_dir}")


def main() -> int:
    args = parse_args()
    defaults = {
        "florence2": ("microsoft/Florence-2-base", Path(r"D:\models\Florence-2-base")),
        "grounding-dino": ("IDEA-Research/grounding-dino-tiny", Path(r"D:\models\grounding-dino-tiny")),
    }
    default_repo, default_dir = defaults[args.model]
    repo_id = args.repo_id or default_repo
    local_dir = args.local_dir or default_dir

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install it with:\n"
            "  pip install huggingface_hub\n"
            "or run this script with the mokiomind .venv."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}")
    print(f"Target: {local_dir}")
    if args.endpoint:
        print(f"Endpoint: {args.endpoint}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=args.resume,
    )

    if args.patch_compat and args.model == "florence2":
        patch_compat(local_dir)

    print("\nDownload complete.")
    print(f"Use with: --vlm-model {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
