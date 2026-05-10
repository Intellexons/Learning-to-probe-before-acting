"""Direct HuggingFace model generation."""
import json
import logging
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_for_generation(
    base_model_path: str,
    lora_base: str | None,
    iter_num: int,
    run_name: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and merge all LoRA adapters up to iter_num.

    For iter 0, loads just the base model.
    For iter N, loads base and merges adapters 0..N-1.
    """
    from pathlib import Path

    # Check base model exists
    base_path = Path(base_model_path)
    if not base_path.exists() or not any(base_path.iterdir()):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            f"Run 'bash scripts/00_setup_env.sh' to download it."
        )

    use_mps = torch.backends.mps.is_available()
    # Use float32 on MPS to avoid inf/nan during sampling with fp16 logits
    dtype = torch.float32 if use_mps else torch.bfloat16

    logger.info(f"Loading base model from {base_model_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if lora_base and iter_num > 0:
        lora_dir = Path(lora_base)
        if run_name:
            lora_dir = lora_dir / run_name
        for i in range(iter_num):
            adapter_path = str(lora_dir / f"iter_{i:03d}")
            if not Path(adapter_path).exists():
                raise FileNotFoundError(
                    f"LoRA adapter not found at {adapter_path}. "
                    f"Expected adapters 0..{iter_num - 1} to exist."
                )
            logger.info(f"  Merging adapter iter_{i:03d}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def extract_code(text: str) -> str:
    """Extract Python code from model response.

    Tries (in order):
    1. JSON with "code" field (legacy format)
    2. Fenced code block (```python ... ``` or ``` ... ```)
    3. Raw text as-is
    """
    # Try JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "code" in parsed:
            return parsed["code"].strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Try fenced code block (any tag). Closing ``` must not be followed by a
    # word char, so double-opener patterns like "```\n```policy\n..." don't
    # match the two openers as open/close and return empty content.
    m = re.search(r"```\w*\s*\n(.*?)```(?!\w)", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try to find the policy function definition and take everything from there
    m = re.search(r"((?:import\s+\w+.*\n)*def policy\b.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Return raw text as last resort
    return text.strip()


def generate_code(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 900,
) -> dict:
    """Generate a single code sample. Returns dict with code, success, error."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        code = extract_code(response)

        if not code:
            return {
                "code": "",
                "success": False,
                "error": "Empty code extracted from response",
                "raw_response": response,
            }
        return {
            "code": code,
            "success": True,
            "error": None,
            "raw_response": response,
        }
    except Exception as e:
        return {
            "code": "",
            "success": False,
            "error": f"Generation failed: {e}",
            "raw_response": "",
        }


def generate_code_from_messages(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 900,
) -> dict:
    """Generate a single code sample from a full message history.

    Accepts a list of {"role": ..., "content": ...} dicts so multi-turn
    callers can pass the complete conversation rather than a single prompt.
    Returns the same dict shape as generate_code.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        code = extract_code(response)

        if not code:
            return {
                "code": "",
                "success": False,
                "error": "Empty code extracted from response",
                "raw_response": response,
            }
        return {
            "code": code,
            "success": True,
            "error": None,
            "raw_response": response,
        }
    except Exception as e:
        return {
            "code": "",
            "success": False,
            "error": f"Generation failed: {e}",
            "raw_response": "",
        }


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n_samples: int,
    **kwargs,
) -> list[dict]:
    """Generate n_samples code candidates for the same prompt."""
    results = []
    for i in range(n_samples):
        logger.info(f"  Generating sample {i + 1}/{n_samples}")
        result = generate_code(model, tokenizer, prompt, **kwargs)
        results.append(result)
        if not result["success"]:
            logger.warning(f"  Sample {i + 1} failed: {result['error']}")
    return results
