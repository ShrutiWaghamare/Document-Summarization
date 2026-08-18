"""Qwen LLM wrapper for document summarization."""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class Summarizer:
    SYSTEM_PROMPT = (
        "You are a professional document summarization assistant. Your task is to produce "
        "clear, accurate summaries of document sections and entire documents.\n\n"
        "FOCUS ON:\n"
        "- Main objectives and purpose of the document\n"
        "- Important findings and research results\n"
        "- Key facts, statistics, and evidence\n"
        "- Major challenges and risks identified\n"
        "- Recommendations and conclusions\n\n"
        "IGNORE COMPLETELY:\n"
        "- Acknowledgements and contributor lists\n"
        "- Author names and affiliations\n"
        "- Copyright and legal notices\n"
        "- Publication metadata\n"
        "- Table of contents and index\n"
        "- Appendices and supplementary material\n\n"
        "RULES:\n"
        "- Use ONLY information present in the provided text.\n"
        "- Do NOT invent or hallucinate facts.\n"
        "- Preserve numbers, percentages, and specific data points.\n"
        "- Maintain factual accuracy and logical flow.\n"
        "- Be concise but comprehensive."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        print(f"Loading model: {model_name}")

        # Use CPU-only execution per instructions (device_map='cpu')
        # Some models require `trust_remote_code=True`.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Auto-detect GPU: use CUDA if available, otherwise fall back to CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.max_length = None

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def summarize_with_instruction(self, text, instruction=None, max_length=300):
        """Summarize text with instruction using system prompt."""
        if not text:
            return ""

        system_instruction = self.SYSTEM_PROMPT
        user_instruction = instruction or "Summarize this document section clearly and accurately."

        prompt = (
            f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_instruction}\n\nDOCUMENT:\n{text}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        result = self.generator(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1
        )

        out = result[0]
        if isinstance(out, dict):
            return out.get("generated_text") or out.get("text") or ""
        return str(out)

    def summarize_text(self, text, max_length=300):
        """Quick summary with default instruction."""
        return self.summarize_with_instruction(
            text,
            "Summarize this passage in 2-3 sentences. Focus on main ideas and key facts.",
            max_length
        )

    def summarize_final(self, combined_text: str, max_length: int = 400) -> str:
        """Generate final comprehensive summary."""
        instruction = (
            "Create a concise, comprehensive summary of the provided document.\n\n"
            "Requirements:\n"
            "- Cover all major topics and key findings.\n"
            "- Preserve important facts, statistics, and conclusions.\n"
            "- Do not invent information not present in the document.\n"
            "- Do not repeat the same point multiple times.\n"
            "- Ignore author names, acknowledgements, copyright, and references unless critical.\n"
            "- Maintain balanced focus across all document sections.\n"
            "- Write in clear, professional language.\n"
            "- NO markdown formatting (no ##, ###, **, -, etc).\n"
            "- Output plain text only.\n\n"
            "Return the summary in 4-6 sentences covering:\n"
            "1. Main purpose of the document\n"
            "2. Major findings or key points\n"
            "3. Important facts or statistics\n"
            "4. Key challenges or limitations\n"
            "5. Recommendations or conclusions"
        )
        return self.summarize_with_instruction(combined_text, instruction, max_length)