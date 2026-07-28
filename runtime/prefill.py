import time


class PrefillAnalyzer:
    """
    Measures tokenization, TTFT, decode latency and throughput.
    """

    def __init__(self, loader):
        self.loader = loader

    def analyze(
        self,
        prompt: str,
        max_tokens: int = 256,
        temp: float = 0.7,
    ):
        # Format prompt with chat template if not already formatted
        if hasattr(self.loader.tokenizer, "apply_chat_template") and not (
            prompt.startswith("<|") or prompt.startswith("[INST]")
        ):
            try:
                formatted = self.loader.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt = str(formatted)
            except Exception:
                pass

        ####################################################
        # Tokenization
        ####################################################

        t0 = time.perf_counter()

        prompt_tokens = len(
            self.loader.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )

        t1 = time.perf_counter()

        tokenization_time = t1 - t0

        ####################################################
        # Streaming generation
        ####################################################

        generation_start = time.perf_counter()

        first_token_time = None

        response_text = ""
        response_tokens = 0

        finish_reason_raw = None
        for chunk in self.loader.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()

            response_text += chunk.text

            response_tokens += 1

            chunk_finish = getattr(chunk, "finish_reason", None)
            if chunk_finish:
                finish_reason_raw = chunk_finish

        generation_end = time.perf_counter()

        ####################################################
        # Metrics
        ####################################################

        if first_token_time is None:
            first_token_time = generation_end

        ttft = first_token_time - generation_start

        decode_time = generation_end - first_token_time

        total_generation = generation_end - generation_start

        estimated_prefill = max(
            0,
            ttft - tokenization_time,
        )

        tps = (
            response_tokens / decode_time
            if decode_time > 0
            else 0
        )

        # Format human-readable finish reason
        if finish_reason_raw == "length" or response_tokens >= max_tokens:
            finish_reason = "Maximum Token Limit Reached"
        elif finish_reason_raw == "stop":
            finish_reason = "Stop Token"
        elif finish_reason_raw in ("eos", "eos_token"):
            finish_reason = "End of Sequence"
        elif finish_reason_raw:
            finish_reason = str(finish_reason_raw)
        else:
            finish_reason = "Stop Token"

        return {
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "max_tokens": max_tokens,
            "tokenization_time": tokenization_time,
            "ttft": ttft,
            "estimated_prefill": estimated_prefill,
            "decode_time": decode_time,
            "generation_time": total_generation,
            "tokens_per_second": tps,
            "finish_reason": finish_reason,
        }