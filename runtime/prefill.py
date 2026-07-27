from string import templatelib
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

        print("\nStreaming Response:\n")

        response_text = ""  #to store the response

        response_tokens = 0  #to store the response tokens

        for chunk in self.loader.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()

            print(chunk.text, end="", flush=True)

            response_text = chunk.text

            response_tokens += 1

        generation_end = time.perf_counter()

        ####################################################
        # Metrics
        ####################################################

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

        return {
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "tokenization_time": tokenization_time,
            "ttft": ttft,
            "estimated_prefill": estimated_prefill,
            "decode_time": decode_time,
            "generation_time": total_generation,
            "tokens_per_second": tps,
        }