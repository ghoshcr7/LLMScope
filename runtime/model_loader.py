import logging
import time
from typing import Dict, Any
# pyrefly: ignore [missing-import]
from mlx_lm import load, generate, stream_generate
# pyrefly: ignore [missing-import]
from mlx_lm.sample_utils import make_sampler

# Configure module-level logger
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    A helper class to manage loading and text generation with MLX-compatible LLMs.
    
    This class wraps mlx_lm's loading and generation APIs to measure performance 
    metrics like loading latency and generation latency, logging key events.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize the ModelLoader and load the model and tokenizer.

        Args:
            model_path (str): The local folder path or Hugging Face hub repository 
                              identifier for the MLX model.
        """
        logger.info("Initializing ModelLoader for model path: %s", model_path)
        self.model_path = model_path
        
        # Measure model loading time
        start_time = time.perf_counter()
        
        # Load model and tokenizer using mlx_lm
        self.model, self.tokenizer = load(model_path)
        
        self.loading_time_seconds = time.perf_counter() - start_time
        logger.info("Model loaded successfully in %.4f seconds", self.loading_time_seconds)
        
        # Performance and metadata state variables
        self.model_name = model_path
        self.model_load_time = self.loading_time_seconds
        self.generation_time = 0.0

    def generate(self, prompt: str, max_tokens: int = 100, temp: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        Generate a text completion for the provided prompt.

        Args:
            prompt (str): The input text prompt for the model.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
            temp (float, optional): Sampling temperature. Defaults to 0.7.
            **kwargs: Extra arguments directly forwarded to mlx_lm.generate.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "model_name": Name of the loaded model.
                - "prompt": The original input prompt.
                - "response": The generated text response.
                - "prompt_tokens": Number of tokens in the prompt.
                - "response_tokens": Number of tokens in the response.
                - "generation_time": Time taken in seconds for generation.
                - "tokens_per_second": Throughput in tokens per second.
        """
        logger.info("Received generation request (max_tokens=%d, temp=%.2f)", max_tokens, temp)
        
        # Measure generation time
        start_time = time.perf_counter()
        
        # Create sampler from temp and top_p (or other kwargs)
        sampler = kwargs.pop("sampler", None)
        if sampler is None:
            top_p = kwargs.pop("top_p", 0.0)
            sampler = make_sampler(temp=temp, top_p=top_p)
            
        # Run inference using mlx_lm.generate
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler,
            **kwargs
        )
        
        self.generation_time = time.perf_counter() - start_time
        logger.info("Generation finished in %.4f seconds", self.generation_time)
        
        # Compute token counts
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        response_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
        
        # Calculate tokens per second
        tokens_per_second = response_tokens / self.generation_time if self.generation_time > 0 else 0.0
        
        return {
            "model_name": self.model_name,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "generation_time": self.generation_time,
            "tokens_per_second": tokens_per_second
        }
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temp: float = 0.7,
        **kwargs,
    ):
        """
        Stream generated text chunks from the model.

        Yields:
            GeneratedResponse objects from mlx_lm.stream_generate().
        """

        sampler = kwargs.pop("sampler", None)

        if sampler is None:
            top_p = kwargs.pop("top_p", 0.0)
            sampler = make_sampler(temp=temp, top_p=top_p)

        return stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **kwargs,
        )

if __name__ == "__main__":
    # Setup basic logging config when run as a standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    import sys
    # Allow specifying a model path as a command-line argument, defaulting to a lightweight Qwen model
    test_model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    logger.info("Running standalone model loader test with model: %s", test_model)
    try:
        loader = ModelLoader(test_model)
        result = loader.generate(
            prompt="Why is the sky blue? Answer in one sentence.",
            max_tokens=50
        )
        print("\n=== Test Results ===")
        print(f"Model: {result['model_name']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Response:\n{result['response'].strip()}")
        print(f"Prompt Tokens: {result['prompt_tokens']}")
        print(f"Response Tokens: {result['response_tokens']}")
        print(f"Generation Time: {result['generation_time']:.4f} seconds")
        print(f"Tokens Per Second: {result['tokens_per_second']:.2f} tok/s")
        print("====================\n")
    except Exception as e:
        logger.exception("An error occurred during testing:")
