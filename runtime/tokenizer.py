import logging
from typing import List, Dict, Any

# Configure module-level logger
logger = logging.getLogger(__name__)

class TokenizerExplorer:
    """
    An educational class to inspect, visualize, and interact with the tokenizer.
    
    It wraps a model's tokenizer to demonstrate chat templates, tokenization,
    and individual token decoding without running model inference.
    """

    def __init__(self, tokenizer: Any) -> None:
        """
        Initialize the TokenizerExplorer.

        Args:
            tokenizer (Any): A tokenizer instance (usually from transformers or mlx_lm).
        """
        self.tokenizer = tokenizer
        logger.info("TokenizerExplorer initialized with tokenizer type: %s", type(tokenizer).__name__)

    def format_prompt(self, prompt: str) -> str:
        """
        Apply the tokenizer's chat template to a user prompt string.

        Args:
            prompt (str): The raw input prompt.

        Returns:
            str: The chat-formatted prompt string. If no template is configured,
                 returns the raw prompt string.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            # Check if tokenizer supports applying chat templates
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return str(formatted)
        except Exception as e:
            logger.warning("Could not apply chat template, falling back to raw prompt. Error: %s", e)
            
        return prompt

    def tokenize(self, prompt: str) -> Dict[str, Any]:
        """
        Tokenize a raw prompt by applying chat templates first.

        Args:
            prompt (str): The raw input prompt.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "prompt": The original raw input prompt.
                - "formatted_prompt": The prompt after applying the chat template.
                - "token_ids": The list of integer token IDs.
                - "token_count": The total number of tokens.
        """
        formatted_prompt = self.format_prompt(prompt)
        
        # Encode the formatted prompt to token IDs
        # add_special_tokens=False is usually handled by apply_chat_template if templates are used
        token_ids = self.tokenizer.encode(formatted_prompt)
        token_count = len(token_ids)
        
        return {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "token_ids": token_ids,
            "token_count": token_count
        }

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Convert a list of token IDs back into their individual readable string representations.

        Args:
            token_ids (List[int]): A list of integer token IDs.

        Returns:
            List[str]: A list of readable token strings.
        """
        readable_tokens = []
        for tid in token_ids:
            # Decode each token individually to observe the text segmentation
            decoded = self.tokenizer.decode([tid])
            readable_tokens.append(decoded)
        return readable_tokens

if __name__ == "__main__":
    # Setup basic logging config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Load tokenizer using mlx_lm to test
    # pyrefly: ignore [missing-import]
    from mlx_lm import load
    
    logger.info("Loading Qwen-0.5B model to extract tokenizer...")
    _, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    
    explorer = TokenizerExplorer(tokenizer)
    
    test_prompt = "What is gravity? Answer in 3 words."
    print(f"\nOriginal Prompt: '{test_prompt}'")
    
    # 1. Test format_prompt
    formatted = explorer.format_prompt(test_prompt)
    print("\nFormatted Prompt (Chat Template):")
    print(repr(formatted))
    
    # 2. Test tokenize
    result = explorer.tokenize(test_prompt)
    print("\nTokenization Result:")
    print(f"  Token IDs  : {result['token_ids']}")
    print(f"  Token Count: {result['token_count']}")
    
    # 3. Test decode_tokens
    decoded = explorer.decode_tokens(result['token_ids'])
    print("\nDecoded Individual Tokens:")
    for idx, (tid, token) in enumerate(zip(result['token_ids'], decoded)):
        print(f"  [{idx:02d}] ID: {tid:<6d} -> {repr(token)}")
    print()
