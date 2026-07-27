import logging
import sys
# pyrefly: ignore [missing-import]
from runtime.model_loader import ModelLoader
# pyrefly: ignore [missing-import]
from runtime.tokenizer import TokenizerExplorer
from runtime.prefill import PrefillAnalyzer

# Configure logging to keep console clean by default, but report issues
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LLMScope")

def main() -> None:
    """
    Main entry point for the interactive LLMScope CLI session.
    
    Loads the default Llama-3.2-1B-Instruct-4bit model and enters
    a read-eval-print loop (REPL) for text generation.
    """
    model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    print(f"Loading model: {model_name} ...")
    print("Note: If this is the first run, the model weights will be downloaded from Hugging Face.")
    
    try:
        loader = ModelLoader(model_name)
    except Exception as e:
        logger.error("Failed to initialize ModelLoader with model %s: %s", model_name, e)
        sys.exit(1)
        
    print("\nModel loaded successfully! Type 'exit' to quit.\n")
    
    while True:
        try:
            # Prompt the user for input
            prompt = input("Prompt: ")
            
            # Skip empty entries
            if not prompt.strip():
                continue
                
            # Exit check
            if prompt.strip().lower() == "exit":
                print("Goodbye!")
                break
                
            # Create a TokenizerExplorer for analysis before generation
            explorer = TokenizerExplorer(loader.tokenizer)
            analysis = explorer.tokenize(prompt)
            decoded_tokens = explorer.decode_tokens(analysis["token_ids"])
            
            # Print tokenization analysis
            print("\n===================================================\n")
            print("Prompt:")
            print(prompt)
            print()
            print("Formatted Prompt:")
            print(analysis["formatted_prompt"])
            print()
            print("Token Count:")
            print(analysis["token_count"])
            print()
            print("Token IDs:")
            print(analysis["token_ids"])
            print()
            print("Decoded Tokens:")
            print(decoded_tokens)
            print("\n===================================================\n")
            
            print("[Generating...]")
            
            # Generate the response
            profiler = PrefillAnalyzer(loader)

            result = profiler.analyze(
                prompt=prompt,
                max_tokens=256,
                temp=0.7
)
            
            # Extract clean model name for display
            model_display_name = loader.model_name.split("/")[-1]
            for suffix in ["-Instruct", "-4bit"]:
                if suffix in model_display_name:
                    model_display_name = model_display_name.split(suffix)[0]
            
            print("\n===================================================")
            print("LLMScope Runtime Profiler")
            print("===================================================\n")

            print(f"Model                 : {model_display_name}")
            print(f"Backend               : MLX")
            print(f"Model Load Time       : {loader.model_load_time:.2f} sec")

            print("\n---------------- Prefill ----------------\n")

            print(f"Prompt Tokens         : {result['prompt_tokens']}")

            print(
                f"Tokenization Time     : {result['tokenization_time']*1000:.2f} ms"
            )

            print(
                f"TTFT                  : {result['ttft']*1000:.2f} ms"
            )

            print(
                f"Estimated Prefill     : {result['estimated_prefill']*1000:.2f} ms"
            )

            print("\n---------------- Decode ----------------\n")

            print(f"Response Tokens       : {result['response_tokens']}")

            print(
                f"Decode Time           : {result['decode_time']:.2f} sec"
            )

            print(
                f"Generation Time       : {result['generation_time']:.2f} sec"
            )

            print(
                f"Tokens / Second       : {result['tokens_per_second']:.2f}"
            )

            print("\n===================================================\n")
                        
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error("An error occurred during prompt processing: %s", e)

if __name__ == "__main__":
    main()
