import logging
import sys
# pyrefly: ignore [missing-import]
from runtime.model_loader import ModelLoader

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
                
            print("\n[Generating...]")
            
            # Generate the response
            result = loader.generate(
                prompt=prompt,
                max_tokens=256,
                temp=0.7
            )
            
            # Extract clean model name for display
            model_display_name = loader.model_name.split("/")[-1]
            for suffix in ["-Instruct", "-4bit"]:
                if suffix in model_display_name:
                    model_display_name = model_display_name.split(suffix)[0]
            
            # Print output details in the requested format
            print("\n===================================================\n")
            print("Model:")
            print(model_display_name)
            print()
            print("Model Load Time:")
            print(f"{loader.model_load_time:.2f} sec")
            print()
            print("Prompt:")
            print(result['prompt'])
            print()
            print("Response:")
            print(result['response'].strip())
            print()
            print("Prompt Tokens:")
            print(result['prompt_tokens'])
            print()
            print("Response Tokens:")
            print(result['response_tokens'])
            print()
            print("Generation Time:")
            print(f"{result['generation_time']:.2f} sec")
            print()
            print("Tokens/sec:")
            print(f"{result['tokens_per_second']:.1f}")
            print("\n===================================================\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error("An error occurred during prompt processing: %s", e)

if __name__ == "__main__":
    main()
