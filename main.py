import logging
import sys
# pyrefly: ignore [missing-import]
from runtime.model_loader import ModelLoader
# pyrefly: ignore [missing-import]
from runtime.tokenizer import TokenizerExplorer
from runtime.prefill import PrefillAnalyzer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown

# Configure logging to keep console clean by default, but report issues
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LLMScope")
console = Console()

def main() -> None:
    """
    Main entry point for the interactive LLMScope CLI session.
    
    Loads the default Llama-3.2-1B-Instruct-4bit model and enters
    a read-eval-print loop (REPL) for text generation.
    """
    model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    console.print(Panel(f"[bold cyan]Loading model:[/bold cyan] [yellow]{model_name}[/yellow]\n[dim]Note: Initial run downloads model weights from Hugging Face.[/dim]", title="[bold magenta]LLMScope Initializer[/bold magenta]", border_style="cyan"))
    
    try:
        loader = ModelLoader(model_name)
    except Exception as e:
        logger.error("Failed to initialize ModelLoader with model %s: %s", model_name, e)
        sys.exit(1)
        
    console.print("\n[bold green]✔ Model loaded successfully![/bold green] Type [bold yellow]'exit'[/bold yellow] to quit.\n")
    
    while True:
        try:
            # Prompt the user for input
            prompt = console.input("[bold magenta]Prompt ❯ [/bold magenta]")
            
            # Skip empty entries
            if not prompt.strip():
                continue
                
            # Exit check
            if prompt.strip().lower() == "exit":
                console.print("[bold green]Goodbye![/bold green]")
                break
                
            # Create a TokenizerExplorer for analysis before generation
            explorer = TokenizerExplorer(loader.tokenizer)
            analysis = explorer.tokenize(prompt)
            decoded_tokens = explorer.decode_tokens(analysis["token_ids"])
            
            # Print tokenization analysis in a rich Panel
            token_info = (
                f"[bold cyan]Token Count:[/bold cyan] [bold yellow]{analysis['token_count']}[/bold yellow]\n\n"
                f"[bold cyan]Token IDs:[/bold cyan]\n[green]{analysis['token_ids']}[/green]\n\n"
                f"[bold cyan]Decoded Tokens:[/bold cyan]\n[dim]{decoded_tokens}[/dim]"
            )
            console.print("\n")
            console.print(Panel(
                f"[bold white]{analysis['formatted_prompt']}[/bold white]",
                title="[bold blue]Formatted Prompt (Chat Template)[/bold blue]",
                border_style="blue"
            ))
            console.print(Panel(
                token_info,
                title="[bold yellow]Tokenizer Explorer Analysis[/bold yellow]",
                border_style="yellow"
            ))
            
            console.print("\n[bold magenta]⚡ [Generating...][/bold magenta]")
            
            # Generate the response
            profiler = PrefillAnalyzer(loader)

            result = profiler.analyze(
                prompt=analysis["formatted_prompt"],
                max_tokens=512,
                temp=0.7
            )
            
            # Render formatted response with Pygments syntax highlighting
            console.print("\n")
            console.print(Panel(
                Markdown(result["response"]),
                title="[bold green]Formatted Output (Syntax Highlighted)[/bold green]",
                border_style="green"
            ))
            
            # Extract clean model name for display
            model_display_name = loader.model_name.split("/")[-1]
            for suffix in ["-Instruct", "-4bit"]:
                if suffix in model_display_name:
                    model_display_name = model_display_name.split(suffix)[0]
            
            # Create a stylized metrics summary table
            metrics_table = Table(title="LLMScope Performance Metrics", border_style="magenta", show_header=True, header_style="bold cyan")
            metrics_table.add_column("Category", style="bold white", width=16)
            metrics_table.add_column("Metric", style="cyan", width=22)
            metrics_table.add_column("Value", style="bold yellow", justify="right")

            metrics_table.add_row("Model Metadata", "Model Name", model_display_name)
            metrics_table.add_row("Model Metadata", "Backend", "MLX (Apple Silicon)")
            metrics_table.add_row("Model Metadata", "Model Load Time", f"{loader.model_load_time:.2f} s")
            metrics_table.add_section()

            metrics_table.add_row("Prefill Phase", "Prompt Tokens", f"{result['prompt_tokens']}")
            metrics_table.add_row("Prefill Phase", "Tokenization Time", f"{result['tokenization_time']*1000:.2f} ms")
            metrics_table.add_row("Prefill Phase", "TTFT (First Token)", f"{result['ttft']*1000:.2f} ms")
            metrics_table.add_row("Prefill Phase", "Estimated Prefill", f"{result['estimated_prefill']*1000:.2f} ms")
            metrics_table.add_section()

            metrics_table.add_row("Decode Phase", "Response Tokens", f"[bold green]{result['response_tokens']}[/bold green]")
            metrics_table.add_row("Decode Phase", "Decode Time", f"{result['decode_time']:.2f} s")
            metrics_table.add_row("Decode Phase", "Total Generation Time", f"{result['generation_time']:.2f} s")
            metrics_table.add_row("Decode Phase", "Throughput (TPS)", f"[bold green]{result['tokens_per_second']:.2f} tok/s[/bold green]")

            console.print("\n")
            console.print(metrics_table)
            console.print("\n")
                        
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            logger.error("An error occurred during prompt processing: %s", e)

if __name__ == "__main__":
    main()
