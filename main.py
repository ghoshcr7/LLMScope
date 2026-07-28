import csv
import json
import logging
import os
import sys
from datetime import datetime

# pyrefly: ignore [missing-import]
from runtime.model_loader import ModelLoader
# pyrefly: ignore [missing-import]
from runtime.tokenizer import TokenizerExplorer
from runtime.prefill import PrefillAnalyzer
from runtime.kv_cache import ModelConfig, KVCacheEstimator, KVCacheMetrics
from runtime.benchmark import analyze_benchmarks, render_benchmark_summary
from runtime.visualizer import (
    render_inference_timeline,
    render_context_evolution,
    render_layerwise_kv_cache,
    generate_inference_insights,
)

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


def save_metrics_to_csv(
    result: dict,
    cache_metrics: KVCacheMetrics,
    model_name: str,
    csv_path: str = "benchmarks/metrics.csv"
) -> None:
    """
    Appends benchmark metrics to a CSV file for long-term historical tracking and dashboarding.
    """
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path)

        fieldnames = [
            "Timestamp",
            "Model",
            "Prompt Tokens",
            "Response Tokens",
            "TTFT",
            "Estimated Prefill",
            "Decode Time",
            "Generation Time",
            "Throughput",
            "Estimated KV Cache Memory",
            "Finish Reason"
        ]

        row = {
            "Timestamp": datetime.now().isoformat(),
            "Model": model_name,
            "Prompt Tokens": result["prompt_tokens"],
            "Response Tokens": result["response_tokens"],
            "TTFT": f"{result['ttft']:.4f}",
            "Estimated Prefill": f"{result['estimated_prefill']:.4f}",
            "Decode Time": f"{result['decode_time']:.4f}",
            "Generation Time": f"{result['generation_time']:.4f}",
            "Throughput": f"{result['tokens_per_second']:.2f}",
            "Estimated KV Cache Memory": f"{cache_metrics.estimated_cache_mb:.2f} MB",
            "Finish Reason": result.get("finish_reason", "Completed")
        }

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.error("Failed to save benchmark metrics to CSV: %s", e)


def save_run_snapshot(
    prompt: str,
    analysis: dict,
    decoded_tokens: str,
    result: dict,
    cache_metrics: KVCacheMetrics,
    model_name: str,
    runs_dir: str = "runs"
) -> str:
    """
    Saves a complete JSON run snapshot to disk for offline profiling replay.
    
    Returns:
        str: Absolute or relative file path to the saved JSON snapshot.
    """
    os.makedirs(runs_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filepath = os.path.join(runs_dir, f"run_{timestamp_str}.json")

    snapshot_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": prompt,
        "formatted_prompt": analysis.get("formatted_prompt", prompt),
        "token_ids": analysis.get("token_ids", []),
        "decoded_tokens": decoded_tokens,
        "prompt_tokens": result["prompt_tokens"],
        "response": result["response"],
        "response_tokens": result["response_tokens"],
        "max_tokens": result.get("max_tokens", 512),
        "tokenization_time": result["tokenization_time"],
        "ttft": result["ttft"],
        "estimated_prefill": result["estimated_prefill"],
        "decode_time": result["decode_time"],
        "generation_time": result["generation_time"],
        "throughput": result["tokens_per_second"],
        "estimated_kv_cache": f"{cache_metrics.estimated_cache_mb:.2f} MB",
        "estimated_kv_cache_mb": cache_metrics.estimated_cache_mb,
        "finish_reason": result.get("finish_reason", "Completed"),
        "context_length": cache_metrics.context_length,
        "num_layers": cache_metrics.num_layers,
        "hidden_size": cache_metrics.hidden_size,
        "dtype_bytes": cache_metrics.dtype_bytes,
        "growth_per_token_kb": cache_metrics.growth_per_token_kb,
        "per_layer_mb": cache_metrics.per_layer_mb,
    }

    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)

    return filepath


def main() -> None:
    """
    Main entry point for the interactive LLMScope CLI session.
    
    Loads the default MLX model and enters a read-eval-print loop (REPL) for text generation.
    """
    model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    console.print(Panel(
        f"[bold cyan]Loading model:[/bold cyan] [yellow]{model_name}[/yellow]\n[dim]Note: Initial run downloads model weights from Hugging Face.[/dim]",
        title="[bold magenta]LLMScope Initializer[/bold magenta]",
        border_style="cyan"
    ))
    
    try:
        loader = ModelLoader(model_name)
    except Exception as e:
        logger.error("Failed to initialize ModelLoader with model %s: %s", model_name, e)
        sys.exit(1)
        
    console.print("\n[bold green]✔ Model loaded successfully![/bold green] Type [bold yellow]'exit'[/bold yellow] to quit.\n")
    
    # Initialize ModelConfig and KVCacheEstimator
    model_config = ModelConfig.from_model(loader.model, model_name=loader.model_name)
    kv_estimator = KVCacheEstimator(model_config)

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
            
            max_tokens = 512
            profiler = PrefillAnalyzer(loader)

            result = profiler.analyze(
                prompt=analysis["formatted_prompt"],
                max_tokens=max_tokens,
                temp=0.7
            )
            
            # Estimate KV Cache Metrics
            cache_metrics = kv_estimator.estimate(
                prompt_tokens=result["prompt_tokens"],
                response_tokens=result["response_tokens"]
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
            
            # Create a stylized metrics summary table: LLMScope Runtime Profiler
            metrics_table = Table(
                title="LLMScope Runtime Profiler",
                border_style="magenta",
                show_header=True,
                header_style="bold cyan"
            )
            metrics_table.add_column("Category", style="bold white", width=18)
            metrics_table.add_column("Metric", style="cyan", width=26)
            metrics_table.add_column("Value", style="bold yellow", justify="right")

            # 1. Model Metadata Category
            metrics_table.add_row("Model Metadata", "Model Name", model_display_name)
            metrics_table.add_row("Model Metadata", "Backend", "MLX (Apple Silicon)")
            metrics_table.add_row("Model Metadata", "Model Load Time", f"{loader.model_load_time:.2f} s")
            metrics_table.add_section()

            # 2. Prefill Phase Category
            metrics_table.add_row("Prefill Phase", "Prompt Tokens", f"{result['prompt_tokens']}")
            metrics_table.add_row("Prefill Phase", "Tokenization Time", f"{result['tokenization_time']*1000:.2f} ms")
            metrics_table.add_row("Prefill Phase", "TTFT (First Token)", f"{result['ttft']*1000:.2f} ms")
            metrics_table.add_row("Prefill Phase", "Estimated Prefill", f"{result['estimated_prefill']*1000:.2f} ms")
            metrics_table.add_section()

            # 3. Decode Phase Category
            metrics_table.add_row(
                "Decode Phase",
                "Response Tokens",
                f"[bold green]{result['response_tokens']} / {result['max_tokens']}[/bold green]"
            )
            metrics_table.add_row(
                "Decode Phase",
                "Finish Reason",
                f"[bold yellow]{result['finish_reason']}[/bold yellow]"
            )
            metrics_table.add_row("Decode Phase", "Decode Time", f"{result['decode_time']:.2f} s")
            metrics_table.add_row("Decode Phase", "Total Generation Time", f"{result['generation_time']:.2f} s")
            metrics_table.add_row(
                "Decode Phase",
                "Throughput (TPS)",
                f"[bold green]{result['tokens_per_second']:.2f} tok/s[/bold green]"
            )
            metrics_table.add_section()

            # 4. KV Cache Category
            metrics_table.add_row("KV Cache", "Current Context Length", f"{cache_metrics.context_length} tokens")
            metrics_table.add_row("KV Cache", "Prompt Tokens", f"{cache_metrics.prompt_tokens}")
            metrics_table.add_row("KV Cache", "Generated Tokens", f"{cache_metrics.response_tokens}")
            metrics_table.add_row("KV Cache", "Transformer Layers", f"{cache_metrics.num_layers}")
            metrics_table.add_row("KV Cache", "Hidden Size", f"{cache_metrics.hidden_size}")
            metrics_table.add_row("KV Cache", "Estimated KV Cache Memory", f"[bold green]{cache_metrics.estimated_cache_mb:.2f} MB[/bold green]")
            metrics_table.add_row("KV Cache", "Cache Growth", f"{cache_metrics.growth_per_token_kb:.2f} KB / token")

            console.print("\n")
            console.print(metrics_table)
            console.print("\n")

            # Feature 1: Inference Execution Timeline Panel
            console.print(render_inference_timeline(result))
            console.print("\n")

            # Feature 2: Context Evolution Panel
            console.print(render_context_evolution(
                result["prompt_tokens"],
                result["response_tokens"],
                max_tokens
            ))
            console.print("\n")

            # Feature 3: Layer-wise KV Cache Memory Allocation Panel
            console.print(render_layerwise_kv_cache(
                cache_metrics.num_layers,
                cache_metrics.context_length,
                cache_metrics.per_layer_mb,
                cache_metrics.estimated_cache_mb
            ))
            console.print("\n")

            # Feature 4: Dynamic Inference Insights Panel
            insights_text = generate_inference_insights(result, cache_metrics)
            console.print(Panel(
                insights_text,
                title="[bold cyan]Inference Insights[/bold cyan]",
                border_style="cyan"
            ))
            console.print("\n")

            # Feature 5 & 6: Save CSV & Render Benchmark Analytics Panel
            save_metrics_to_csv(result, cache_metrics, loader.model_name)
            analytics = analyze_benchmarks()
            console.print(render_benchmark_summary(analytics))
            console.print("\n")

            # Feature 7: Save JSON Run Snapshot for Offline Replay
            snapshot_file = save_run_snapshot(
                prompt=prompt,
                analysis=analysis,
                decoded_tokens=decoded_tokens,
                result=result,
                cache_metrics=cache_metrics,
                model_name=loader.model_name
            )
            console.print(f"[dim]Run profile saved for replay: [cyan]{snapshot_file}[/cyan][/dim]\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            logger.error("An error occurred during prompt processing: %s", e)


if __name__ == "__main__":
    main()
