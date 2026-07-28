"""
LLMScope - Inference Replay Engine

Replays recorded transformer profiling sessions from JSON run snapshots
WITHOUT loading model weights, WITHOUT running inference, and WITHOUT MLX.

Usage:
    python replay.py [runs/run_2026_07_28_020352.json]
"""

import json
import os
import sys
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from runtime.benchmark import analyze_benchmarks, render_benchmark_summary
from runtime.visualizer import (
    render_inference_timeline,
    render_context_evolution,
    render_layerwise_kv_cache,
    generate_inference_insights,
)

console = Console()


def find_latest_run_file(runs_dir: str = "runs") -> Optional[str]:
    """
    Finds the most recent JSON run snapshot file in the specified directory.
    """
    if not os.path.exists(runs_dir):
        return None
    
    json_files = [
        os.path.join(runs_dir, f)
        for f in os.listdir(runs_dir)
        if f.endswith(".json")
    ]
    if not json_files:
        return None
    
    # Sort files by modification time descending
    json_files.sort(key=os.path.getmtime, reverse=True)
    return json_files[0]


def replay_session(snapshot_path: str) -> None:
    """
    Loads a JSON profiling snapshot and reconstructs the complete LLMScope Rich profiler UI.
    """
    if not os.path.exists(snapshot_path):
        console.print(f"[bold red]Error:[/bold red] Snapshot file not found: [yellow]{snapshot_path}[/yellow]")
        sys.exit(1)

    try:
        with open(snapshot_path, mode="r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading snapshot file:[/bold red] {e}")
        sys.exit(1)

    console.print(Panel(
        f"[bold cyan]Replaying Profiling Run:[/bold cyan] [yellow]{snapshot_path}[/yellow]\n"
        f"[bold white]Timestamp:[/bold white] [dim]{snapshot.get('timestamp', 'N/A')}[/dim]\n"
        f"[bold white]Model:[/bold white] [dim]{snapshot.get('model', 'N/A')}[/dim]\n"
        f"[bold green]✔ Replay mode active — NO model loaded, NO inference executed.[/bold green]",
        title="[bold magenta]LLMScope Inference Replay Engine[/bold magenta]",
        border_style="cyan"
    ))

    # 1. Formatted Prompt & Tokenizer Explorer Analysis
    formatted_prompt = snapshot.get("formatted_prompt", snapshot.get("prompt", ""))
    token_count = snapshot.get("prompt_tokens", 0)
    token_ids = snapshot.get("token_ids", [])
    decoded_tokens = snapshot.get("decoded_tokens", "")

    token_info = (
        f"[bold cyan]Token Count:[/bold cyan] [bold yellow]{token_count}[/bold yellow]\n\n"
        f"[bold cyan]Token IDs:[/bold cyan]\n[green]{token_ids}[/green]\n\n"
        f"[bold cyan]Decoded Tokens:[/bold cyan]\n[dim]{decoded_tokens}[/dim]"
    )

    console.print("\n")
    console.print(Panel(
        f"[bold white]{formatted_prompt}[/bold white]",
        title="[bold blue]Formatted Prompt (Chat Template)[/bold blue]",
        border_style="blue"
    ))
    console.print(Panel(
        token_info,
        title="[bold yellow]Tokenizer Explorer Analysis[/bold yellow]",
        border_style="yellow"
    ))

    # 2. Formatted Output (Syntax Highlighted)
    console.print("\n")
    console.print(Panel(
        Markdown(snapshot.get("response", "")),
        title="[bold green]Formatted Output (Syntax Highlighted)[/bold green]",
        border_style="green"
    ))

    # Clean model display name
    model_raw = snapshot.get("model", "Llama-3.2-1B")
    model_display_name = model_raw.split("/")[-1]
    for suffix in ["-Instruct", "-4bit"]:
        if suffix in model_display_name:
            model_display_name = model_display_name.split(suffix)[0]

    # 3. LLMScope Runtime Profiler Table
    metrics_table = Table(
        title="LLMScope Runtime Profiler (Replayed)",
        border_style="magenta",
        show_header=True,
        header_style="bold cyan"
    )
    metrics_table.add_column("Category", style="bold white", width=18)
    metrics_table.add_column("Metric", style="cyan", width=26)
    metrics_table.add_column("Value", style="bold yellow", justify="right")

    # Category 1: Model Metadata
    metrics_table.add_row("Model Metadata", "Model Name", model_display_name)
    metrics_table.add_row("Model Metadata", "Backend", "MLX (Replay Mode)")
    metrics_table.add_row("Model Metadata", "Model Load Time", "0.00 s (Replayed)")
    metrics_table.add_section()

    # Category 2: Prefill Phase
    tok_time = snapshot.get("tokenization_time", 0.0)
    ttft = snapshot.get("ttft", 0.0)
    est_prefill = snapshot.get("estimated_prefill", 0.0)
    metrics_table.add_row("Prefill Phase", "Prompt Tokens", f"{snapshot.get('prompt_tokens', 0)}")
    metrics_table.add_row("Prefill Phase", "Tokenization Time", f"{tok_time*1000:.2f} ms")
    metrics_table.add_row("Prefill Phase", "TTFT (First Token)", f"{ttft*1000:.2f} ms")
    metrics_table.add_row("Prefill Phase", "Estimated Prefill", f"{est_prefill*1000:.2f} ms")
    metrics_table.add_section()

    # Category 3: Decode Phase
    resp_tokens = snapshot.get("response_tokens", 0)
    max_tokens = snapshot.get("max_tokens", 512)
    decode_time = snapshot.get("decode_time", 0.0)
    gen_time = snapshot.get("generation_time", 0.0)
    tps = snapshot.get("throughput", 0.0)
    reason = snapshot.get("finish_reason", "Stop Token")

    metrics_table.add_row("Decode Phase", "Response Tokens", f"[bold green]{resp_tokens} / {max_tokens}[/bold green]")
    metrics_table.add_row("Decode Phase", "Finish Reason", f"[bold yellow]{reason}[/bold yellow]")
    metrics_table.add_row("Decode Phase", "Decode Time", f"{decode_time:.2f} s")
    metrics_table.add_row("Decode Phase", "Total Generation Time", f"{gen_time:.2f} s")
    metrics_table.add_row("Decode Phase", "Throughput (TPS)", f"[bold green]{tps:.2f} tok/s[/bold green]")
    metrics_table.add_section()

    # Category 4: KV Cache
    context_length = snapshot.get("context_length", snapshot.get("prompt_tokens", 0) + resp_tokens)
    num_layers = snapshot.get("num_layers", 16)
    hidden_size = snapshot.get("hidden_size", 2048)
    cache_mb = snapshot.get("estimated_kv_cache_mb", 0.0)
    growth_kb = snapshot.get("growth_per_token_kb", (2 * num_layers * hidden_size * 2.0) / 1024)
    per_layer_mb = snapshot.get("per_layer_mb", cache_mb / num_layers if num_layers else 0.0)

    metrics_table.add_row("KV Cache", "Current Context Length", f"{context_length} tokens")
    metrics_table.add_row("KV Cache", "Prompt Tokens", f"{snapshot.get('prompt_tokens', 0)}")
    metrics_table.add_row("KV Cache", "Generated Tokens", f"{resp_tokens}")
    metrics_table.add_row("KV Cache", "Transformer Layers", f"{num_layers}")
    metrics_table.add_row("KV Cache", "Hidden Size", f"{hidden_size}")
    metrics_table.add_row("KV Cache", "Estimated KV Cache Memory", f"[bold green]{cache_mb:.2f} MB[/bold green]")
    metrics_table.add_row("KV Cache", "Cache Growth", f"{growth_kb:.2f} KB / token")

    console.print("\n")
    console.print(metrics_table)
    console.print("\n")

    # 4. Inference Execution Timeline Panel
    console.print(render_inference_timeline(snapshot))
    console.print("\n")

    # 5. Context Evolution Panel
    console.print(render_context_evolution(snapshot.get("prompt_tokens", 0), resp_tokens, max_tokens))
    console.print("\n")

    # 6. Layer-wise KV Cache Memory Allocation Panel
    console.print(render_layerwise_kv_cache(num_layers, context_length, per_layer_mb, cache_mb))
    console.print("\n")

    # 7. Dynamic Inference Insights Panel
    # Construct dummy metrics dataclass for insight generator
    class DummyCacheMetrics:
        pass
    dummy_metrics = DummyCacheMetrics()
    dummy_metrics.context_length = context_length
    dummy_metrics.estimated_cache_mb = cache_mb
    dummy_metrics.growth_per_token_kb = growth_kb

    insights_text = generate_inference_insights(snapshot, dummy_metrics)
    console.print(Panel(
        insights_text,
        title="[bold cyan]Inference Insights (Replayed)[/bold cyan]",
        border_style="cyan"
    ))
    console.print("\n")

    # 8. Benchmark Analytics & Performance Trends
    analytics = analyze_benchmarks()
    console.print(render_benchmark_summary(analytics))
    console.print("\n")


def main() -> None:
    """
    Main entry point for replay CLI tool.
    """
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = find_latest_run_file()
        if not target_file:
            console.print("[bold yellow]No recorded JSON snapshots found in 'runs/' directory.[/bold yellow]")
            console.print("Run [bold cyan]python main.py[/bold cyan] first to generate profiling runs.")
            sys.exit(0)
        console.print(f"[dim]No snapshot specified. Replaying latest run: {target_file}[/dim]\n")

    replay_session(target_file)


if __name__ == "__main__":
    main()
