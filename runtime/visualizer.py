"""
LLMScope Runtime - Rich UI Visualizer & Educational Panel Renderers

Provides reusable Rich UI renderers for Inference Timeline, Context Evolution,
Layer-wise KV Cache breakdown, and Inference Insights.
"""

from typing import Dict, Any, List, Optional
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group


def render_inference_timeline(result: Dict[str, Any]) -> Panel:
    """
    Renders the execution timeline showing the sequence of runtime stages and latency metrics.
    
    Args:
        result (Dict[str, Any]): Generation metrics dictionary from PrefillAnalyzer.
    """
    tok_ms = result.get("tokenization_time", 0.0) * 1000
    prefill_ms = result.get("estimated_prefill", 0.0) * 1000
    ttft_ms = result.get("ttft", 0.0) * 1000
    decode_s = result.get("decode_time", 0.0)
    tps = result.get("tokens_per_second", 0.0)
    reason = result.get("finish_reason", "Completed")

    timeline_str = (
        "[bold white]Prompt Submitted[/bold white]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold cyan]Tokenizer[/bold cyan] [dim]({tok_ms:.2f} ms)[/dim]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold blue]Prefill Phase[/bold blue] [dim]({prefill_ms:.2f} ms)[/dim]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold yellow]First Token Generated[/bold yellow] [dim](TTFT: {ttft_ms:.2f} ms)[/dim]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold green]Decode Phase[/bold green] [dim]({decode_s:.2f} s @ {tps:.2f} tok/s)[/dim]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold white]Generation Complete[/bold white] [dim]({reason})[/dim]"
    )

    return Panel(
        timeline_str,
        title="[bold magenta]Inference Execution Timeline[/bold magenta]",
        border_style="magenta"
    )


def render_context_evolution(prompt_tokens: int, response_tokens: int, max_tokens: int) -> Panel:
    """
    Renders an educational visualization of context length expanding step-by-step during decode.
    """
    total_context = prompt_tokens + response_tokens
    
    if response_tokens > 3:
        step_sample = f"{prompt_tokens + 1} → {prompt_tokens + 2} → {prompt_tokens + 3} ... → {total_context}"
    elif response_tokens > 0:
        step_sample = " → ".join(str(prompt_tokens + i) for i in range(1, response_tokens + 1))
    else:
        step_sample = str(prompt_tokens)

    content = (
        f"[bold cyan]Prompt Phase:[/bold cyan] [bold yellow]{prompt_tokens}[/bold yellow] Initial Tokens\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold blue]Sequential Decode Evolution:[/bold blue]\n"
        f"      [dim white]{step_sample} Tokens[/dim white]\n"
        "      [dim cyan]│[/dim cyan]\n"
        f"      [dim cyan]▼[/dim cyan] [bold green]Final Context Size:[/bold green] [bold white]{total_context}[/bold white] / [dim]{max_tokens} Max Budget[/dim]\n\n"
        "[dim cyan]Educational Note: Every autoregressively generated token extends the attention KV cache context window.[/dim cyan]"
    )

    return Panel(
        content,
        title="[bold blue]Context Evolution (Attention Sequence Expansion)[/bold blue]",
        border_style="blue"
    )


def render_layerwise_kv_cache(
    num_layers: int,
    context_length: int,
    per_layer_mb: float,
    total_mb: float
) -> Panel:
    """
    Renders a layer-wise conceptual KV Cache memory allocation breakdown across transformer layers.
    """
    lines = []
    # Show up to 16 layers (or first 8 + last 4 if layer count is huge)
    display_layers = range(1, num_layers + 1)
    
    for layer in display_layers:
        bar = "████████████████████████"
        lines.append(
            f"[bold cyan]Layer {layer:02d}[/bold cyan]  [bold green]{bar}[/bold green]  "
            f"[bold white]{context_length}[/bold white] Tokens [dim]({per_layer_mb:.2f} MB)[/dim]"
        )

    header = (
        f"[bold white]Transformer Layer Memory Distribution ({num_layers} Layers):[/bold white]\n"
        f"[bold yellow]Total Estimated KV Cache Memory:[/bold yellow] [bold green]{total_mb:.2f} MB[/bold green]\n\n"
    )
    
    footer = (
        "\n\n[dim cyan]Educational Note: This conceptual breakdown illustrates the 2 × L × N × H × dtype formula.\n"
        "Frameworks like MLX allocate metal GPU buffers dynamically; this shows per-layer mathematical state memory.[/dim cyan]"
    )

    full_text = header + "\n".join(lines) + footer

    return Panel(
        full_text,
        title="[bold yellow]Layer-wise KV Cache Memory Allocation[/bold yellow]",
        border_style="yellow"
    )


def generate_inference_insights(result: Dict[str, Any], cache_metrics: Any) -> str:
    """
    Generates dynamic, metric-driven inference insights for display in a Rich Panel.
    """
    insights = []

    pt = result.get("prompt_tokens", 0)
    rt = result.get("response_tokens", 0)
    max_t = result.get("max_tokens", 512)
    tok_ms = result.get("tokenization_time", 0.0) * 1000
    ttft_ms = result.get("ttft", 0.0) * 1000
    prefill_ms = result.get("estimated_prefill", 0.0) * 1000
    reason = result.get("finish_reason", "Stop Token")
    context_len = getattr(cache_metrics, "context_length", pt + rt)
    cache_mb = getattr(cache_metrics, "estimated_cache_mb", 0.0)
    growth_kb = getattr(cache_metrics, "growth_per_token_kb", 0.0)

    # 1. Prompt Size Insight
    if pt < 50:
        insights.append(f"[bold green]✓[/bold green] Short prompt ({pt} tokens) resulted in low prefill latency.")
    elif pt < 200:
        insights.append(f"[bold green]✓[/bold green] Medium prompt ({pt} tokens) required standard matrix prefill work.")
    else:
        insights.append(f"[bold green]✓[/bold green] Large prompt ({pt} tokens) increased prefill compute across all transformer layers.")

    # 2. Tokenization Overhead Insight
    if tok_ms < 5.0:
        insights.append("[bold green]✓[/bold green] Tokenization overhead was negligible.")
    else:
        insights.append(f"[bold green]✓[/bold green] Tokenization phase required {tok_ms:.2f} ms execution time.")

    # 3. TTFT & Prefill Insight
    if ttft_ms > 0:
        if prefill_ms / ttft_ms > 0.8 if ttft_ms else False:
            insights.append("[bold green]✓[/bold green] TTFT increased because every prompt token must pass through all transformer layers before decoding begins.")
        else:
            insights.append(f"[bold green]✓[/bold green] First token produced in {ttft_ms:.2f} ms.")

    # 4. Finish Reason Insight
    if reason == "Maximum Token Limit Reached":
        insights.append(f"[bold green]✓[/bold green] Generation stopped because the configured maximum generation length ({max_t}) was reached.")
    elif reason == "Stop Token":
        insights.append("[bold green]✓[/bold green] Model emitted a stop token before reaching the generation limit.")
    elif reason == "End of Sequence":
        insights.append("[bold green]✓[/bold green] Model reached End of Sequence (EOS) naturally.")
    else:
        insights.append(f"[bold green]✓[/bold green] Generation completed with status: {reason}.")

    # 5. KV Cache Memory Scaling Insight
    insights.append(
        f"[bold green]✓[/bold green] Estimated KV Cache memory increased linearly with context length "
        f"([cyan]{context_len} tokens[/cyan] → [yellow]{cache_mb:.2f} MB[/yellow] @ [dim]{growth_kb:.2f} KB/token[/dim])."
    )

    return "\n".join(insights)
