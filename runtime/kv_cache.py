"""
LLMScope Runtime - KV Cache Estimator & Educational Visualizer

================================================================================
EDUCATIONAL FOUNDATIONS & ARCHITECTURAL DOCUMENTATION
================================================================================
1. What is the Key-Value (KV) Cache and Why does it exist?
   During autoregressive Transformer inference, each newly generated token must
   attend to all previous tokens in the context (prompt tokens + past generated
   tokens). Without caching, calculating Attention(Q, K, V) at decoding step 't'
   would require re-computing the Key (K) and Value (V) matrix projections for all
   t-1 previous tokens at every single step, leading to O(N^2) redundant compute.

   The Key-Value (KV) Cache stores the pre-computed K and V tensor projections for all
   prior tokens across all layers. At each step t, the model only computes Q, K, and V
   for token t, appends its K and V to the cache, and performs attention over the
   accumulated K and V matrices in O(N) memory access time.

2. Why Prefill creates the KV Cache:
   During the initial Prefill phase, all prompt tokens are processed concurrently in
   parallel. The transformer computes the K and V representations for every prompt
   token across all attention layers and initializes the KV Cache with these tensors.

3. Why Decode reuses the KV Cache:
   During the iterative Decode phase, generation proceeds sequentially (one token per
   step). Instead of reprocessing the entire sequence, the model retrieves the cached
   K and V tensors from previous steps, appends the newly computed K and V vectors for
   the current token, and performs attention lookup without repeating past matrix math.

4. Why longer prompts increase memory:
   Because the KV cache stores K and V projections for every token in the sequence across
   all transformer layers, memory footprint grows linearly with context length:
   Context Length = Prompt Tokens + Generated Tokens.

5. Why Decode becomes efficient after prefill:
   Prefill is compute-bound (processing prompt tokens in parallel via matrix multiplications).
   Decode is memory-bandwidth bound (fetching stored KV Cache tensors for each token).
   By reusing the KV Cache, decode avoids O(N^2) redundant floating-point operations.

6. Why the Multiplier '2' exists in the memory formula:
   For every token and every layer, the transformer maintains TWO distinct hidden state
   vectors:
   - Key (K) vector: used to score attention weights (Query · Key^T).
   - Value (V) vector: used to compute the context representation (Attention_Weights · Value).
   
   Memory Formula:
       Estimated KV Cache Memory = 2 × Layers × Context_Length × Hidden_Size × dtype_bytes
       
   Where:
   - 2: Accounting for both Key and Value tensors.
   - Layers: Total transformer layers (num_hidden_layers).
   - Context Length: Total tokens active in context (Prompt Tokens + Generated Tokens).
   - Hidden Size: Total hidden dimension (num_heads × head_dim).
   - dtype_bytes: Memory footprint per element (2 bytes for FP16/BF16, 4 bytes for FP32).

7. Why this implementation is an Estimation:
   MLX and high-performance inference engines handle KV cache allocations dynamically
   (e.g., contiguous buffer allocation, PagedAttention, or quantized KV cache formats).
   MLX does not expose internal private memory allocations. Thus, this calculation
   serves as an mathematically precise theoretical estimation of the KV cache memory
    footprint required by the transformer architecture. Always refer to it as
   "Estimated KV Cache".
================================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """
    Configuration dataclass containing transformer architectural parameters.
    
    Attributes:
        model_name (str): Model name or identifier.
        hidden_size (int): Hidden dimension size (d_model).
        num_layers (int): Total number of transformer layers.
        num_heads (int): Total number of attention heads.
        head_dim (int): Dimension of each attention head.
        dtype_bytes (float): Precision size in bytes per element (default: 2.0 for FP16/BF16).
    """
    model_name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: float = 2.0

    @classmethod
    def from_model(cls, model: Any, model_name: str = "Transformer") -> "ModelConfig":
        """
        Dynamically extracts architectural metadata from an MLX model or fallback defaults.
        """
        args = getattr(model, "args", None)
        if args is not None:
            num_layers = getattr(args, "num_hidden_layers", getattr(args, "n_layers", 16))
            hidden_size = getattr(args, "hidden_size", 2048)
            num_heads = getattr(args, "num_attention_heads", getattr(args, "n_heads", 32))
            head_dim = getattr(args, "head_dim", hidden_size // num_heads if num_heads else 64)
            return cls(
                model_name=model_name,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype_bytes=2.0
            )
        
        # Default fallback config (e.g. Llama-3.2-1B)
        return cls(
            model_name=model_name,
            hidden_size=2048,
            num_layers=16,
            num_heads=32,
            head_dim=64,
            dtype_bytes=2.0
        )


@dataclass
class KVCacheMetrics:
    """
    Structured container for calculated KV Cache metrics.
    """
    context_length: int
    prompt_tokens: int
    response_tokens: int
    num_layers: int
    hidden_size: int
    dtype_bytes: float
    estimated_cache_bytes: float
    estimated_cache_mb: float
    growth_per_token_bytes: float
    growth_per_token_kb: float
    per_layer_mb: float


class KVCacheEstimator:
    """
    Computes mathematically documented KV Cache estimations for transformer inference.
    
    No printing, no Rich UI, no model inference. Returns structured metrics only.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def estimate(self, prompt_tokens: int, response_tokens: int) -> KVCacheMetrics:
        """
        Calculates context length, total estimated KV cache memory, per-token growth rate, and per-layer allocation.
        
        Args:
            prompt_tokens (int): Number of tokens in the prompt (prefill phase).
            response_tokens (int): Number of tokens generated (decode phase).
            
        Returns:
            KVCacheMetrics: Dataclass containing structured memory and context metrics.
        """
        context_length = prompt_tokens + response_tokens
        
        # Formula: Growth per token = 2 × Layers × Hidden Size × dtype_bytes
        growth_per_token_bytes = (
            2 * self.config.num_layers * self.config.hidden_size * self.config.dtype_bytes
        )
        
        # Formula: Total Estimated KV Cache = Context Length × Growth per token
        estimated_cache_bytes = context_length * growth_per_token_bytes
        estimated_cache_mb = estimated_cache_bytes / (1024 * 1024)
        growth_per_token_kb = growth_per_token_bytes / 1024

        # Per-layer memory allocation: 2 × Context Length × Hidden Size × dtype_bytes
        per_layer_bytes = 2 * context_length * self.config.hidden_size * self.config.dtype_bytes
        per_layer_mb = per_layer_bytes / (1024 * 1024)

        return KVCacheMetrics(
            context_length=context_length,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            dtype_bytes=self.config.dtype_bytes,
            estimated_cache_bytes=estimated_cache_bytes,
            estimated_cache_mb=estimated_cache_mb,
            growth_per_token_bytes=growth_per_token_bytes,
            growth_per_token_kb=growth_per_token_kb,
            per_layer_mb=per_layer_mb
        )

    def get_layer_breakdown(self, metrics: KVCacheMetrics):
        """
        Generates layer-by-layer conceptual memory breakdowns for educational visualization.
        """
        return [
            {
                "layer_idx": i + 1,
                "context_tokens": metrics.context_length,
                "memory_mb": metrics.per_layer_mb
            }
            for i in range(metrics.num_layers)
        ]

    def explain_behaviour(self, metrics: KVCacheMetrics) -> Dict[str, str]:
        """
        Generates text explanations of the calculated KV cache metrics.
        
        Returns:
            Dict[str, str]: Map of key concepts to educational descriptions.
        """
        return {
            "formula": (
                f"2 × {metrics.num_layers} layers × {metrics.context_length} context tokens × "
                f"{metrics.hidden_size} hidden_dim × {metrics.dtype_bytes:.0f} bytes"
            ),
            "total_memory": f"{metrics.estimated_cache_mb:.2f} MB",
            "growth_rate": f"{metrics.growth_per_token_kb:.2f} KB / token",
            "per_layer_memory": f"{metrics.per_layer_mb:.2f} MB / layer",
            "multiplier_reason": "Factor of 2 accounts for maintaining separate Key (K) and Value (V) tensors for each token per layer.",
            "linear_growth_reason": "Each generated token appends 1 new Key vector and 1 new Value vector to the cache, resulting in linear memory growth."
        }
