"""
Export attention data to optimized binary format (.attnbin) for the web visualizer.

Binary layout:
    [4 bytes]  header_length (uint32 LE)
    [H bytes]  JSON header (UTF-8)
    [B*N*4]    input_bounds (float32, row-major)
    [B*H*N*N*4] layer_0 attention (float32)
    [B*H*N*N*4] layer_1 attention (float32)
    ...

The web visualizer reads slices as zero-copy Float32Array views,
so only the active (batch, head) slice (~160KB for N=200) is ever
touched at render time.

Usage:
    from export_attention import export_attention_data
    export_attention_data(
        'my_model.attnbin',
        reaction_ids,           # list[str] length N
        reaction_names_dict,    # dict  id -> full name
        attention_tensors,      # list of L np arrays, each (B, H, N, N)
        input_bounds            # np array (B, N)
    )
"""

import json
import struct
import numpy as np


def export_attention_data(filepath, reaction_ids, reaction_names_dict,
                          attention_tensors, input_bounds):
    """
    Export attention data to a compact binary file for the web visualizer.

    Parameters
    ----------
    filepath : str
        Output path (recommended extension: .attnbin).
    reaction_ids : list of str, length N
        Short reaction identifiers (embedding labels).
    reaction_names_dict : dict
        Mapping from short ID to full human-readable name.
    attention_tensors : list of L numpy arrays
        Each array has shape (B, H, N, N) — float32 or float64.
    input_bounds : array-like, shape (B, N)
        Per-sample per-reaction bound values.
    """
    L = len(attention_tensors)
    B, H, N, _ = attention_tensors[0].shape
    assert len(reaction_ids) == N, \
        f"reaction_ids length {len(reaction_ids)} != context size {N}"

    input_bounds = np.asarray(input_bounds, dtype=np.float32)
    assert input_bounds.shape == (B, N), \
        f"input_bounds shape {input_bounds.shape} != ({B}, {N})"

    # Validate all layers
    for i, t in enumerate(attention_tensors):
        assert t.shape == (B, H, N, N), \
            f"Layer {i} shape {t.shape} != ({B}, {H}, {N}, {N})"

    full_names = [reaction_names_dict.get(rid, rid) for rid in reaction_ids]

    header = json.dumps({
        'format': 'attnbin_v1',
        'N': int(N),
        'B': int(B),
        'H': int(H),
        'L': int(L),
        'reaction_ids': list(reaction_ids),
        'full_names': full_names,
    }, separators=(',', ':')).encode('utf-8')

    # Compute sizes for verification
    bounds_bytes = B * N * 4
    layer_bytes = B * H * N * N * 4
    total_data = bounds_bytes + L * layer_bytes
    total_file = 4 + len(header) + total_data

    print(f"Export summary:")
    print(f"  Reactions:  {N}")
    print(f"  Batches:    {B}")
    print(f"  Heads:      {H}")
    print(f"  Layers:     {L}")
    print(f"  Header:     {len(header):,} bytes")
    print(f"  Bounds:     {bounds_bytes:,} bytes")
    print(f"  Per layer:  {layer_bytes:,} bytes")
    print(f"  Total file: {total_file:,} bytes ({total_file / 1e6:.1f} MB)")

    with open(filepath, 'wb') as f:
        # Header length (uint32 little-endian)
        f.write(struct.pack('<I', len(header)))
        # JSON header
        f.write(header)
        # Input bounds
        f.write(input_bounds.tobytes())
        # Attention layers
        for i, t in enumerate(attention_tensors):
            f.write(np.asarray(t, dtype=np.float32).tobytes())

    print(f"  Written to: {filepath}")


# ---- Convenience: export with fp16 for ~50% smaller files ----

def export_attention_data_fp16(filepath, reaction_ids, reaction_names_dict,
                               attention_tensors, input_bounds):
    """
    Same as export_attention_data but stores attention weights as float16.
    ~50% smaller files at the cost of ~0.001 precision loss.
    The web visualizer auto-detects fp16 from the header.
    """
    L = len(attention_tensors)
    B, H, N, _ = attention_tensors[0].shape
    assert len(reaction_ids) == N

    input_bounds = np.asarray(input_bounds, dtype=np.float32)
    assert input_bounds.shape == (B, N)

    full_names = [reaction_names_dict.get(rid, rid) for rid in reaction_ids]

    header = json.dumps({
        'format': 'attnbin_v1',
        'N': int(N),
        'B': int(B),
        'H': int(H),
        'L': int(L),
        'dtype': 'float16',
        'reaction_ids': list(reaction_ids),
        'full_names': full_names,
    }, separators=(',', ':')).encode('utf-8')

    bounds_bytes = B * N * 4  # bounds always float32
    layer_bytes = B * H * N * N * 2  # fp16
    total_file = 4 + len(header) + bounds_bytes + L * layer_bytes

    print(f"Export summary (fp16):")
    print(f"  Reactions:  {N}")
    print(f"  Batches:    {B}")
    print(f"  Heads:      {H}")
    print(f"  Layers:     {L}")
    print(f"  Total file: {total_file:,} bytes ({total_file / 1e6:.1f} MB)")

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', len(header)))
        f.write(header)
        f.write(input_bounds.tobytes())
        for t in attention_tensors:
            f.write(np.asarray(t, dtype=np.float16).tobytes())

    print(f"  Written to: {filepath}")


def export_attention_subset(filepath, reaction_ids, reaction_names_dict,
                           attention_tensors, input_bounds,
                           batch_indices=None, fp16=False):
    """
    Export a subset of batches. Useful for large datasets.

    Parameters
    ----------
    batch_indices : list of int or None
        Which batch indices to include. None = all batches.
    fp16 : bool
        If True, store attention weights as float16 (~50% smaller).

    Example — export every 10th sample:
        export_attention_subset('sampled.attnbin', ids, names, tensors, bounds,
                                batch_indices=list(range(0, 1000, 10)))
    """
    if batch_indices is not None:
        batch_indices = list(batch_indices)
        input_bounds = np.asarray(input_bounds)[batch_indices]
        attention_tensors = [t[batch_indices] for t in attention_tensors]

    if fp16:
        export_attention_data_fp16(filepath, reaction_ids, reaction_names_dict,
                                   attention_tensors, input_bounds)
    else:
        export_attention_data(filepath, reaction_ids, reaction_names_dict,
                              attention_tensors, input_bounds)


def export_attention_chunked(output_dir, reaction_ids, reaction_names_dict,
                             attention_tensors, input_bounds,
                             chunk_size=50, fp16=True):
    """
    Export large datasets as multiple chunk files for manageable web loading.

    Creates files:  chunk_0.attnbin, chunk_1.attnbin, ...
    Each contains `chunk_size` batches.

    Parameters
    ----------
    output_dir : str
        Directory to write chunk files.
    chunk_size : int
        Number of batches per chunk file.
    fp16 : bool
        Use float16 for ~50% smaller files.

    Example for 1000 samples:
        export_attention_chunked('chunks/', ids, names, tensors, bounds,
                                 chunk_size=50, fp16=True)
        # Creates 20 files of ~192 MB each (for N=200, H=8, L=6)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    B = attention_tensors[0].shape[0]
    chunk_idx = 0
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        indices = list(range(start, end))
        fname = os.path.join(output_dir, f'chunk_{chunk_idx}.attnbin')
        print(f"\n--- Chunk {chunk_idx}: batches [{start}, {end}) ---")
        export_attention_subset(fname, reaction_ids, reaction_names_dict,
                                attention_tensors, input_bounds,
                                batch_indices=indices, fp16=fp16)
        chunk_idx += 1

    print(f"\n=== Exported {chunk_idx} chunks to {output_dir} ===")
    print(f"Load individual chunks in the web visualizer as needed.")


# ---- Size estimation utility ----

def estimate_file_size(N, B, H, L, fp16=False):
    """Print estimated file sizes for planning."""
    bpf = 2 if fp16 else 4
    header_est = 500 + N * 20  # rough estimate
    bounds = B * N * 4
    layers = L * B * H * N * N * bpf
    total = 4 + header_est + bounds + layers
    dtype = 'fp16' if fp16 else 'fp32'
    print(f"Estimate ({dtype}): N={N}, B={B}, H={H}, L={L}")
    print(f"  Bounds:     {bounds/1e6:>8.1f} MB")
    print(f"  Layers:     {layers/1e6:>8.1f} MB")
    print(f"  Total:      {total/1e6:>8.1f} MB  ({total/1e9:.2f} GB)")
    print(f"  Per-batch:  {(bounds/B + layers/B)/1e6:.1f} MB")
    return total


if __name__ == '__main__':
    print("=== Size estimates for common scenarios ===\n")
    for B in [50, 100, 500, 1000]:
        estimate_file_size(200, B, 8, 6, fp16=True)
        print()

    print("=== Test export ===\n")
    N, B, H, L = 150, 20, 8, 3
    ids = [f'R{i}' for i in range(N)]
    names = {f'R{i}': f'Reaction {i}' for i in range(N)}
    tensors = [np.random.rand(B, H, N, N).astype(np.float32) for _ in range(L)]
    bounds = np.random.rand(B, N).astype(np.float32)

    export_attention_data('test_fp32.attnbin', ids, names, tensors, bounds)
    export_attention_data_fp16('test_fp16.attnbin', ids, names, tensors, bounds)
    export_attention_subset('test_subset.attnbin', ids, names, tensors, bounds,
                            batch_indices=[0, 5, 10, 15], fp16=True)
