#!/usr/bin/env python3
"""
tools/extract-qwen3-tensor-map.py

Parses the canonical GGUF model (Qwen3-4B-Q4_K_M.gguf) and generates
a structured tensor-type map in bench/results/m3/qwen3-q4k-tensor-map.json.
Zero external dependencies (uses Python standard library struct/json).
"""

import struct
import json
import sys
import os

def parse_gguf_tensors(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Canonical GGUF model not found at: {filepath}")

    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"File {filepath} is not a valid GGUF file (magic: {magic})")

        version, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_str():
            s_len, = struct.unpack("<Q", f.read(8))
            return f.read(s_len).decode("utf-8", errors="replace")

        def read_val(val_type):
            if val_type == 0: return struct.unpack("<B", f.read(1))[0]
            elif val_type == 1: return struct.unpack("<b", f.read(1))[0]
            elif val_type == 2: return struct.unpack("<H", f.read(2))[0]
            elif val_type == 3: return struct.unpack("<h", f.read(2))[0]
            elif val_type == 4: return struct.unpack("<I", f.read(4))[0]
            elif val_type == 5: return struct.unpack("<i", f.read(4))[0]
            elif val_type == 6: return struct.unpack("<f", f.read(4))[0]
            elif val_type == 7: return struct.unpack("<?", f.read(1))[0]
            elif val_type == 8: return read_str()
            elif val_type == 9:
                arr_type, arr_len = struct.unpack("<IQ", f.read(12))
                return [read_val(arr_type) for _ in range(arr_len)]
            elif val_type == 10: return struct.unpack("<Q", f.read(8))[0]
            elif val_type == 11: return struct.unpack("<q", f.read(8))[0]
            elif val_type == 12: return struct.unpack("<d", f.read(8))[0]
            else: raise ValueError(f"Unknown GGUF val_type: {val_type}")

        metadata = {}
        for _ in range(n_kv):
            k = read_str()
            v_type, = struct.unpack("<I", f.read(4))
            v = read_val(v_type)
            metadata[k] = v

        type_names = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
            14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
            19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
            24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M"
        }

        type_bpw = {
            "F32": 32.0, "F16": 16.0, "Q4_0": 4.5, "Q4_1": 5.0, "Q5_0": 5.5, "Q5_1": 6.0,
            "Q8_0": 8.5, "Q8_1": 9.0, "Q2_K": 2.5625, "Q3_K": 3.4375, "Q4_K": 4.5,
            "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_K": 8.5, "I8": 8.0, "I16": 16.0, "I32": 32.0
        }

        tensors = []
        type_counts = {}
        total_bytes = 0

        for _ in range(n_tensors):
            name = read_str()
            n_dims, = struct.unpack("<I", f.read(4))
            ne = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            t_type, = struct.unpack("<I", f.read(4))
            offset, = struct.unpack("<Q", f.read(8))

            type_str = type_names.get(t_type, f"TYPE_{t_type}")
            type_counts[type_str] = type_counts.get(type_str, 0) + 1

            n_elements = 1
            for d in ne:
                n_elements *= d

            bpw = type_bpw.get(type_str, 8.0)
            approx_bytes = int(n_elements * bpw / 8.0)
            total_bytes += approx_bytes

            category = "other"
            layer_idx = None
            if name.startswith("blk."):
                parts = name.split(".")
                try:
                    layer_idx = int(parts[1])
                except (ValueError, IndexError):
                    pass
                if "attn_q" in name: category = "attn_q"
                elif "attn_k" in name: category = "attn_k"
                elif "attn_v" in name: category = "attn_v"
                elif "attn_output" in name: category = "attn_output"
                elif "ffn_gate" in name: category = "ffn_gate"
                elif "ffn_up" in name: category = "ffn_up"
                elif "ffn_down" in name: category = "ffn_down"
                elif "norm" in name: category = "normalization"
            elif "output" in name: category = "output_head"
            elif "token_embd" in name: category = "embedding"

            tensors.append({
                "name": name,
                "type": type_str,
                "is_q4_k": (type_str == "Q4_K"),
                "dimensions": ne,
                "n_elements": n_elements,
                "approx_bytes": approx_bytes,
                "offset": offset,
                "category": category,
                "layer_idx": layer_idx
            })

        return {
            "model_path": filepath,
            "gguf_version": version,
            "n_tensors": n_tensors,
            "type_counts": type_counts,
            "q4_k_count": type_counts.get("Q4_K", 0),
            "q6_k_count": type_counts.get("Q6_K", 0),
            "f32_count": type_counts.get("F32", 0),
            "q4_k_projection_types": {
                "attn_q": "Q4_K (actual tensor in Qwen3-4B-Q4_K_M)",
                "attn_k": "Q4_K (actual tensor in Qwen3-4B-Q4_K_M)",
                "attn_v": "Q6_K (actual tensor; evaluated in M3 as synthetic Q4_K shape for cross-milestone comparability)",
                "attn_output": "Q4_K (actual tensor in Qwen3-4B-Q4_K_M)",
                "ffn_gate": "Q4_K (actual tensor in Qwen3-4B-Q4_K_M)",
                "ffn_up": "Q4_K (actual tensor in Qwen3-4B-Q4_K_M)",
                "ffn_down": "Q6_K (actual tensor; evaluated in M3 as synthetic Q4_K shape for cross-milestone comparability)"
            },
            "tensors": tensors
        }

def main():
    default_model = "/home/taly/abra/models/Qwen3-4B-Q4_K_M.gguf"
    model_path = sys.argv[1] if len(sys.argv) > 1 else default_model
    output_path = sys.argv[2] if len(sys.argv) > 2 else "bench/results/m3/qwen3-q4k-tensor-map.json"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Extracting tensor map from: {model_path}")
    tensor_map = parse_gguf_tensors(model_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tensor_map, f, indent=2)

    print(f"Wrote canonical tensor map to: {output_path}")
    print(f"  Total tensors: {tensor_map['n_tensors']}")
    print(f"  Q4_K count: {tensor_map['q4_k_count']}")
    print(f"  Q6_K count: {tensor_map['q6_k_count']}")
    print(f"  F32 count:  {tensor_map['f32_count']}")

if __name__ == "__main__":
    main()
