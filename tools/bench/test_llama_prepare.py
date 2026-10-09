"""Offline tensor inventory checks for the explicit external conversion."""
import unittest
import llama_prepare


class Names(unittest.TestCase):
    def test_qwen_tensor_mapping(self):
        pairs = {
            "model.embed_tokens.weight": "token_embd.weight",
            "model.norm.weight": "output_norm.weight",
            "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
            "model.layers.23.post_attention_layernorm.weight": "blk.23.ffn_norm.weight",
            "model.layers.1.self_attn.q_proj.weight": "blk.1.attn_q.weight",
            "model.layers.1.self_attn.k_proj.bias": "blk.1.attn_k.bias",
            "model.layers.1.self_attn.v_proj.bias": "blk.1.attn_v.bias",
            "model.layers.1.self_attn.o_proj.weight": "blk.1.attn_output.weight",
            "model.layers.2.mlp.gate_proj.weight": "blk.2.ffn_gate.weight",
            "model.layers.2.mlp.up_proj.weight": "blk.2.ffn_up.weight",
            "model.layers.2.mlp.down_proj.weight": "blk.2.ffn_down.weight",
        }
        for source, target in pairs.items():
            self.assertEqual(llama_prepare.mapped_name(source), target)

    def test_unknown_names_fail(self):
        for name in ("lm_head.weight", "model.layers.0.unsupported.weight", "model.foo"):
            with self.assertRaises(ValueError):
                llama_prepare.mapped_name(name)


if __name__ == "__main__":
    unittest.main()
