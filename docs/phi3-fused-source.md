# B2 Phi3 fused-source contract

Selected prospective representative:
[mlx-community/Phi-3-mini-4k-instruct-4bit](https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-4bit/tree/5b3819ed6317784fb20eddeae9bed984f778d0d0),
revision `5b3819ed6317784fb20eddeae9bed984f778d0d0`. Astra inspected public config,
tokenizer assets, source implementations, index and the exact safetensors header.
Only metadata and header ranges were fetched. The complete 2,149,696,167-byte
payload and advertised SHA256 `8d75680621a09474f6601e9176f2f61f92a5e4c079d68d583901f51699fda50a`
are **not locally downloaded or verified**. This is not Phi execution or app
qualification.

## Concrete shared-block mapping

The model is bias-free pre-attention/post-attention RMSNorm, ordinary dense
attention and SiLU MLP with residuals, untied output, 32 layers, hidden3072,
MLP8192, 32Q/32KV heads, head96 and vocabulary32064. RMS epsilon is1e-5;
full-head ordinary split-half RoPE uses theta10000 and no LongRoPE.

| Physical source matrix `[out,in]` | Ordered logical output rows |
| --- | --- |
| `self_attn.qkv_proj [9216,3072]` | Q`0..3072`, K`3072..6144`, V`6144..9216` |
| `mlp.gate_up_proj [16384,3072]` | gate`0..8192`, up`8192..16384` |
| `self_attn.o_proj [3072,3072]` | complete matrix |
| `mlp.down_proj [3072,8192]` | complete matrix |
| embedding/head, each`[32064,3072]` | two distinct owners |

The header contains455 tensors:130 affine4/group64 matrices with U32 codes and
F16 scales/biases, plus65 F16 RMS vectors. Actual tensor dtypes take precedence
over the config's BF16 description. Source groups run across input columns;
for3072 columns each code row has1536 bytes and each coefficient row96 bytes.
The data starts at absolute byte51879, deliberately unaligned for a U32 pointer.
Use bounded byte reads and original row spans; never introduce a typed mmap cast
or reject this valid file merely because its header is unaligned.

The [pinned MLX Phi3 implementation](https://github.com/ml-explore/mlx-lm/blob/f39c4aa2af5d61a5e85bd50cc3f4294fedaff116/mlx_lm/models/phi3.py)
and [vLLM adapter](https://github.com/vllm-project/vllm/blob/ab0e5ef7954190de2a74a6715d1382b08f5accc4/vllm/model_executor/models/phi3.py)
confirm these splits and reuse of ordinary dense blocks. Toklet first maps rows
into its existing logical projections; a fused execution kernel is not required.
No remote `auto_map` Python code will execute.

## Explicit semantic limits

The source declares4096 positions **and sliding_window2047**. Pinned MLX Phi3
ignores the latter; that implementation cannot establish full-window equivalence.
For the initial512-token vertical, full/causal attention is mathematically equal
to the declared window. Retain and report both source fields and check this
bounded equivalence in CPU/Metal admission. Wider requests must reject until
actual sliding semantics exist. No silent window truncation or unrestricted4096
claim. Non-null LongRoPE and partial rotary variants remain unimplemented and
must reject.

Tokenizer behavior differs from Llama3: metaspace plus ByteFallback/Fuse/Strip,
no automatic BOS. ConfigEOS32000 and tokenizerEOS32007 form the source wrapper's
stop union. The tokenizer ends at32010 while the head has53 additional unmapped
IDs. Production text must explicitly handle/reject such generated IDs with valid
state; numerical tests retain the entire declared vocabulary and do not mask
those logits. Converter lineage to a specific original unquantized revision is
not proven by its old model card; later quality work must preserve that limitation.

## Implementation tasks

1. **B2a:** reuse the existing borrowed `Matrix::row_range`; add one checked
   `Layout::row_span` plan and source range reads. Preserve all whole-tensor APIs.
   Ranged reads reject/sticky-poison audited sources, even for a full-length range;
   whole audited reads keep their existing proof path. Selected metadata is
   validated; a partial read grants no identity for unselected bytes.
2. **B2b:** strict Phi3 adapter, row mapping, head96 tiny independent graph and
   bounded-window admission. Reuse ordinary projection/norm/rotary/attention/MLP;
   no generic graph interpreter or new fused math kernel.
3. **B2c:** explicitly verify the complete pinned payload, independent short
   real-model/state/text protocol and measured Mac memory. Keep full adoption
   quality, application and phone work separate, with unchanged gates.

Prospective native capacity512 storage is about2.43/2.47GB for chunk32/128 before
small Rust metadata. An independent source ledger using existing reserves is
about4.884GB for4bit; the analogous8bit ledger is6.794GB and exceeds the6GiB
budget. These are planning estimates, not observed device or whole-process fit.

Private source-byte inventory:
`artifacts/new-goal-b2-phi-research-v1/manifest.json`, SHA256
`06bd35bf7cc497739d241cb88a2d6f4b4dac20bd111b914ae6772f8830df2cf5`.
See [active plan](shared-block-model-plan.md) and [TODO](../TODO.md).
