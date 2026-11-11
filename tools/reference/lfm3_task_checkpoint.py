"""Read one bank of authenticated BF16 LFM3 source tensors at a time."""
import hashlib
import json
import os
import struct
from pathlib import Path
import stat

import torch

from lfm3_attention_bf16_fixture import source_functions, TORCH_REVISION

SOURCE_CONTRACT = "d9a5b636fcb3e7eb472f3cd1e58848bbaa4703a6acfca4dc4a398eb963751af2"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(info):
    assert stat.S_ISREG(info.st_mode)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class Checkpoint:
    def __init__(self, source, model, role, stream):
        self.stream = stream
        self.model = model
        contract = (source / "prospective-contract-1.json").read_bytes()
        assert sha(contract) == SOURCE_CONTRACT
        metadata = json.loads(contract)
        if role == "stock":
            identity, header, config, self.namespace = (
                "stock_metadata_reference", "stock_mlx-header.bin", "stock_mlx-config.json", "language_model.",
            )
        else:
            assert role == "modified_parent"
            identity, header, config, self.namespace = "candidate", "child-header.bin", "parent-config.json", ""
        self.checkpoint = metadata[identity]
        self.canonical = metadata["parent"] if role == "modified_parent" else self.checkpoint
        self.identity = file_identity(os.fstat(stream.fileno()))
        assert self.identity[2] == self.checkpoint["bytes"]
        size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < size <= 1024**2
        raw = stream.read(size)
        assert sha(raw) == metadata["metadata_sha256"][header]
        self.schema = json.loads(raw)
        self.origin = 8 + size
        config_path = source / "metadata-1" / config
        config_bytes = config_path.read_bytes()
        assert sha(config_bytes) == metadata["metadata_sha256"][config]
        self.layer_kinds = json.loads(config_bytes)["layer_types"]
        assert len(self.layer_kinds) == 24
        assert torch.version.git_version == TORCH_REVISION
        _, self.eager = source_functions(source / "metadata-1/parent-modeling_lfm3.py")

    def check(self, full=False):
        assert file_identity(os.fstat(self.stream.fileno())) == file_identity(self.model.lstat()) == self.identity
        if full:
            self.stream.seek(0)
            assert hashlib.file_digest(self.stream, "sha256").hexdigest() == self.checkpoint["sha256"]
            assert file_identity(os.fstat(self.stream.fileno())) == self.identity

    def tensor(self, name, start=0, rows=None):
        meta = self.schema[self.namespace + name]
        assert meta["dtype"] == "BF16"
        low, high = meta["data_offsets"]
        shape = meta["shape"]
        if rows is not None:
            assert len(shape) == 2 and 0 <= start < start + rows <= shape[0]
            low += start * shape[1] * 2
            high = low + rows * shape[1] * 2
            shape = [rows, shape[1]]
        raw = bytearray(high - low)
        self.stream.seek(self.origin + low)
        assert self.stream.readinto(raw) == len(raw)
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape)

    def layer(self, index):
        prefix = f"model.layers.{index}."
        return {key.removeprefix(self.namespace + prefix): self.tensor(key.removeprefix(self.namespace))
                for key in self.schema if key.startswith(self.namespace + prefix)}
