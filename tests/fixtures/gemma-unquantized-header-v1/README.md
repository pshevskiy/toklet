# Official Gemma E2B header and text layout v1

This fixture contains only metadata: the original acquired safetensors header,
official config and inspected text layout for `google/gemma-4-E2B-it` revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`. It contains no weight payload.
See the [source, actual checks and limits](../../../docs/gemma-unquantized-layout.md).
The raw header includes one intentional final padding byte. Its exact length
participates in every absolute tensor offset. The narrowly scoped Git binary
attribute preserves its bytes and avoids treating container padding as a source
whitespace error; do not trim or reformat this artifact.

| File | SHA-256 |
| --- | --- |
| `header.json` | `12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e` |
| `config.json` | `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330` |
| `layout.json` | `2419f70e7232fd75a6d361e0ef8e8acaf37f87f52124e63e1e04e814d1040840` |

Run offline stdlib tests from the repository root:

```sh
python3.12 -I -S -B -c 'import runpy,sys;sys.path.insert(0,"tools/reference");runpy.run_path("tools/reference/test_gemma4_unquantized_layout.py",run_name="__main__")'
```

The producer CLI accepts `--header`, `--config`, and a new `--output` file.
The independent checker accepts those first two paths, `--layout` and the
externally frozen `--layout-sha`. Both run with only Python3.12's standard
library. The checker also authenticates the earlier versioned resident-storage
inventory. It does not import the producer or use downloaded model data.

Header acquisition and metadata provenance are retained in
`artifacts/m8-gemma-unquantized-header-v1`, with exact identities in the public
evidence linked by the report. The source repository declares Apache-2.0.
The reviewed Transformers5.7.0 source rule is separately identified by its
full modeling-file hash and original line ranges. Inspecting descriptors and
unused consumer parameters does not verify the complete advertised weight
file, qualify numerical arithmetic, establish model fit or support media.
