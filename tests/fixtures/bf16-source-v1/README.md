# BF16 source and exact-bit fixture v1

The40-byte `source.bin` contains a fixed8-byte prefix, a4×3 finite BF16 matrix,
one BF16 scalar, and three nonfinite BF16 words. The48-byte `finite.f32le` is
constructed independently with Python's `struct` from explicit IEEE bit words.
It includes signed zero, subnormals, normal/finite extremes and simple fractions.
Neither array is exported from NumPy or the target row reader.

The [manifest](manifest.json) records all words, offsets and hashes. This is a
synthetic mechanism fixture; it contains no model checkpoint or task text.
See the [scope and actual evidence](../../../docs/bf16-source.md).

Run ordinary ownership tests from the repository root with only Python3.12:

```sh
python3.12 -I -S -B -c 'import runpy,sys;sys.path.insert(0,"tools/reference");runpy.run_path("tools/reference/test_bf16_source.py",run_name="__main__")'
```

Numerical conversion is explicit: `bf16_source_reference.py` accepts an external
runtime manifest/SHA and a new output directory. Run cold with the pinned
repository virtual environment, `-s -B`, a fresh absolute `-X pycache_prefix`,
and the admitted file inventory from its reviewed collection recipe. The source
report explains the existing runtime union and its unused packages. Raw outputs
and structured partials are retained before comparisons. The large sparse input
stays in ignored artifacts; its exact construction/hash and source are retained
in the public evidence. No runtime/model/quality support follows from this fixture.
