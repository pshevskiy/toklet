"""Concrete original E2B row-backed weights for the existing reference graph."""
import json
import bf16_source
import gemma4_text_math as graph

PREFIX = 'model.language_model.'


def weights(owner, layout, deadline):
    expected = graph.shapes(graph.E2B); matrices = {}; vectors = {}
    # The first view authenticates the fixed full source/layout identity before
    # parsing names. Each subsequent binding preserves the same contract.
    bf16_source.official_rows(owner, PREFIX + 'norm.weight', layout, deadline=deadline)
    selected = json.loads(layout)['selected_text']
    if set(selected) != {PREFIX + name for name in expected}:
        raise ValueError('Exact official graph weight inventory required')
    for full in sorted(selected):
        name = full.removeprefix(PREFIX); shape = expected[name]
        row = bf16_source.official_rows(owner, full, layout, deadline=deadline)
        if (row.rows, row.columns) != (shape if len(shape) == 2 else (1, shape[0])):
            raise ValueError('Official graph row geometry differs')
        if len(shape) == 2: matrices[name] = row
        else: vectors[name] = row.read_rows(0, 1).reshape(shape)
    return graph.Weights(graph.E2B, matrices, vectors)
