# MiniCPM5 metadata fixture

Only the pinned config, index and exact header bytes are included. Tests create
a sparse container of the declared size; its holes are zeros, not publisher
weights. This exercises metadata admission without downloads, a GPU, model
allocation or payload authentication. Never use it for numerical or fit claims.
