# Address Bundle Schema (v1)

Bundles are portable artifacts representing the "address plane" for routing:
- compressed signatures
- optional compiled dispatch contract
- metadata/config
- optional validation report
- optional `state_dict`

Bundles are directories.

## Files

Required:
- `bundle.json`
- `compressed_signatures.json`

Optional:
- `dispatch_table.json`
- `validation.json`
- `state_dict.pt`
- `manifest.json` (Mesa 15 integrity)

## bundle.json

```json
{
  "meta": {
    "schema_version": 1,
    "created_at": 0,
    "trix_version": "0.12.0",
    "exported_by": "trix export-bundle"
  },
  "config": {
    "ffn_type": "SparseLookupFFNv2",
    "d_model": 64,
    "num_tiles": 8,
    "tiles_per_cluster": 4,
    "grid_size": 16,
    "use_score_calibration": false,
    "routing_backend": "hierarchical_dot",
    "ternary_weight": 0.05,
    "sparsity_weight": 0.05,
    "diversity_weight": 0.05
  },
  "files": {
    "compressed_signatures": "compressed_signatures.json",
    "dispatch_table": "dispatch_table.json",
    "validation": "validation.json",
    "state_dict": "state_dict.pt"
  }
}
```

## Compatibility

The `meta.schema_version` controls compatibility.

- v1 guarantees that:
  - `compressed_signatures.json` can be imported to reconstruct ternary signatures
  - `dispatch_table.json` can be imported into `CompiledDispatch`

## CLI

- Export: `trix export-bundle ...`
- Load/validate: `trix load-bundle --outdir <dir> --validate`

## manifest.json (Mesa 15)

`manifest.json` enables tamper detection and lightweight reproducibility checks.

Fields (v1):

```json
{
  "manifest_version": 1,
  "bundle_schema_version": 1,
  "created_at": "2026-03-01T00:00:00+00:00",
  "trix_version": "0.12.0",
  "file_hashes": {
    "bundle.json": "<sha256>",
    "compressed_signatures.json": "<sha256>",
    "dispatch_table.json": "<sha256>",
    "validation.json": "<sha256>",
    "state_dict.pt": "<sha256>"
  },
  "config_fingerprint": "<sha256(json(config))>",
  "address_plane_fingerprint": "<sha256(json(compressed_signatures))>",
  "contract_fingerprint": "<sha256(json(dispatch_table))>",
  "validation_fingerprint": "<sha256(json(validation))>"
}
```
