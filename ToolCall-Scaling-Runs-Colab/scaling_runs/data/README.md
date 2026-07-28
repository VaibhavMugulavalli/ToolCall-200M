# Required scaling data placement

The M13, M30, and M60 notebooks all use the same generated bundle. After the
notebook setup cell finishes, this directory must look like:

```text
data/
└── scaling_470m/
    ├── COMPLETE
    ├── bundle_manifest.json
    ├── tokenizer/
    │   └── toolcall_spm_32k.model
    ├── train/
    ├── validation_general/
    └── validation_structured/
```

Do not place the tokenizer beside `scaling_470m`. Its exact required path is:

```text
data/scaling_470m/tokenizer/toolcall_spm_32k.model
```

The generated `scaling_470m.tar.gz` already contains this structure. In Colab,
upload that archive into `/content` using the Files pane before running the
notebook's local-data setup cell. If the bundle lacks a tokenizer, upload the
frozen model as `/content/tokenizer.model` or
`/content/toolcall_spm_32k.model`; the notebook installs it at the required
path.
