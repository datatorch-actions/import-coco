<h1 align="center">
  Import COCO
</h1>

<h4 align="center">Import coco formated json files to DataTorch.</h4>

<p align="center">
  <img alt="DataTorch Action" src="https://img.shields.io/static/v1?label=DataTorch%20Action&message=datatorch/import-coco@v1&color=blueviolet">
  <img alt="Open Issues" src="https://img.shields.io/github/issues/datatorch-actions/import-coco">
</p>

## Quick Start

```yaml
name: 'Import Coco Example'
jobs:
  add:
    steps:
      - name: Import COCO File
        action: datatorch/import-coco@v1
        inputs:
          path: /path/on/agent/to/coco.json
```
