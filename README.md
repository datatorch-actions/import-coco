# Import COCO

Action to import coco from a file.

## Example

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
