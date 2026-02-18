# scope-deeplivecam

Real-time face swap plugin for [Daydream Scope](https://github.com/daydream-ai/scope), powered by [InsightFace](https://github.com/deepinsight/insightface) and the inswapper ONNX model.

## Installation

Install from the Scope UI or CLI:

```bash
uv run daydream-scope install -e /path/to/scope-deeplivecam
```

Models (~900 MB) are downloaded automatically on first load.

## Usage

1. Select **Face Swap** from the pipeline list.
2. Set a **Source Face** image in Input & Controls.
3. Start streaming — all detected faces in the video feed are swapped with the source face.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Source Face | image | — | The face to swap onto targets |
| Detection Size | int (128–1024) | 640 | Face detection resolution (load-time) |
| Swap All Faces | toggle | on | Swap all faces vs. largest only |

## License

[AGPL-3.0](LICENSE)
