from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class FaceSwapConfig(BasePipelineConfig):
    """Configuration for the Deep Live Cam face swap pipeline."""

    pipeline_id = "deeplivecam-faceswap"
    pipeline_name = "Face Swap"
    pipeline_description = (
        "Real-time face swapping powered by InsightFace and inswapper. "
        "Provide a source face image and the pipeline will swap detected "
        "faces in the input video stream."
    )
    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="ezioruan/inswapper_128.onnx",
            files=["inswapper_128.onnx"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="public-data/insightface",
            files=[
                "models/buffalo_l/1k3d68.onnx",
                "models/buffalo_l/2d106det.onnx",
                "models/buffalo_l/det_10g.onnx",
                "models/buffalo_l/genderage.onnx",
                "models/buffalo_l/w600k_r50.onnx",
            ],
        ),
    ]

    source_face_image: str | None = Field(
        default=None,
        description="Path to the source face image. The face in this image will be swapped onto faces detected in the video stream.",
        json_schema_extra=ui_field_config(
            order=1,
            component="image",
            category="input",
            label="Source Face",
        ),
    )

    det_size: int = Field(
        default=640,
        ge=128,
        le=1024,
        description="Face detection resolution. Higher values detect smaller faces but are slower.",
        json_schema_extra=ui_field_config(
            order=2,
            is_load_param=True,
            label="Detection Size",
        ),
    )

    swap_all_faces: bool = Field(
        default=True,
        description="Swap all detected faces in the frame. When disabled, only the largest face is swapped.",
        json_schema_extra=ui_field_config(
            order=3,
            label="Swap All Faces",
        ),
    )
