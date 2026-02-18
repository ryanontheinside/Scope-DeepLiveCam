"""Face swap pipeline using InsightFace and inswapper ONNX model."""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import FaceSwapConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class FaceSwapPipeline(Pipeline):
    """Real-time face swapping using InsightFace analysis and inswapper ONNX model."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return FaceSwapConfig

    def __init__(
        self,
        det_size: int = 640,
        device: torch.device | None = None,
        **kwargs,
    ):
        import insightface
        from insightface.app import FaceAnalysis

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize face analysis (detection + recognition)
        # Point insightface root to Scope's models dir where buffalo_l was downloaded
        # via the HuggingfaceRepoArtifact(repo_id="public-data/insightface")
        # This gives us: {models_dir}/insightface/models/buffalo_l/*.onnx
        insightface_root = str(get_models_dir() / "insightface")
        self.face_analyser = FaceAnalysis(name="buffalo_l", root=insightface_root)
        self.face_analyser.prepare(ctx_id=0, det_size=(det_size, det_size))

        # Load inswapper model
        model_path = str(
            get_model_file_path("inswapper_128.onnx/inswapper_128.onnx")
        )
        self.swapper = insightface.model_zoo.get_model(
            model_path, download=False
        )

        # Cache for the analysed source face
        self._source_face = None
        self._source_face_path: str | None = None

    def _get_source_face(self, image_path: str | None):
        """Load and cache the source face from the given image path."""
        if image_path is None:
            self._source_face = None
            self._source_face_path = None
            return None

        # Return cached face if the path hasn't changed
        if image_path == self._source_face_path and self._source_face is not None:
            return self._source_face

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Could not read source face image: %s", image_path)
            self._source_face = None
            self._source_face_path = None
            return None

        faces = self.face_analyser.get(img)
        if not faces:
            logger.warning("No face detected in source image: %s", image_path)
            self._source_face = None
            self._source_face_path = None
            return None

        # Pick the largest face by bounding box area
        self._source_face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        self._source_face_path = image_path
        logger.info("Source face loaded from: %s", image_path)
        return self._source_face

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Swap faces in the input video frame.

        Args:
            video: List of input frame tensors, each (1, H, W, C) in [0, 255].
            source_face_image: Path to the source face image.
            swap_all_faces: Whether to swap all detected faces or just the largest.

        Returns:
            Dict with "video" key containing processed frames in [0, 1] range.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for FaceSwapPipeline")

        source_face_image = kwargs.get("source_face_image")
        swap_all_faces = kwargs.get("swap_all_faces", True)

        source_face = self._get_source_face(source_face_image)

        # Normalize frame sizes
        video = normalize_frame_sizes(video)

        # Stack frames: (T, H, W, C)
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)

        # If no source face, pass through as-is
        if source_face is None:
            return {"video": frames.to(dtype=torch.float32) / 255.0}

        # Process each frame
        result_frames = []
        for i in range(frames.shape[0]):
            # Convert to numpy BGR for insightface (expects uint8 BGR)
            frame_rgb = frames[i].cpu().numpy().astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Detect faces in the current frame
            target_faces = self.face_analyser.get(frame_bgr)

            if target_faces:
                if swap_all_faces:
                    for face in target_faces:
                        frame_bgr = self.swapper.get(
                            frame_bgr, face, source_face, paste_back=True
                        )
                else:
                    # Only swap the largest face
                    largest = max(
                        target_faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    frame_bgr = self.swapper.get(
                        frame_bgr, largest, source_face, paste_back=True
                    )

            # Convert back to RGB float [0, 1]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result_frames.append(
                torch.from_numpy(frame_rgb).to(dtype=torch.float32) / 255.0
            )

        result = torch.stack(result_frames, dim=0)
        return {"video": result}
