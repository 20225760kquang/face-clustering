"""
Flexible Face Embedding Pipeline using InsightFace (ArcFace) or custom ONNX models.

Two modes:
  1. Raw images   → detect → align → embed  (auto mode)
  2. Cropped faces → embed directly           (cropped mode)

Install:
    pip install insightface onnxruntime numpy opencv-python
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os


class FaceEmbedder:
    """
    Flexible face embedding extractor.

    - For raw/uncropped images: detects faces, aligns, then extracts embeddings.
    - For pre-cropped face images: directly extracts embeddings (skips detection).
    - Supports loading a custom recognition model in ONNX format.
    """

    def __init__(self, model_name="buffalo_l", providers=None, det_size=(640, 640), custom_rec_onnx=None,
                 adaface_arch=None, adaface_ckpt_path=None):
        """
        Args:
            model_name: InsightFace model pack name (default: buffalo_l), or 'adaface'.
            providers: ONNX Runtime providers. None = auto-detect (CUDA → CPU fallback).
            det_size: Detection input size (only used in auto mode).
            custom_rec_onnx: Path to a custom recognition ONNX model. If provided, replaces InsightFace's recognition model.
            adaface_arch: Backbone architecture for AdaFace (e.g. 'ir_50').
            adaface_ckpt_path: Path to AdaFace PyTorch checkpoint file (.ckpt).
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.adaface_model = None
        self.custom_rec_onnx = custom_rec_onnx

        if model_name == "adaface":
            # For face detection and keypoints in raw mode, we use buffalo_l
            # to detect faces and extract keypoints, then align.
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.rec_model = None

            # Load PyTorch model
            import torch
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adaface_dir = os.path.join(current_dir, "adaface")
            if adaface_dir not in sys.path:
                sys.path.insert(0, adaface_dir)
            import net

            # Build PyTorch model
            self.adaface_arch = adaface_arch or "ir_50"
            self.adaface_model = net.build_model(self.adaface_arch)

            # Detect device
            use_cuda = torch.cuda.is_available() and any("CUDA" in p for p in providers)
            self.device = torch.device("cuda" if use_cuda else "cpu")

            # Load weights
            if not adaface_ckpt_path or not os.path.exists(adaface_ckpt_path):
                raise FileNotFoundError(f"AdaFace checkpoint file not found at: {adaface_ckpt_path}")
            
            statedict = torch.load(adaface_ckpt_path, map_location=self.device, weights_only=False)['state_dict']
            model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
            self.adaface_model.load_state_dict(model_statedict)
            self.adaface_model.to(self.device)
            self.adaface_model.eval()
            print(f"FaceEmbedder: Loaded AdaFace model ({self.adaface_arch}) from {adaface_ckpt_path} on {self.device}")
        else:
            # Full pipeline (detection + recognition) for raw images
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=det_size)

            # Extract the recognition model for direct embedding on cropped faces
            self.rec_model = self.app.models.get("recognition")

            if self.custom_rec_onnx is not None:
                import onnxruntime as ort
                self.custom_sess = ort.InferenceSession(self.custom_rec_onnx, providers=providers)
                print(f"FaceEmbedder: Loaded custom recognition model from {self.custom_rec_onnx}")
            else:
                if self.rec_model is None:
                    raise RuntimeError("Could not find recognition model in the model pack.")
                print(f"FaceEmbedder ready | Recognition model: {type(self.rec_model).__name__}")

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def align_face(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Align face using 5-point landmarks to standard 112x112 size."""
        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(kps, dst_pts, method=cv2.LMEDS)
        if M is None:
            return cv2.resize(img, (112, 112))
        aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
        return aligned

    def embed_raw(self, img: np.ndarray, max_faces: int = 1) -> list[dict]:
        """
        Full pipeline: detect → align → embed.
        Use for raw/uncropped images that may contain multiple faces.

        Args:
            img: BGR image (as read by cv2.imread).
            max_faces: Max number of faces to return (sorted by size, largest first).

        Returns:
            List of dicts: [{"embedding": np.array, "bbox": [x1,y1,x2,y2], "score": float}, ...]
        """
        faces = self.app.get(img)
        if not faces:
            return []

        # Sort by face area (largest first)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        results = []
        for face in faces[:max_faces]:
            if self.custom_rec_onnx is not None or self.adaface_model is not None:
                # Custom ONNX or AdaFace PyTorch alignment & embedding
                aligned_crop = self.align_face(img, face.kps)
                embedding = self.embed_cropped(aligned_crop)
            else:
                embedding = face.embedding

            results.append({
                "embedding": embedding,
                "bbox": face.bbox.tolist(),
                "score": float(face.det_score),
            })
        return results

    def embed_cropped(self, face_img: np.ndarray) -> np.ndarray:
        """
        Direct embedding: skip detection, assume input is already a cropped & aligned face.

        Args:
            face_img: BGR image of a cropped face. Will be resized to 112x112 internally.

        Returns:
            512-d embedding vector (np.ndarray), L2-normalized.
        """
        if self.adaface_model is not None:
            import torch
            # Preprocess BGR crop for PyTorch AdaFace: resize to 112x112, scale/normalize to [-1, 1], transpose to CHW
            face_resized = cv2.resize(face_img, (112, 112))
            face_float = face_resized.astype(np.float32)
            face_norm = (face_float - 127.5) / 127.5
            face_input = np.transpose(face_norm, (2, 0, 1))
            face_batch = np.expand_dims(face_input, axis=0) # shape (1, 3, 112, 112)

            with torch.no_grad():
                tensor = torch.tensor(face_batch).float().to(self.device)
                feature, _ = self.adaface_model(tensor)
                embedding = feature[0].cpu().numpy()
        elif self.custom_rec_onnx is not None:
            # Preprocess BGR crop: resize, convert to RGB, normalize to [-1, 1], transpose to CHW
            face_resized = cv2.resize(face_img, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_float = face_rgb.astype(np.float32)
            face_norm = (face_float - 127.5) / 127.5
            face_input = np.transpose(face_norm, (2, 0, 1))
            face_batch = np.expand_dims(face_input, axis=0) # shape (1, 3, 112, 112)

            # Run inference
            outputs = self.custom_sess.run(None, {'data': face_batch})
            embedding = outputs[0].flatten()
        else:
            # Resize to model's expected input
            face_resized = cv2.resize(face_img, self.rec_model.input_size)
            # get_feat expects a list of HWC images — it handles preprocessing internally
            embedding = self.rec_model.get_feat([face_resized]).flatten()

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        return embedding

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def embed_file(self, img_path: str, is_cropped: bool = False, max_faces: int = 1):
        """
        Load image from file and extract embedding(s).

        Args:
            img_path: Path to the image file.
            is_cropped: If True, treat as pre-cropped face (skip detection).
            max_faces: Max faces to return (only for is_cropped=False).

        Returns:
            - If is_cropped=True: 512-d embedding (np.ndarray).
            - If is_cropped=False: list of dicts with embedding, bbox, score.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        if is_cropped:
            return self.embed_cropped(img)
        else:
            return self.embed_raw(img, max_faces=max_faces)

    def embed_batch(self, img_paths: list[str], is_cropped: bool = False) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Args:
            img_paths: List of image file paths.
            is_cropped: If True, treat all images as pre-cropped faces.

        Returns:
            np.ndarray of shape (N, 512) — one embedding per image.
            For raw images, only the largest face per image is used.
        """
        embeddings = []
        for path in img_paths:
            try:
                if is_cropped:
                    emb = self.embed_file(path, is_cropped=True)
                else:
                    results = self.embed_file(path, is_cropped=False, max_faces=1)
                    if not results:
                        print(f"  Warning: No face detected in {path}, skipping.")
                        continue
                    emb = results[0]["embedding"]
                embeddings.append(emb)
            except Exception as e:
                print(f"  Error processing {path}: {e}")
        return np.array(embeddings)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
