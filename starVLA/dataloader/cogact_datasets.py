"""Native loader for CogACT/AGIBot episodic datasets.

This adapter intentionally reads the source dataset in place instead of converting
hundreds of gigabytes of video to LeRobot.  It reproduces the robot-space,
absolute-pose data contract used by CogACT's ``AGIBotEpisodicDataset`` while
returning StarVLA's model-agnostic sample dictionary.
"""

from __future__ import annotations

import bisect
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation

# Compact representation used by StarVLA. CogACT stores these values in a
# sparse 470-D ActionFeature tensor; only the following 20 dimensions are
# active for the pick-place datasets.
FEATURE_KEYS = (
    "ROBOT_LEFT_TRANS",
    "ROBOT_LEFT_ROT_6D",
    "ROBOT_LEFT_GRIPPER",
    "ROBOT_RIGHT_TRANS",
    "ROBOT_RIGHT_ROT_6D",
    "ROBOT_RIGHT_GRIPPER",
)
FEATURE_DIMS = (3, 6, 1, 3, 6, 1)
ACTION_DIM = sum(FEATURE_DIMS)
VIEW_NAMES = ("head_color", "hand_left_color", "hand_right_color")

# Raw action.npy fields. Files are raw float32 arrays written with ndarray.tofile.
RAW_FIELDS = (
    "action_head_position",
    "action_joint_position",
    "action_robot_velocity",
    "action_waist_position",
    "action_effector_position",
    "state_end_orientation",
    "state_end_position",
    "state_head_position",
    "state_joint_position",
    "state_waist_position",
    "state_effector_position",
)


def _as_dict(value: Any) -> dict:
    if hasattr(value, "items"):
        return {key: val for key, val in value.items()}
    raise TypeError(f"Expected a mapping, got {type(value).__name__}")


def _matrix_to_cogact_6d(matrix: np.ndarray) -> np.ndarray:
    """Match CogACT's matrix_to_rotation_6d_ext(..., plane='xy').

    Despite the upstream docstring saying "columns", the implementation takes
    ``matrix[..., :2, :]``. Keep that behavior for baseline parity.
    """
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def _compact_stat(stat: dict[str, Sequence[float]]) -> np.ndarray:
    return np.concatenate([np.asarray(stat[key], dtype=np.float32) for key in FEATURE_KEYS])


def _resize_image(image: Image.Image, size: tuple[int, int] | None, mode: str) -> Image.Image:
    if size is None:
        return image
    height, width = size
    if mode == "stretch":
        return image.resize((width, height), Image.Resampling.BILINEAR)
    if mode == "letterbox":
        contained = ImageOps.contain(image, (width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), "black")
        canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
        return canvas
    raise ValueError(f"Unsupported image_resize_mode={mode!r}; use 'stretch' or 'letterbox'.")


class _LMDBFrameReader:
    """Small, worker-local reader for CogACT's JPEG-per-frame LMDB format."""

    def __init__(self, dataset_root: Path):
        self.path = dataset_root / "lmdb" / "frames.lmdb"
        self._env = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def _get_env(self):
        if self._env is None:
            try:
                import lmdb
            except ImportError as exc:
                raise ImportError("CogACT LMDB loading requires `pip install lmdb`.") from exc
            self._env = lmdb.open(
                str(self.path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=512
            )
        return self._env

    def get_frame(self, video_path: str, view_name: str, frame_idx: int) -> Image.Image:
        key = f"{video_path}:{view_name}:{frame_idx:05d}".encode()
        with self._get_env().begin(write=False) as txn:
            encoded = txn.get(key)
        if encoded is None:
            raise KeyError(f"Frame not found in {self.path}: {key.decode()}")
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB")


class CogACTSingleDataset:
    """One CogACT episodic dataset folder exposed as frame-level samples."""

    def __init__(self, data_root_dir: str | Path, spec: dict, common: dict):
        spec = _as_dict(spec)
        common = _as_dict(common)
        self.root = Path(data_root_dir).expanduser() / str(spec["folder"])
        self.name = str(spec.get("name", self.root.name))
        self.weight = float(spec.get("weight", 1.0))
        if self.weight <= 0:
            raise ValueError(f"Dataset weight must be positive for {self.name}, got {self.weight}.")

        self.action_horizon = int(spec.get("action_horizon", common.get("action_horizon", 25)))
        self.stride = int(spec.get("stride", common.get("stride", 1)))
        self.finished_padding_tolerance = int(
            spec.get("finished_padding_tolerance", common.get("finished_padding_tolerance", 3))
        )
        self.quality = set(spec.get("quality", common.get("quality", ["good", "medium"])))
        self.view_names = tuple(spec.get("view_names", common.get("view_names", VIEW_NAMES)))
        self.image_drop_strategy = str(
            spec.get("image_drop_strategy", common.get("image_drop_strategy", "none"))
        )
        if self.image_drop_strategy not in {"none", "balanced_5way"}:
            raise ValueError(
                f"Unsupported image_drop_strategy={self.image_drop_strategy!r}; "
                "use 'none' or 'balanced_5way'."
            )
        if self.image_drop_strategy == "balanced_5way" and set(self.view_names) != set(VIEW_NAMES):
            raise ValueError("balanced_5way requires exactly head_color, hand_left_color, and hand_right_color.")
        self.image_augmentation = bool(
            spec.get("image_augmentation", common.get("image_augmentation", False))
        )
        if self.image_augmentation:
            raise NotImplementedError(
                "Native CogACT image augmentation is not implemented; set image_augmentation=false."
            )
        self.language_augmentation = bool(
            spec.get("language_augmentation", common.get("language_augmentation", False))
        )
        if self.language_augmentation:
            raise NotImplementedError(
                "Canonical language augmentation is not enabled for this baseline; "
                "set language_augmentation=false."
            )
        self.use_lmdb = bool(spec.get("use_lmdb", common.get("use_lmdb", True)))
        self.image_resize_mode = str(spec.get("image_resize_mode", common.get("image_resize_mode", "stretch")))
        image_size = spec.get("image_size", common.get("image_size", [224, 224]))
        self.image_size = tuple(map(int, image_size)) if image_size is not None else None
        self.normalization = str(spec.get("normalization", common.get("normalization", "mean_std")))
        if self.normalization not in {"mean_std", "q99", "none"}:
            raise ValueError(f"Unsupported normalization={self.normalization!r}")

        metadata_file = str(spec["metadata_file"])
        metadata_path = self.root / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        count = len(metadata["video_path"])
        qualities = metadata.get("quality")
        keep = np.ones(count, dtype=bool)
        if qualities is not None:
            keep = np.asarray([value in self.quality for value in qualities], dtype=bool)

        self.video_paths = np.asarray(metadata["video_path"])[keep].astype(str)
        self.instructions = np.asarray(metadata["instructions"])[keep].astype(str)
        self.start_end = np.asarray(metadata["start_end"], dtype=np.int64)[keep]
        self.episode_finished = np.asarray(metadata.get("episode_finished", np.ones(count, dtype=bool)))[keep]
        self.trajectory_lengths = np.maximum(self.start_end[:, 1] - self.start_end[:, 0] - 1, 0)
        valid = self.trajectory_lengths > 0
        self.video_paths = self.video_paths[valid]
        self.instructions = self.instructions[valid]
        self.start_end = self.start_end[valid]
        self.episode_finished = self.episode_finished[valid]
        self.trajectory_lengths = self.trajectory_lengths[valid]
        self._cumulative_lengths = np.cumsum(self.trajectory_lengths).tolist()

        self.action_folder = self._resolve_action_folder(spec.get("action_source", common.get("action_source", "auto")))
        with (self.action_folder / "meta_data.json").open() as stream:
            self.action_metadata = json.load(stream)

        self.unnorm_key, self.statistics = self._load_statistics(
            str(spec.get("statistics_file", common.get("statistics_file", "dataset_statistics.json"))),
            spec.get("statistics_key"),
        )
        self._lmdb = _LMDBFrameReader(self.root) if self.use_lmdb else None
        self._video_readers: OrderedDict[tuple[str, str, int], Any] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_readers"] = OrderedDict()
        return state

    def _resolve_action_folder(self, source: str) -> Path:
        candidates = (
            ("actions_new_gripper", "actions_cmd", "actions_gaussian", "actions")
            if source == "auto"
            else (source,)
        )
        for candidate in candidates:
            path = self.root / candidate
            if path.is_dir():
                return path
        raise FileNotFoundError(f"No action folder in {self.root}; tried {list(candidates)}")

    def _expected_statistics_key(self) -> str:
        base_name = self.root.name.split("_split_", 1)[0]
        return (
            f"{base_name}_robot_space_6d_abs_abs_stride_{self.stride}_"
            f"num_steps_{self.action_horizon}_padding"
        )

    def _load_statistics(self, filename: str, explicit_key: str | None) -> tuple[str, dict]:
        path = self.root / filename
        with path.open() as stream:
            all_statistics = json.load(stream)
        key = explicit_key or self._expected_statistics_key()
        if key not in all_statistics:
            matches = [candidate for candidate in all_statistics if candidate.endswith(
                f"robot_space_6d_abs_abs_stride_{self.stride}_num_steps_{self.action_horizon}_padding"
            )]
            if len(matches) != 1:
                raise KeyError(f"Statistics key {key!r} not found in {path}; matching keys: {matches}")
            key = matches[0]
        source = all_statistics[key]
        compact = {}
        for modality in ("action", "state"):
            compact[modality] = {
                stat_name: _compact_stat(values).tolist()
                for stat_name, values in source[modality].items()
            }
        compact["normalization"] = self.normalization
        compact["feature_keys"] = list(FEATURE_KEYS)
        return key, compact

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if self._cumulative_lengths else 0

    def _index_to_episode_frame(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[episode - 1] if episode else 0
        return episode, index - previous

    def _read_action_rows(self, video_path: str, indices: np.ndarray) -> dict[str, np.ndarray]:
        metadata = self.action_metadata[video_path]
        dims = list(map(int, metadata["dim_list"]))
        total_dim = sum(dims)
        start, stop = int(indices.min()), int(indices.max()) + 1
        path = self.action_folder / video_path / "action.npy"
        with path.open("rb") as stream:
            stream.seek(start * total_dim * np.dtype(np.float32).itemsize)
            raw = np.fromfile(stream, dtype=np.float32, count=(stop - start) * total_dim)
        if raw.size != (stop - start) * total_dim:
            raise ValueError(f"Short action read from {path}: expected {(stop-start)*total_dim}, got {raw.size}")
        raw = raw.reshape(stop - start, total_dim)[indices - start]
        offsets = np.cumsum([0, *dims])
        return {name: raw[:, offsets[i] : offsets[i + 1]] for i, name in enumerate(RAW_FIELDS)}

    def _load_split_video_frame(self, video_path: str, view_name: str, frame_idx: int) -> Image.Image:
        try:
            import decord
        except ImportError as exc:
            raise ImportError("CogACT split-video loading requires `decord`; enable LMDB or install decord.") from exc
        split_length = int(self.root.name.split("_split_", 1)[1].split("_", 1)[0])
        split_id, local_idx = divmod(frame_idx, split_length)
        key = (video_path, view_name, split_id)
        reader = self._video_readers.get(key)
        if reader is None:
            path = self.root / "videos_h264" / video_path / "videos" / f"{view_name}_{split_id}.mp4"
            reader = decord.VideoReader(str(path), num_threads=1)
            self._video_readers[key] = reader
            self._video_readers.move_to_end(key)
            while len(self._video_readers) > 8:
                self._video_readers.popitem(last=False)
        array = reader[local_idx].asnumpy()
        return Image.fromarray(array).convert("RGB")

    def _load_images(self, video_path: str, frame_idx: int) -> list[Image.Image]:
        images_by_view = {}
        for view_name in self.view_names:
            if self._lmdb is not None:
                image = self._lmdb.get_frame(video_path, view_name, frame_idx)
            else:
                image = self._load_split_video_frame(video_path, view_name, frame_idx)
            images_by_view[view_name] = _resize_image(image, self.image_size, self.image_resize_mode)

        if self.image_drop_strategy == "none":
            kept_views = self.view_names
        else:
            # Match CogACT's five equiprobable view combinations:
            # all; head only; both hands; head+right; head+left.
            kept_views = (
                VIEW_NAMES,
                ("head_color",),
                ("hand_left_color", "hand_right_color"),
                ("head_color", "hand_right_color"),
                ("head_color", "hand_left_color"),
            )[np.random.randint(5)]
        return [images_by_view[name] for name in kept_views]

    @staticmethod
    def _compact_features(rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        position = rows["state_end_position"]
        quaternion = rows["state_end_orientation"]
        rotation = Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix().reshape(len(quaternion), 2, 3, 3)
        rotation_6d = _matrix_to_cogact_6d(rotation)
        state_gripper = rows["state_effector_position"] / 120.0
        state = np.concatenate(
            [position[:, :3], rotation_6d[:, 0], state_gripper[:, :1],
             position[:, 3:6], rotation_6d[:, 1], state_gripper[:, 1:2]], axis=-1
        ).astype(np.float32)
        action_gripper = rows["action_effector_position"][:-1]
        action = np.concatenate(
            [position[1:, :3], rotation_6d[1:, 0], action_gripper[:, :1],
             position[1:, 3:6], rotation_6d[1:, 1], action_gripper[:, 1:2]], axis=-1
        ).astype(np.float32)
        return state, action

    def _normalize(self, value: np.ndarray, modality: str) -> np.ndarray:
        stats = self.statistics[modality]
        if self.normalization == "mean_std":
            return ((value - np.asarray(stats["mean"])) / (np.asarray(stats["std"]) + 1e-8)).astype(np.float32)
        if self.normalization == "q99":
            low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
            scale = np.maximum(high - low, 1e-8)
            return np.clip(2 * (value - low) / scale - 1, -2.2, 2.2).astype(np.float32)
        return value.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        episode, relative_frame = self._index_to_episode_frame(index)
        episode_start, episode_end = self.start_end[episode]
        current = int(episode_start + relative_frame)
        indices = current + np.arange(self.action_horizon + 1, dtype=np.int64) * self.stride
        # CogACT treats the first three padded targets of a normally finished
        # episode as valid. This tolerance is zero for unfinished episodes.
        tolerance = self.finished_padding_tolerance if self.episode_finished[episode] else 0
        valid_steps = indices[1:] < episode_end + tolerance
        indices = np.clip(indices, current, episode_end - 1)
        rows = self._read_action_rows(self.video_paths[episode], indices)
        state, action = self._compact_features(rows)
        return {
            "image": self._load_images(self.video_paths[episode], current),
            "lang": self.instructions[episode].strip().lower(),
            "state": self._normalize(state[:1], "state"),
            "action": self._normalize(action, "action"),
            "action_mask": valid_steps,
            "unnorm_key": self.unnorm_key,
            "dataset_name": self.name,
        }


class CogACTMixtureDataset:
    """Configurable weighted mixture of native CogACT datasets.

    Sampling probability and merged Gaussian statistics are weighted by
    ``dataset weight * number of valid frames``, matching CogACT's
    ``enable-norm-merge-v2`` training path. The merged normalizers are written
    back to every in-memory sub-dataset before sampling starts.
    """

    def __init__(
        self,
        datasets: list[CogACTSingleDataset],
        seed: int = 42,
        merge_statistics: bool = True,
    ):
        if not datasets:
            raise ValueError("At least one CogACT dataset must be configured.")
        self.datasets = datasets
        self.seed = int(seed)
        weighted_lengths = np.asarray([len(dataset) * dataset.weight for dataset in datasets], dtype=np.float64)
        self.dataset_sampling_weights = weighted_lengths / weighted_lengths.sum()
        self._length = max(1, int(weighted_lengths.sum()))
        self.trajectory_lengths = np.concatenate([dataset.trajectory_lengths for dataset in datasets])
        self.merged_statistics = None
        if merge_statistics and len(datasets) > 1:
            self.merged_statistics = self._merge_gaussian_statistics()
            for dataset in self.datasets:
                for modality in ("action", "state"):
                    dataset.statistics[modality]["mean"] = self.merged_statistics[modality]["mean"]
                    dataset.statistics[modality]["std"] = self.merged_statistics[modality]["std"]
                dataset.unnorm_key = "merged"

    def _merge_gaussian_statistics(self) -> dict:
        normalizations = {dataset.normalization for dataset in self.datasets}
        if normalizations != {"mean_std"}:
            raise ValueError(
                "Merged multi-dataset statistics require normalization='mean_std'; "
                f"got {sorted(normalizations)}"
            )

        merged = {}
        weights = self.dataset_sampling_weights
        for modality in ("action", "state"):
            means = np.stack(
                [np.asarray(dataset.statistics[modality]["mean"], dtype=np.float64) for dataset in self.datasets]
            )
            stds = np.stack(
                [np.asarray(dataset.statistics[modality]["std"], dtype=np.float64) for dataset in self.datasets]
            )
            mean = np.sum(weights[:, None] * means, axis=0)
            second_moment = np.sum(weights[:, None] * (stds**2 + means**2), axis=0)
            std = np.sqrt(np.maximum(second_moment - mean**2, 0.0))
            merged[modality] = {
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
            }
        merged["normalization"] = "mean_std"
        merged["feature_keys"] = list(FEATURE_KEYS)
        return merged

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        dataset = self.datasets[int(rng.choice(len(self.datasets), p=self.dataset_sampling_weights))]
        return dataset[int(rng.integers(len(dataset)))]

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        if self.merged_statistics is not None:
            output = {"merged": self.merged_statistics}
        else:
            output = {dataset.unnorm_key: dataset.statistics for dataset in self.datasets}
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            json.dump(output, stream, indent=2)


def get_cogact_dataset(data_cfg) -> CogACTMixtureDataset:
    specs = data_cfg.get("dataset_list")
    if not specs:
        raise ValueError("datasets.vla_data.dataset_list must contain at least one dataset spec.")
    common = _as_dict(data_cfg.get("common", {}))
    datasets = [CogACTSingleDataset(data_cfg.data_root_dir, _as_dict(spec), common) for spec in specs]
    return CogACTMixtureDataset(
        datasets,
        seed=int(data_cfg.get("seed", 42)),
        merge_statistics=bool(data_cfg.get("merge_statistics", True)),
    )


def collate_fn(batch):
    return batch


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Smoke-test the native CogACT dataloader.")
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    dataset = get_cogact_dataset(cfg.datasets.vla_data)
    sample = dataset[args.index]
    print(f"Mixture samples per epoch: {len(dataset):,}")
    print(f"Dataset probabilities: {dataset.dataset_sampling_weights.tolist()}")
    print(f"Selected dataset: {sample['dataset_name']}")
    print(f"Instruction: {sample['lang']}")
    print(f"Images: {[image.size for image in sample['image']]}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Valid action steps: {int(sample['action_mask'].sum())}/{len(sample['action_mask'])}")
    print(f"Normalization key: {sample['unnorm_key']}")
