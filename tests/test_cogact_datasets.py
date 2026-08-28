import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from starVLA.dataloader.cogact_datasets import (
    ACTION_DIM,
    FEATURE_KEYS,
    CogACTMixtureDataset,
    CogACTSingleDataset,
)


class CogACTDatasetTest(unittest.TestCase):
    def _make_dataset(self, root: Path) -> CogACTSingleDataset:
        folder = root / "demo_pickplace_split_500_240h"
        action_dir = folder / "actions_cmd" / "15" / "video_1"
        action_dir.mkdir(parents=True)

        dims = [2, 14, 2, 2, 2, 8, 6, 2, 14, 2, 2]
        with (folder / "actions_cmd" / "meta_data.json").open("w") as stream:
            json.dump({"15/video_1": {"dim_list": dims, "length": 4}}, stream)

        raw = np.zeros((4, sum(dims)), dtype=np.float32)
        # state_end_orientation [22:30]: identity quaternion, xyzw.
        raw[:, 25] = 1
        raw[:, 29] = 1
        # state_end_position [30:36].
        raw[:, 30:36] = np.arange(4, dtype=np.float32)[:, None] + np.arange(6, dtype=np.float32)[None]
        # action gripper [20:22], state gripper [54:56].
        raw[:, 20:22] = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        raw[:, 54:56] = np.asarray([[12, 24], [36, 48], [60, 72], [84, 96]])
        raw.tofile(action_dir / "action.npy")

        metadata = {
            "video_path": np.asarray(["15/video_1", "15/video_1"]),
            "instructions": np.asarray(["Pick an object", "bad episode"]),
            "start_end": np.asarray([[0, 4], [0, 2]]),
            "episode_finished": np.asarray([True, True]),
            "quality": np.asarray(["good", "fail"]),
        }
        np.save(folder / "episodes.npy", metadata)

        key = "demo_pickplace_robot_space_6d_abs_abs_stride_1_num_steps_2_padding"
        feature_stats = {
            stat: {name: [0.0] * dim if stat != "std" else [1.0] * dim
                   for name, dim in zip(FEATURE_KEYS, (3, 6, 1, 3, 6, 1))}
            for stat in ("mean", "std", "min", "max", "q01", "q99")
        }
        with (folder / "dataset_statistics.json").open("w") as stream:
            json.dump({key: {"action": feature_stats, "state": feature_stats}}, stream)

        dataset = CogACTSingleDataset(
            root,
            {"folder": folder.name, "metadata_file": "episodes.npy"},
            {
                "action_horizon": 2,
                "finished_padding_tolerance": 0,
                "normalization": "none",
                "use_lmdb": False,
                "image_size": None,
            },
        )
        return dataset

    def test_filters_quality_and_uses_command_action_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.action_folder.name, "actions_cmd")
            self.assertEqual(dataset.unnorm_key, "demo_pickplace_robot_space_6d_abs_abs_stride_1_num_steps_2_padding")

    def test_compact_action_alignment_and_episode_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            dataset._load_images = lambda *_: [Image.new("RGB", (8, 8))]
            first = dataset[0]
            self.assertEqual(first["action"].shape, (2, ACTION_DIM))
            self.assertEqual(first["state"].shape, (1, ACTION_DIM))
            # Pose target is next-frame state, while gripper target is current-row command.
            np.testing.assert_allclose(first["action"][0, :3], [1, 2, 3])
            self.assertAlmostEqual(float(first["action"][0, 9]), 0.1)
            np.testing.assert_allclose(first["action"][0, 10:13], [4, 5, 6])
            self.assertAlmostEqual(float(first["action"][0, 19]), 0.2)
            np.testing.assert_allclose(first["state"][0, [9, 19]], [0.1, 0.2])
            np.testing.assert_allclose(first["action"][0, 3:9], [1, 0, 0, 0, 1, 0])

            last = dataset[2]
            np.testing.assert_array_equal(last["action_mask"], [True, False])
            pose_dims = list(range(0, 9)) + list(range(10, 19))
            np.testing.assert_allclose(last["action"][0, pose_dims], last["action"][1, pose_dims])

    def test_balanced_five_way_image_dropout(self):
        class FakeLMDB:
            COLORS = {
                "head_color": "red",
                "hand_left_color": "green",
                "hand_right_color": "blue",
            }

            def get_frame(self, video_path, view_name, frame_idx):
                return Image.new("RGB", (8, 8), self.COLORS[view_name])

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            dataset._lmdb = FakeLMDB()
            dataset.image_drop_strategy = "balanced_5way"
            expected_counts = [3, 1, 2, 2, 2]
            for case, expected_count in enumerate(expected_counts):
                with mock.patch("numpy.random.randint", return_value=case):
                    images = dataset._load_images("15/video_1", 0)
                self.assertEqual(len(images), expected_count)

    def test_mixture_applies_and_saves_frame_weighted_global_statistics(self):
        class FakeDataset:
            def __init__(self, length, weight, mean, std, name):
                self._length = length
                self.weight = weight
                self.name = name
                self.unnorm_key = name
                self.normalization = "mean_std"
                self.trajectory_lengths = np.asarray([length])
                block = {
                    "mean": [mean] * ACTION_DIM,
                    "std": [std] * ACTION_DIM,
                }
                self.statistics = {"action": dict(block), "state": dict(block)}

            def __len__(self):
                return self._length

        first = FakeDataset(length=3, weight=1.0, mean=1.0, std=2.0, name="first")
        second = FakeDataset(length=1, weight=1.0, mean=5.0, std=4.0, name="second")
        mixture = CogACTMixtureDataset([first, second])

        # w=(.75,.25): mean=2; E[x^2]=.75*(2^2+1^2)+.25*(4^2+5^2)=14;
        # variance=14-2^2=10.
        np.testing.assert_allclose(mixture.merged_statistics["action"]["mean"], 2.0)
        np.testing.assert_allclose(mixture.merged_statistics["action"]["std"], np.sqrt(10.0))
        np.testing.assert_allclose(first.statistics["action"]["mean"], 2.0)
        np.testing.assert_allclose(second.statistics["state"]["std"], np.sqrt(10.0))
        self.assertEqual(first.unnorm_key, "merged")
        self.assertEqual(second.unnorm_key, "merged")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_statistics.json"
            mixture.save_dataset_statistics(path)
            saved = json.loads(path.read_text())
        self.assertEqual(list(saved), ["merged"])
        np.testing.assert_allclose(saved["merged"]["state"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
