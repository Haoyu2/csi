from pathlib import Path

import pytest
from scipy.io import loadmat

import run_pipeline as rp


REPO_ROOT = Path(__file__).resolve().parents[2]
CSI_PREFIX = REPO_ROOT / "CSI" / "20181208" / "20181208" / "user2" / "user2-1-1-1-1"
BVP_ROOT = REPO_ROOT / "BVP" / "BVP" / "BVP"
EXPECTED_BVP_PATH = BVP_ROOT / "20181208-VS" / "6-link" / "user2" / "user2-1-1-1-1-20181208.mat"


def _has_dataset_clip() -> bool:
    return EXPECTED_BVP_PATH.exists() and all(CSI_PREFIX.with_name(f"{CSI_PREFIX.name}-r{rx}.dat").exists() for rx in range(1, 7))


@pytest.mark.skipif(not _has_dataset_clip(), reason="requires the bundled Widar CSI/BVP sample clip")
def test_build_released_bvp_path_matches_dataset_layout():
    assert rp.build_released_bvp_path(CSI_PREFIX, BVP_ROOT) == EXPECTED_BVP_PATH


@pytest.mark.skipif(not _has_dataset_clip(), reason="requires the bundled Widar CSI/BVP sample clip")
def test_released_bvp_matches_raw_clip_segmentation():
    # One logical clip is stored as six synchronized receiver files:
    #   <clip>-r1.dat ... <clip>-r6.dat
    # The released BVP MAT file stores the result of collapsing those six
    # receiver streams into one velocity_spectrum_ro tensor.
    expected_bvp = loadmat(EXPECTED_BVP_PATH)["velocity_spectrum_ro"]
    raw_frames = rp.load_csi(CSI_PREFIX.with_name(f"{CSI_PREFIX.name}-r1.dat")).shape[0]

    assert expected_bvp.shape[:2] == (20, 20)
    assert expected_bvp.shape[2] == raw_frames // rp.MappingConfig().seg_length

    doppler, freq_bin = rp.compute_doppler_spectrum(CSI_PREFIX)
    assert doppler.shape[:2] == (6, freq_bin.shape[0])
    assert doppler.shape[2] // rp.MappingConfig().seg_length == expected_bvp.shape[2]
