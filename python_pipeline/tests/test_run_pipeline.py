import types
import numpy as np
import run_pipeline as rp


def test_get_velocity2doppler_mapping_matrix_shape_and_hits():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    velocity_bin = np.array([-1, 0, 1])
    freq_bin = np.array([-1, 0, 1])
    VDM = rp.get_velocity2doppler_mapping_matrix(A, wave_length=1.0, velocity_bin=velocity_bin, freq_bin=freq_bin)
    assert VDM.shape == (2, 3, 3, 3)
    # For A = identity and wave_length=1, mapping of (1,0) should hit freq +1 on rx1, zero elsewhere
    assert VDM[0, 2, 1, 2] == 1.0  # rx0, vx=1, vy=0 -> +1 Hz (index 2 because freq_bin min=-1)
    assert np.all(VDM[1, 2, 1, :] == 0)  # rx1 sees freq 0 because A[1]=[0,1] and vy=0


def test_compute_doppler_spectrum_shapes(monkeypatch):
    # Synthetic CSI: 300 frames, 30 subcarriers * 1 antenna = 30 cols
    frames = 300
    t = np.arange(frames)
    tone = np.exp(1j * 2 * np.pi * 5 * t / 1000)  # 5 Hz tone
    csi_single = np.tile(tone[:, None], (1, 30))

    def fake_load_csi(path):
        return csi_single

    monkeypatch.setattr(rp, "load_csi", fake_load_csi)
    doppler, freq = rp.compute_doppler_spectrum(prefix="dummy", rx_cnt=2, rx_acnt=1)
    assert doppler.shape[0] == 2
    assert doppler.shape[1] == freq.shape[0]
    # Ensure time axis is at least a few frames from STFT
    assert doppler.shape[2] > 1
    # Spectra should be normalized per time slice
    col_sums = doppler.sum(axis=1)
    assert np.allclose(col_sums[col_sums > 0], 1.0, atol=1e-6)


def test_doppler_to_bvp_runs_small_problem(monkeypatch):
    # Small synthetic doppler: rx_cnt=2, F=3, T=6 (two segments of length 3)
    doppler = np.abs(np.random.rand(2, 3, 6)) + 0.1
    freq_bin = np.array([-1, 0, 1])

    # Slim config for speed
    cfg = rp.MappingConfig(
        lambda_=1e-7,
        norm_l1=True,
        seg_length=3,
        V_max=1.0,
        V_min=-1.0,
        V_bins=2,
        MaxFunctionEvaluations=50,
        wave_length=1.0,
    )

    bvp = rp.doppler_to_bvp(doppler, freq_bin, rx_cnt=2, pos_sel=1, ori_sel=1, map_cfg=cfg)
    # Shape: M x M x segments, where M = (V_max - V_min)/resolution = 2
    assert bvp.shape == (2, 2, 2)
    assert np.all(np.isfinite(bvp))
