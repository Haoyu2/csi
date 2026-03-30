#!/usr/bin/env python
"""
Python reimplementation of the Widar 3.0 preprocessing pipeline:
    CSI (.dat from Intel 5300) -> Doppler spectrum -> Body-coordinate Velocity Profile (BVP).

This closely follows the Matlab scripts bundled with the dataset
(`get_doppler_spectrum.m` and `Doppler2VelocityMapping/DVM_main.m`) while using
only Python/NumPy/SciPy/scikit-learn.
"""
from __future__ import annotations

import argparse
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import io as sio
from scipy import optimize, signal
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DopplerConfig:
    samp_rate: int = 1000
    uppe_stop: float = 60.0
    lowe_stop: float = 2.0
    uppe_orde: int = 6
    lowe_orde: int = 3
    method: str = "stft"  # "stft" or "cwt"
    window_fraction: float = 0.25  # window_size = samp_rate * window_fraction


@dataclass
class MappingConfig:
    lambda_: float = 1e-7
    norm_l1: bool = False  # Matlab code uses L0-ish term when norm==0
    seg_length: int = 100
    V_max: float = 2.0
    V_min: float = -2.0
    V_bins: int = 20
    MaxFunctionEvaluations: int = 100_000
    wave_length: float = 299_792_458 / 5.825e9  # 5.825 GHz carrier
    torso_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.365, 0.455],
                [0.455, 0.455],
                [0.455, 1.365],
                [1.365, 1.365],
                [0.91, 0.91],
                [2.275, 1.365],
                [2.275, 2.275],
                [1.365, 2.275],
            ]
        )
    )
    torso_ori: np.ndarray = field(default_factory=lambda: np.array([-90, -45, 0, 45, 90]))
    Tx_pos: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0]]))
    Rx_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.455, -0.455],
                [1.365, -0.455],
                [2.0, 0.0],
                [-0.455, 0.455],
                [-0.455, 1.365],
                [0.0, 2.0],
            ]
        )
    )


# ---------------------------------------------------------------------------
# CSI loading
# ---------------------------------------------------------------------------
def _make_csiread_reader(csiread_module, path: pathlib.Path):
    """
    Support both the newer ``nrx/ntx`` and older ``nrxnum/ntxnum`` csiread APIs.
    """
    constructor_options = (
        {"nrx": 3, "ntx": 1, "if_report": False},
        {"nrxnum": 3, "ntxnum": 1, "if_report": False},
    )
    last_exc = None
    for kwargs in constructor_options:
        try:
            return csiread_module.Intel(str(path), **kwargs)
        except TypeError as exc:
            last_exc = exc
    raise RuntimeError(f"Unsupported csiread.Intel signature for {path}") from last_exc


def load_csi(path: pathlib.Path) -> np.ndarray:
    """
    Read one receiver's CSI .dat file with csiread and return complex array shaped (frames, subcarriers*antennas).
    """
    try:
        import csiread
    except ImportError as exc:
        raise RuntimeError("csiread is required; pip install -r python_pipeline/requirements.txt") from exc

    reader = _make_csiread_reader(csiread, path)
    reader.read()
    csi = reader.csi  # shape: (frames, Nrx, Ntx, Nsub) or (frames, Nsub, Nrx)
    csi = csi.reshape(csi.shape[0], -1)  # flatten antenna & subcarrier dims -> (frames, 90)
    return csi.astype(np.complex128)


def infer_clip_date(base_prefix: pathlib.Path) -> str:
    """
    Infer the recording date embedded in the CSI directory hierarchy.
    """
    prefix = pathlib.Path(base_prefix)
    for part in reversed(prefix.parts[:-1]):
        if len(part) == 8 and part.isdigit():
            return part
    raise ValueError(f"Could not infer clip date from {base_prefix}")


def build_released_bvp_path(base_prefix: pathlib.Path, bvp_root: pathlib.Path) -> pathlib.Path:
    """
    Map one logical CSI clip to the matching released BVP MAT file.

    Relationship:
    - Raw CSI stores one clip as six receiver recordings:
      ``<clip>-r1.dat`` ... ``<clip>-r6.dat``.
    - The released BVP dataset collapses those six synchronized receiver files
      into one ``velocity_spectrum_ro`` tensor stored at:
      ``<date>-VS/6-link/<user>/<clip>-<date>.mat``.
    """
    prefix = pathlib.Path(base_prefix)
    clip_id = prefix.name
    user_id = clip_id.split("-")[0]
    clip_date = infer_clip_date(prefix)
    return pathlib.Path(bvp_root) / f"{clip_date}-VS" / "6-link" / user_id / f"{clip_id}-{clip_date}.mat"


# ---------------------------------------------------------------------------
# Doppler spectrum (mirrors get_doppler_spectrum.m)
# ---------------------------------------------------------------------------
def bandpass_filter(x: np.ndarray, cfg: DopplerConfig) -> np.ndarray:
    samp_rate = cfg.samp_rate
    b_low, a_low = signal.butter(cfg.uppe_orde, cfg.uppe_stop / (samp_rate / 2), btype="low")
    b_high, a_high = signal.butter(cfg.lowe_orde, cfg.lowe_stop / (samp_rate / 2), btype="high")
    x = signal.lfilter(b_low, a_low, x, axis=0)
    x = signal.lfilter(b_high, a_high, x, axis=0)
    return x


def _select_reference(csi: np.ndarray, rx_acnt: int) -> Tuple[np.ndarray, int]:
    """
    Choose the reference antenna index following the Matlab heuristic:
    pick the antenna whose mean/variance ratio is largest.
    """
    csi_mean = np.mean(np.abs(csi), axis=0)
    csi_std = np.std(np.abs(csi), axis=0, ddof=0)
    ratio = csi_mean / (csi_std + 1e-9)
    ratio = ratio.reshape(rx_acnt, -1).mean(axis=1)
    idx = int(np.argmax(ratio))  # 0-based
    # Build reference matrix (repeat selected antenna across antennas)
    ref = np.tile(csi[:, idx * 30 : (idx + 1) * 30], (1, rx_acnt))
    return ref, idx


def _amp_adjust(csi: np.ndarray, rx_acnt: int) -> np.ndarray:
    """
    Apply amplitude adjustment analogous to the Matlab implementation.
    """
    adjusted = np.zeros_like(csi, dtype=np.complex128)
    alpha_sum = 0.0
    for col in range(csi.shape[1]):
        amp = np.abs(csi[:, col])
        alpha = np.min(amp[amp != 0]) if np.any(amp != 0) else 0.0
        alpha_sum += alpha
        adjusted[:, col] = np.abs(amp - alpha) * np.exp(1j * np.angle(csi[:, col]))
    beta = 1000 * alpha_sum / (30 * rx_acnt)
    ref = np.zeros_like(csi, dtype=np.complex128)
    for col in range(csi.shape[1]):
        ref[:, col] = (np.abs(csi[:, col]) + beta) * np.exp(1j * np.angle(csi[:, col]))
    return adjusted, ref


def compute_doppler_spectrum(
    prefix: pathlib.Path,
    rx_cnt: int = 6,
    rx_acnt: int = 3,
    cfg: DopplerConfig = DopplerConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Doppler spectrum for all receivers.
    Returns:
        doppler_spectrum: (rx_cnt, F, T)
        freq_bin: (F,)

    ``prefix`` identifies one logical gesture clip, for example
    ``user2-1-1-1-1``. The raw CSI for that clip is spread across six files
    ``<prefix>-r1.dat`` ... ``<prefix>-r6.dat``. The released BVP MAT file for
    the same clip is a single tensor derived from those six receiver streams.
    """
    spectra: List[np.ndarray] = []
    freq_bin = None

    for rx in range(1, rx_cnt + 1):
        path = pathlib.Path(f"{prefix}-r{rx}.dat")
        logging.info("Loading CSI %s", path)
        csi = load_csi(path)

        # Bandpass filtering per column
        adjusted, ref = _amp_adjust(csi, rx_acnt)
        conj_mult = adjusted * np.conj(ref)

        # Drop reference antenna's own columns
        _, idx = _select_reference(csi, rx_acnt)
        mask = np.ones(conj_mult.shape[1], dtype=bool)
        mask[idx * 30 : (idx + 1) * 30] = False
        conj_mult = conj_mult[:, mask]

        conj_mult = bandpass_filter(conj_mult, cfg)

        # PCA -> first principal component
        pca = PCA(n_components=1)
        # Matlab's PCA path operates on values that are effectively real after
        # conjugate multiplication and filtering; scikit-learn requires us to
        # drop the residual numerical imaginary component explicitly.
        principal = pca.fit_transform(np.real(conj_mult)).squeeze(axis=-1)  # (frames,)

        # STFT
        window_size = int(round(cfg.samp_rate * cfg.window_fraction + 1))
        if window_size % 2 == 0:
            window_size += 1
        f, t, Zxx = signal.stft(
            principal,
            fs=cfg.samp_rate,
            window=("gaussian", window_size / 6.0),
            nperseg=window_size,
            # Matlab's tfrsp evaluates the STFT at every time index.
            noverlap=window_size - 1,
            nfft=cfg.samp_rate,  # 1 Hz resolution
            boundary="zeros",
            padded=True,
        )

        # Select |freq| <= uppe_stop
        freq_mask = (f <= cfg.uppe_stop) & (f >= -cfg.uppe_stop)
        f_sel = f[freq_mask]
        spec = np.abs(Zxx[freq_mask, :])

        # Normalize columns
        col_sum = np.sum(spec, axis=0, keepdims=True) + 1e-12
        spec = spec / col_sum

        if freq_bin is None:
            freq_bin = f_sel
        spectra.append(spec)

    # Pad to common time length (max across receivers)
    max_T = max(s.shape[1] for s in spectra)
    padded = []
    for spec in spectra:
        if spec.shape[1] < max_T:
            spec = np.pad(spec, ((0, 0), (0, max_T - spec.shape[1])), mode="constant")
        padded.append(spec)

    doppler_spectrum = np.stack(padded, axis=0)
    return doppler_spectrum, freq_bin


# ---------------------------------------------------------------------------
# Doppler -> Velocity mapping (mirrors DVM_main.m)
# ---------------------------------------------------------------------------
def get_A_matrix(torso_pos: np.ndarray, Tx_pos: np.ndarray, Rx_pos: np.ndarray, rxcnt: int) -> np.ndarray:
    if rxcnt > Rx_pos.shape[0]:
        raise ValueError("rxcnt exceeds provided Rx_pos rows")
    A = np.zeros((rxcnt, 2))
    for ii in range(rxcnt):
        dis_torso_tx = np.linalg.norm(torso_pos - Tx_pos)
        dis_torso_rx = np.linalg.norm(torso_pos - Rx_pos[ii, :])
        A[ii, :] = (torso_pos - Tx_pos) / dis_torso_tx + (torso_pos - Rx_pos[ii, :]) / dis_torso_rx
    return A


def get_velocity2doppler_mapping_matrix(A: np.ndarray, wave_length: float, velocity_bin: np.ndarray, freq_bin: np.ndarray):
    rx_cnt = A.shape[0]
    F = freq_bin.shape[0]
    M = velocity_bin.shape[0]
    freq_min = int(np.min(freq_bin))
    freq_max = int(np.max(freq_bin))
    VDM = np.zeros((rx_cnt, M, M, F))
    for rx in range(rx_cnt):
        for i in range(M):
            for j in range(M):
                plcr_hz = int(round(A[rx, :].dot(np.array([velocity_bin[i], velocity_bin[j]])) / wave_length))
                if plcr_hz > freq_max or plcr_hz < freq_min:
                    VDM[rx, i, j, :] = 1e10
                    continue
                idx = plcr_hz - freq_min
                VDM[rx, i, j, idx] = 1.0
    return VDM


def align_doppler_bins_for_mapping(doppler_spectrum: np.ndarray, freq_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder wrapped STFT bins into ascending frequency order before DVM.

    Matlab does this with a circshift in ``DVM_main.m``. Sorting both the
    Doppler tensor and the frequency vector is equivalent and makes the mapping
    math explicit.
    """
    order = np.argsort(freq_bin)
    return doppler_spectrum[:, order, :], freq_bin[order]


def dvm_loss(P_flat, VDM, target, lambda_, norm_l1):
    """
    Implements DVM_target_func in Matlab (EMD + optional regularization).
    """
    rx_cnt, F = target.shape
    M = int(math.sqrt(P_flat.size))
    P = P_flat.reshape(M, M)
    # Approx doppler: einsum over MxM with VDM -> (rx,F)
    approx = np.einsum("ij,rijf->rf", P, VDM)
    diff_cum = np.cumsum(approx - target, axis=1)
    floss = np.sum(np.abs(diff_cum))
    if norm_l1:
        floss += lambda_ * np.sum(np.abs(P))
    else:
        floss += lambda_ * np.count_nonzero(P)
    return floss


def doppler_to_bvp(
    doppler_spectrum: np.ndarray,
    freq_bin: np.ndarray,
    rx_cnt: int,
    pos_sel: int,
    ori_sel: int,
    map_cfg: MappingConfig = MappingConfig(),
) -> np.ndarray:
    """
    Convert doppler_spectrum (rx_cnt, F, T) to rotated velocity spectrum [M, M, segs].
    """
    # One released BVP MAT file corresponds to the six CSI receiver files for
    # a single clip token. This step is the many-file -> one-tensor collapse.
    doppler_spectrum, freq_bin = align_doppler_bins_for_mapping(doppler_spectrum, freq_bin)
    V_max, V_min, V_bins = map_cfg.V_max, map_cfg.V_min, map_cfg.V_bins
    V_resolution = (V_max - V_min) / V_bins
    M = int((V_max - V_min) / V_resolution)
    velocity_bin = ((np.arange(1, M + 1) - M / 2) / (M / 2)) * V_max
    A = get_A_matrix(map_cfg.torso_pos[pos_sel - 1], map_cfg.Tx_pos[0], map_cfg.Rx_pos, rx_cnt)
    freq_bin_int = np.round(freq_bin).astype(int)
    VDM = get_velocity2doppler_mapping_matrix(A, map_cfg.wave_length, velocity_bin, freq_bin_int)

    seg_length = map_cfg.seg_length
    seg_number = doppler_spectrum.shape[2] // seg_length
    doppler_max = np.max(doppler_spectrum)
    U_bound = np.full((M, M), doppler_max)
    velocity_spectrum = np.zeros((M, M, seg_number))

    for ii in tqdm(range(seg_number), desc="Mapping Doppler->BVP"):
        doppler_seg = doppler_spectrum[:, :, ii * seg_length : (ii + 1) * seg_length]
        doppler_tgt = doppler_seg.mean(axis=2)  # average over time within segment

        # Normalize receivers to first receiver power
        for jj in range(1, doppler_tgt.shape[0]):
            if np.any(doppler_tgt[jj, :]):
                doppler_tgt[jj, :] *= doppler_tgt[0, :].sum() / (doppler_tgt[jj, :].sum() + 1e-12)

        x0 = np.zeros(M * M)
        bounds = optimize.Bounds(lb=np.zeros_like(x0), ub=U_bound.flatten())
        res = optimize.minimize(
            dvm_loss,
            x0,
            args=(VDM, doppler_tgt, map_cfg.lambda_, map_cfg.norm_l1),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": map_cfg.MaxFunctionEvaluations, "ftol": 1e-6, "disp": False},
        )
        velocity_spectrum[:, :, ii] = res.x.reshape(M, M)

    # Rotate by orientation
    orientation_deg = map_cfg.torso_ori[ori_sel - 1]
    if M % 2 == 0:
        # Matlab pads even grids to odd size before rotating, then crops back.
        velocity_spectrum = np.pad(velocity_spectrum, ((1, 0), (1, 0), (0, 0)), mode="constant")
        rotated = np.stack(
            [
                rotate(
                    velocity_spectrum[:, :, k],
                    orientation_deg,
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                for k in range(seg_number)
            ],
            axis=2,
        )[1:, 1:, :]
    else:
        rotated = np.stack(
            [
                rotate(
                    velocity_spectrum[:, :, k],
                    orientation_deg,
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                for k in range(seg_number)
            ],
            axis=2,
        )
    return rotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end CSI -> Doppler -> BVP pipeline (Python).")
    parser.add_argument("--base-prefix", required=True, help="Path prefix to clip, e.g., D:/.../Data/userA-1-1-1-1")
    parser.add_argument("--people", default="userA")
    parser.add_argument("--motion", type=int, default=1)
    parser.add_argument("--pos", type=int, default=1)
    parser.add_argument("--ori", type=int, default=1)
    parser.add_argument("--ges", type=int, default=1)
    parser.add_argument("--rx-cnt", type=int, default=6)
    parser.add_argument("--rx-acnt", type=int, default=3)
    parser.add_argument("--out-dir", default="python_pipeline/output", help="Where to store doppler and BVP outputs")
    parser.add_argument("--save-mat", action="store_true", help="Also save .mat files alongside .npz")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prefix = pathlib.Path(args.base_prefix)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doppler, freq_bin = compute_doppler_spectrum(prefix, rx_cnt=args.rx_cnt, rx_acnt=args.rx_acnt)
    doppler_path = out_dir / f"{args.people}-{args.motion}-{args.pos}-{args.ori}-{args.ges}_doppler.npz"
    np.savez_compressed(doppler_path, doppler=doppler, freq_bin=freq_bin)
    logging.info("Saved doppler spectrum -> %s", doppler_path)

    bvp = doppler_to_bvp(doppler, freq_bin, args.rx_cnt, args.pos, args.ori)
    bvp_path = out_dir / f"{args.people}-{args.motion}-{args.pos}-{args.ori}-{args.ges}_bvp.npz"
    np.savez_compressed(bvp_path, velocity_spectrum_ro=bvp)
    logging.info("Saved BVP -> %s", bvp_path)

    if args.save_mat:
        mat_path = bvp_path.with_suffix(".mat")
        sio.savemat(mat_path, {"velocity_spectrum_ro": bvp})
        logging.info("Saved Matlab-compatible BVP -> %s", mat_path)


if __name__ == "__main__":
    main()
