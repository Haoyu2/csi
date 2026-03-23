"""
Test suite for our Intel 5300 CSI reader (main.py) against the csiread library.

csiread (https://github.com/citysu/csiread) is a well-established Cython-based
CSI parser.  We use it as ground-truth to validate that our pure-Python
implementation produces identical results.

Run with:
    pytest test_csi_reader.py -v

Requirements:
    pip install pytest numpy csiread
"""

import os
import numpy as np
import pytest
import csiread

# Import our implementation
from main import read_bf_file, read_bfee, NUM_SUBCARRIERS, SUBCARRIER_INDICES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DAT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "user1-1-1-1-1-r1.dat"
)


@pytest.fixture(scope="module")
def our_records():
    """Parse the .dat file with our implementation (once per test module)."""
    records = read_bf_file(DAT_FILE)
    assert len(records) > 0, "Our reader returned no records"
    return records


@pytest.fixture(scope="module")
def csiread_data():
    """Parse the .dat file with csiread (once per test module).

    csiread.Intel stores all records as arrays indexed by packet number:
        .timestamp_low  : shape (count,)
        .bfee_count     : shape (count,)
        .Nrx            : shape (count,)
        .Ntx            : shape (count,)
        .rssi_a/b/c     : shape (count,)
        .noise          : shape (count,)
        .agc            : shape (count,)
        .perm           : shape (count, 3)  — 0-based indices
        .rate           : shape (count,)
        .csi            : shape (count, 30, nrxnum, ntxnum) — complex128
    """
    # nrxnum and ntxnum are the *maximum* number of antennas across the file;
    # for the Intel 5300 these are always ≤ 3.
    csi = csiread.Intel(DAT_FILE, nrxnum=3, ntxnum=3, pl_size=0)
    csi.read()
    assert csi.count > 0, "csiread returned no records"
    return csi


# ---------------------------------------------------------------------------
# Tests: Record count
# ---------------------------------------------------------------------------


class TestRecordCount:
    """Verify that both parsers find the same number of CSI records."""

    def test_same_count(self, our_records, csiread_data):
        assert len(our_records) == csiread_data.count, (
            f"Record count mismatch: ours={len(our_records)}, "
            f"csiread={csiread_data.count}"
        )


# ---------------------------------------------------------------------------
# Tests: Per-record metadata fields
# ---------------------------------------------------------------------------


class TestMetadataFields:
    """Compare scalar metadata fields record-by-record."""

    def test_timestamp_low(self, our_records, csiread_data):
        """timestamp_low should match exactly (uint32)."""
        for i, rec in enumerate(our_records):
            assert rec["timestamp_low"] == csiread_data.timestamp_low[i], (
                f"Record {i}: timestamp_low mismatch: "
                f"ours={rec['timestamp_low']}, "
                f"csiread={csiread_data.timestamp_low[i]}"
            )

    def test_bfee_count(self, our_records, csiread_data):
        """bfee_count should match exactly (uint16)."""
        for i, rec in enumerate(our_records):
            assert rec["bfee_count"] == csiread_data.bfee_count[i], (
                f"Record {i}: bfee_count mismatch"
            )

    def test_nrx(self, our_records, csiread_data):
        """Number of receive antennas should match."""
        for i, rec in enumerate(our_records):
            assert rec["Nrx"] == csiread_data.Nrx[i], (
                f"Record {i}: Nrx mismatch: "
                f"ours={rec['Nrx']}, csiread={csiread_data.Nrx[i]}"
            )

    def test_ntx(self, our_records, csiread_data):
        """Number of transmit antennas should match."""
        for i, rec in enumerate(our_records):
            assert rec["Ntx"] == csiread_data.Ntx[i], (
                f"Record {i}: Ntx mismatch: "
                f"ours={rec['Ntx']}, csiread={csiread_data.Ntx[i]}"
            )

    def test_rssi_a(self, our_records, csiread_data):
        for i, rec in enumerate(our_records):
            assert rec["rssi_a"] == csiread_data.rssi_a[i], (
                f"Record {i}: rssi_a mismatch"
            )

    def test_rssi_b(self, our_records, csiread_data):
        for i, rec in enumerate(our_records):
            assert rec["rssi_b"] == csiread_data.rssi_b[i], (
                f"Record {i}: rssi_b mismatch"
            )

    def test_rssi_c(self, our_records, csiread_data):
        for i, rec in enumerate(our_records):
            assert rec["rssi_c"] == csiread_data.rssi_c[i], (
                f"Record {i}: rssi_c mismatch"
            )

    def test_noise(self, our_records, csiread_data):
        """Noise floor (signed int8) should match."""
        for i, rec in enumerate(our_records):
            assert rec["noise"] == csiread_data.noise[i], (
                f"Record {i}: noise mismatch: "
                f"ours={rec['noise']}, csiread={csiread_data.noise[i]}"
            )

    def test_agc(self, our_records, csiread_data):
        for i, rec in enumerate(our_records):
            assert rec["agc"] == csiread_data.agc[i], (
                f"Record {i}: agc mismatch"
            )

    def test_rate(self, our_records, csiread_data):
        """fake_rate_n_flags should match."""
        for i, rec in enumerate(our_records):
            assert rec["rate"] == csiread_data.rate[i], (
                f"Record {i}: rate mismatch: "
                f"ours={rec['rate']}, csiread={csiread_data.rate[i]}"
            )


# ---------------------------------------------------------------------------
# Tests: Antenna permutation
# ---------------------------------------------------------------------------


class TestAntennaPerm:
    """Compare antenna permutation arrays.

    Our implementation uses 1-based indices (matching the MATLAB reference).
    csiread uses 0-based indices.
    """

    def test_perm_values(self, our_records, csiread_data):
        for i, rec in enumerate(our_records):
            # Convert our 1-based perm to 0-based for comparison
            our_perm_0based = [p - 1 for p in rec["perm"]]
            csiread_perm = list(csiread_data.perm[i])
            assert our_perm_0based == csiread_perm, (
                f"Record {i}: perm mismatch: "
                f"ours(0-based)={our_perm_0based}, "
                f"csiread={csiread_perm}"
            )


# ---------------------------------------------------------------------------
# Tests: CSI matrix
# ---------------------------------------------------------------------------


class TestCSIMatrix:
    """Compare the complex CSI matrices.

    Shape conventions:
        Our code  : per-record dict with csi shape (Ntx, Nrx, 30)
        csiread   : csi shape (count, 30, nrxnum, ntxnum)

    Note: csiread allocates the full (count, 30, nrxnum, ntxnum) array even
    if some records have fewer antennas.  We compare only the active
    Ntx × Nrx slice per record.
    """

    def test_csi_shape_per_record(self, our_records, csiread_data):
        """Each record's CSI should have shape (Ntx, Nrx, 30)."""
        for i, rec in enumerate(our_records):
            expected_shape = (rec["Ntx"], rec["Nrx"], NUM_SUBCARRIERS)
            assert rec["csi"].shape == expected_shape, (
                f"Record {i}: CSI shape mismatch: "
                f"got {rec['csi'].shape}, expected {expected_shape}"
            )

    def test_csi_values_all_records(self, our_records, csiread_data):
        """CSI complex values should match csiread for every record,
        every subcarrier, every TX-RX pair."""
        mismatches = []

        for i, rec in enumerate(our_records):
            Ntx = rec["Ntx"]
            Nrx = rec["Nrx"]

            for sc in range(NUM_SUBCARRIERS):
                for tx in range(Ntx):
                    for rx in range(Nrx):
                        ours = rec["csi"][tx, rx, sc]
                        # csiread shape: (count, 30, nrxnum, ntxnum)
                        theirs = csiread_data.csi[i, sc, rx, tx]

                        if ours != theirs:
                            mismatches.append(
                                f"  Record {i}, SC {sc}, TX {tx}, RX {rx}: "
                                f"ours={ours}, csiread={theirs}"
                            )

                        # Stop collecting after a reasonable number
                        if len(mismatches) > 20:
                            break
                    if len(mismatches) > 20:
                        break
                if len(mismatches) > 20:
                    break
            if len(mismatches) > 20:
                break

        assert len(mismatches) == 0, (
            f"CSI value mismatches found ({len(mismatches)} shown):\n"
            + "\n".join(mismatches)
        )

    def test_csi_amplitude_close(self, our_records, csiread_data):
        """CSI amplitudes should be numerically identical (not just close)."""
        for i, rec in enumerate(our_records):
            Ntx = rec["Ntx"]
            Nrx = rec["Nrx"]

            our_amp = np.abs(rec["csi"][:Ntx, :Nrx, :])
            # Transpose csiread's (30, Nrx, Ntx) → (Ntx, Nrx, 30)
            cr_slice = csiread_data.csi[i, :, :Nrx, :Ntx]  # (30, Nrx, Ntx)
            cr_amp = np.abs(cr_slice.transpose(2, 1, 0))     # (Ntx, Nrx, 30)

            np.testing.assert_array_almost_equal(
                our_amp, cr_amp, decimal=10,
                err_msg=f"Record {i}: CSI amplitude mismatch"
            )

    def test_csi_phase_close(self, our_records, csiread_data):
        """CSI phases should be numerically identical."""
        for i, rec in enumerate(our_records):
            Ntx = rec["Ntx"]
            Nrx = rec["Nrx"]

            our_phase = np.angle(rec["csi"][:Ntx, :Nrx, :])
            cr_slice = csiread_data.csi[i, :, :Nrx, :Ntx]
            cr_phase = np.angle(cr_slice.transpose(2, 1, 0))

            np.testing.assert_array_almost_equal(
                our_phase, cr_phase, decimal=10,
                err_msg=f"Record {i}: CSI phase mismatch"
            )


# ---------------------------------------------------------------------------
# Tests: First and last record spot checks
# ---------------------------------------------------------------------------


class TestSpotChecks:
    """Quick spot checks on the first and last records for sanity."""

    def test_first_record_ntx_nrx(self, our_records, csiread_data):
        assert our_records[0]["Ntx"] == csiread_data.Ntx[0]
        assert our_records[0]["Nrx"] == csiread_data.Nrx[0]

    def test_last_record_ntx_nrx(self, our_records, csiread_data):
        assert our_records[-1]["Ntx"] == csiread_data.Ntx[-1]
        assert our_records[-1]["Nrx"] == csiread_data.Nrx[-1]

    def test_first_record_timestamp(self, our_records, csiread_data):
        assert our_records[0]["timestamp_low"] == csiread_data.timestamp_low[0]

    def test_last_record_timestamp(self, our_records, csiread_data):
        assert our_records[-1]["timestamp_low"] == csiread_data.timestamp_low[-1]

    def test_first_record_csi_first_element(self, our_records, csiread_data):
        """Check the very first CSI coefficient of the first record."""
        ours = our_records[0]["csi"][0, 0, 0]
        theirs = csiread_data.csi[0, 0, 0, 0]
        assert ours == theirs, f"First CSI element: ours={ours}, csiread={theirs}"

    def test_first_record_csi_last_element(self, our_records, csiread_data):
        """Check the last CSI coefficient of the first record."""
        rec = our_records[0]
        ours = rec["csi"][rec["Ntx"] - 1, rec["Nrx"] - 1, 29]
        theirs = csiread_data.csi[0, 29, rec["Nrx"] - 1, rec["Ntx"] - 1]
        assert ours == theirs, f"Last CSI element: ours={ours}, csiread={theirs}"


# ---------------------------------------------------------------------------
# Tests: Edge cases and sanity
# ---------------------------------------------------------------------------


class TestSanity:
    """General sanity checks on the parsed data."""

    def test_nrx_in_valid_range(self, our_records):
        """Intel 5300 has at most 3 RX antennas."""
        for i, rec in enumerate(our_records):
            assert 1 <= rec["Nrx"] <= 3, (
                f"Record {i}: Nrx={rec['Nrx']} out of range [1,3]"
            )

    def test_ntx_in_valid_range(self, our_records):
        """Intel 5300 has at most 3 TX spatial streams."""
        for i, rec in enumerate(our_records):
            assert 1 <= rec["Ntx"] <= 3, (
                f"Record {i}: Ntx={rec['Ntx']} out of range [1,3]"
            )

    def test_csi_not_all_zero(self, our_records):
        """CSI matrix should not be all zeros (sanity check)."""
        for i, rec in enumerate(our_records):
            assert np.any(rec["csi"] != 0), (
                f"Record {i}: CSI matrix is all zeros"
            )

    def test_subcarrier_count(self, our_records):
        """Each record should have exactly 30 subcarrier groups."""
        for i, rec in enumerate(our_records):
            assert rec["csi"].shape[-1] == 30, (
                f"Record {i}: expected 30 subcarriers, got {rec['csi'].shape[-1]}"
            )

    def test_noise_is_negative(self, our_records):
        """Noise floor should typically be negative (in dBm)."""
        for i, rec in enumerate(our_records):
            assert rec["noise"] < 0, (
                f"Record {i}: noise={rec['noise']} dBm is non-negative"
            )

    def test_timestamps_non_decreasing(self, our_records):
        """Timestamps should generally be non-decreasing (with possible wraps
        at 2^32, so we just check that most are non-decreasing)."""
        decreasing_count = 0
        for i in range(1, len(our_records)):
            if our_records[i]["timestamp_low"] < our_records[i - 1]["timestamp_low"]:
                decreasing_count += 1
        # Allow at most 1 wrap-around
        assert decreasing_count <= 1, (
            f"Timestamps decreased {decreasing_count} times "
            f"(expected at most 1 wrap-around)"
        )
