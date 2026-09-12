"""
16-QAM over AWGN - dataset generation for COM 837 Assignment 2, Question 2.

Builds a square 16-QAM constellation on the grid {-3,-1,1,3} x {-3,-1,1,3},
normalises it to unit average symbol energy, and transmits 200 copies of every
constellation point through an AWGN channel at each SNR in [0..30] dB.

Output: qam16_dataset.csv
    snr_db      SNR in dB, filter on this to work at one SNR
    symbol_id   0..15, index of the transmitted constellation point (ground truth)
    tx_I, tx_Q  noiseless transmitted symbol
    rx_I, rx_Q  received symbol after AWGN
"""

import numpy as np
import pandas as pd

SEED = 67
LEVELS = [-3, -1, 1, 3]
SNR_LIST = [0, 5, 10, 15, 20, 25, 30]
N_PER_POINT = 200               # samples per constellation point per SNR

np.random.seed(SEED)


def make_constellation():
    """Square 16-QAM grid, scaled so that the average symbol energy is 1."""
    points = np.array([i + 1j * q for q in LEVELS for i in LEVELS])

    # raw average energy is 10 for the {-3,-1,1,3} grid, so divide by sqrt(10)
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    return points


def transmit(points, snr_db):
    """Repeat every constellation point N_PER_POINT times and add complex AWGN."""
    tx = np.repeat(points, N_PER_POINT)
    symbol_id = np.repeat(np.arange(len(points)), N_PER_POINT)

    # symbol energy is 1, so total noise power is 1/SNR, split over I and Q
    snr_lin = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr_lin))

    noise = (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx))) * sigma
    return symbol_id, tx, tx + noise


def build_dataset():
    points = make_constellation()
    frames = []

    for snr_db in SNR_LIST:
        symbol_id, tx, rx = transmit(points, snr_db)
        frames.append(pd.DataFrame({
            'snr_db': snr_db,
            'symbol_id': symbol_id,
            'tx_I': tx.real,
            'tx_Q': tx.imag,
            'rx_I': rx.real,
            'rx_Q': rx.imag,
        }))
        print(f"  SNR = {snr_db:2d} dB  ->  {len(symbol_id)} symbols")

    return pd.concat(frames, ignore_index=True)


def sanity_check(df):
    """Average symbol energy should be 1, and the measured SNR should match."""
    tx_energy = (df.tx_I ** 2 + df.tx_Q ** 2).mean()
    print(f"\nMean transmitted symbol energy : {tx_energy:.4f}  (expected 1.0)")

    print("\n  snr_db   measured SNR (dB)   symbols per point")
    for snr_db in SNR_LIST:
        sub = df[df.snr_db == snr_db]
        noise_power = ((sub.rx_I - sub.tx_I) ** 2 + (sub.rx_Q - sub.tx_Q) ** 2).mean()
        measured = 10 * np.log10((sub.tx_I ** 2 + sub.tx_Q ** 2).mean() / noise_power)
        counts = sub.symbol_id.value_counts()
        print(f"  {snr_db:6d}   {measured:16.2f}   {counts.min()}-{counts.max()}")


if __name__ == '__main__':
    print("Generating 16-QAM symbols ...")
    dataset = build_dataset()

    dataset.to_csv('qam16_dataset.csv', index=False)
    print(f"\nSaved qam16_dataset.csv  ({len(dataset)} rows)")

    sanity_check(dataset)
