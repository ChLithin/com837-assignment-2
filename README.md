
# COM 837 - Assignment 2
CH Lithin Sai Kumar, IMT2023598, IIIT Bangalore

Q1 is LOS/NLOS identification with an SVM on channel impulse response features.
Q2 is 16-QAM demodulation treated as a clustering problem with K-means.

Figures and written observations are in `assignment2_report.pdf`.

## Contents

```
q1/  generate_dataset.py        provided with the assignment, unmodified
     Q1_LOS_NLOS_SVM.ipynb      parts (a), (b), (c)
q2/  generate_qam_dataset.py    16-QAM over AWGN, seed 67
     Q2_QAM_Kmeans.ipynb        parts (a), (b), (c)
```

The generated CSVs are not committed since both generators are seeded and
reproduce them exactly.

## Running

```bash
pip install numpy pandas scikit-learn matplotlib jupyter

cd q1
python generate_dataset.py        # writes los_nlos_dataset.csv
jupyter notebook Q1_LOS_NLOS_SVM.ipynb

cd ../q2
python generate_qam_dataset.py    # writes qam16_dataset.csv
jupyter notebook Q2_QAM_Kmeans.ipynb
```

Run each notebook top to bottom. They save their figures as .png into the folder
they are run from.

## Settings

Q1: `SVC(kernel='rbf', C=1)`, 80/20 stratified split per SNR with
`random_state=42`, `StandardScaler` fitted on the training split only. In part (c)
the 25 dB scaler is carried over to every test SNR, since it is part of the frozen
model.

Q2: `KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)`. Purity is
computed by giving each cluster the label of its most frequent `symbol_id`, then
taking the overall fraction of correctly assigned samples.

Nothing is random beyond the two generator seeds and the fixed `random_state`, so
the numbers in the report reproduce exactly.

## Two notes on the implementations

The Rician K-factor uses the power ratio
`|h|max^2 / (sum|h_l|^2 - |h|max^2)`, not Eq. (9) of the reference paper. With only
6 taps the dominant LOS tap inflates `var(|h|)` in Eq. (9), which makes it return a
lower value for LOS than for NLOS. Both versions are in the notebook and the
comparison is in the report.

The phase in Q2 uses `np.arctan2(rx_Q, rx_I)` rather than `arctan(rx_Q/rx_I)`,
which would fold four quadrants onto two and give symbols in opposite quadrants the
same phase.

## Reference

C. Huang, A. F. Molisch, R. He, R. Wang, P. Tang, B. Ai, Z. Zhong, "Machine
Learning-Enabled LOS/NLOS Identification for MIMO Systems in Dynamic Environments",
IEEE Trans. Wireless Commun., vol. 19, no. 6, pp. 3643-3657, 2020.
