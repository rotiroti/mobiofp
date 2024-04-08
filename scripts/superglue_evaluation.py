#!/usr/bin/env python3

import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_matrix(input_dir):
    M = {}

    for file in tqdm(list(input_dir.glob("*.npz"))):
        filename = file.stem
        p_subj, p_ill, p_fin, p_back, p_imp, g_subj, g_ill, g_fin, g_back, g_imp, _ = (
            filename.split("_")
        )
        probe_template = f"{p_subj}_{p_ill}_{p_fin}_{p_back}_{p_imp}"
        gallery_template = f"{g_subj}_{g_ill}_{g_fin}_{g_back}_{g_imp}"
        npz = np.load(file)
        similarity = np.sum(npz["matches"] > -1)

        if probe_template not in M:
            M[probe_template] = {}
        M[probe_template][gallery_template] = similarity

    return M


def probe_vs_gallery_single_instance(M, thresholds):
    results = []

    for t in thresholds:
        GA = 0  # Genuine Acceptances
        FA = 0  # False Acceptances
        FR = 0  # False Rejections
        GR = 0  # Genuine Rejections

        for probe_template, gallery_dict in M.items():
            for gallery_template, similarity in gallery_dict.items():
                p_id = probe_template.split("_")[0]
                g_id = gallery_template.split("_")[0]

                if similarity >= t:
                    if p_id == g_id:  # Genuine pair
                        GA += 1
                    else:  # Impostor pair
                        FA += 1
                else:
                    if p_id == g_id:
                        FR += 1
                    else:
                        GR += 1

        total_genuine = GA + FR
        total_impostor = FA + GR

        GAR = GA / total_genuine if total_genuine > 0 else 0
        FAR = FA / total_impostor if total_impostor > 0 else 0
        FRR = FR / total_genuine if total_genuine > 0 else 0
        GRR = GR / total_impostor if total_impostor > 0 else 0

        results.append((t, GAR, FAR, FRR, GRR))

    return results


def compute_best_matches(M, operator=max):
    matrix_of_best = {}

    for row, columns in M.items():
        results_per_group = {}
        matrix_of_best[row] = {}

        for column, value in columns.items():
            current_subject = column.split("_")[0]

            if current_subject in results_per_group:
                results_per_group[current_subject].append(value)
            else:
                results_per_group[current_subject] = [value]

        for subject, results in results_per_group.items():
            matrix_of_best[row][subject] = operator(results)

    return matrix_of_best


def probe_vs_gallery_multiple_instance(best_matches, thresholds):
    results = []

    for t in thresholds:
        GA = 0  # Genuine Acceptances
        FA = 0  # False Acceptances
        FR = 0  # False Rejections
        GR = 0  # Genuine Rejections

        for p_template in best_matches.keys():
            p_id = p_template.split("_")[0]
            for g_template in best_matches[p_template].keys():
                g_id = g_template.split("_")[0]
                similarity = best_matches[p_template][g_template]

                if similarity >= t:
                    if p_id == g_id:
                        GA += 1
                    else:
                        FA += 1
                else:
                    if p_id == g_id:
                        FR += 1
                    else:
                        GR += 1

        total_genuine = GA + FR
        total_impostor = FA + GR

        GAR = GA / total_genuine if total_genuine > 0 else 0
        FAR = FA / total_impostor if total_impostor > 0 else 0
        FRR = FR / total_genuine if total_genuine > 0 else 0
        GRR = GR / total_impostor if total_impostor > 0 else 0

        results.append((t, GAR, FAR, FRR, GRR))

    return results


def plot_metrics(results, title="Evaluation Metrics"):
    thresholds, _, FAR, FRR, _ = zip(*results)
    FAR = np.array(FAR)
    FRR = np.array(FRR)
    diff = np.abs(FAR - FRR)

    # Compute EER
    eer_index = np.argmin(diff)
    eer_threshold = thresholds[eer_index]
    eer = (FAR[eer_index] + FRR[eer_index]) / 2

    print("=" * 40)
    print(title)
    print(f"Equal Error Rate (EER): {eer:.2f}")
    print(f"Best Threshold: {eer_threshold:.2f}")
    print(f"Accuracy: {1 - eer:.2f}")
    print("=" * 40)

    ONE_MINUS_FRR_sorted = 1 - FRR

    _, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    plt.suptitle(title)

    # Plot FAR, FRR and ERR
    sns.lineplot(x=thresholds, y=FAR, color="blue", label="FAR", ax=axes[0])
    sns.lineplot(x=thresholds, y=FRR, color="red", label="FRR", ax=axes[0])
    axes[0].axvline(x=eer_threshold, color="green", linestyle="--", label="EER Threshold")
    axes[0].set_xlabel("Tolerance Threshold")
    axes[0].set_ylabel("Error Rate")
    axes[0].set_title("FAR, FRR, and EER Threshold")
    axes[0].legend()

    # Plot ROC curve
    sns.lineplot(x=FAR, y=ONE_MINUS_FRR_sorted, ax=axes[1], marker="s")
    axes[1].set_xlabel("False Accept Rate (FAR)")
    axes[1].set_ylabel("1 - False Rejection Rate (1-FRR)")
    axes[1].set_title("WI vs. NO")
    axes[1].grid(True)

    # Plot DET curve
    sns.lineplot(x=FAR, y=FRR, ax=axes[2], color="red")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("False Accept Rate (FAR)")
    axes[2].set_ylabel("False Reject Rate (FRR)")
    axes[2].set_title("Detection Error Tradeoff (DET) Curve")
    axes[2].grid(True)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="The file to evaluate")
    parser.add_argument("--th_min", type=int, default=1, help="Minimum threshold")
    parser.add_argument("--th_max", type=int, default=101, help="Maximum threshold")
    args = parser.parse_args()
    M = create_matrix(args.input_dir)
    thresholds = np.arange(args.th_min, args.th_max, 5)

    single_res = probe_vs_gallery_single_instance(M, thresholds)
    plot_metrics(single_res, "Single Instance Evaluation")

    best_matches = compute_best_matches(M)
    results = probe_vs_gallery_multiple_instance(best_matches, thresholds)
    plot_metrics(results, "Multiple Instance Evaluation")
