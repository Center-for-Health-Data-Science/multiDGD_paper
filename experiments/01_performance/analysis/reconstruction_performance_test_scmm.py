import numpy as np
import os
import pandas as pd
import torch
from scipy.io import mmread

data_dir = "../../data/"

def compute_error_per_sample(target, output, reduction_type="ms"):
    """compute sample-wise error
    It can be of type `ms` (mean squared) or `ma` (mean absolute)
    """
    error = target - output
    if reduction_type == "ms":
        return torch.mean(error**2, dim=-1)
    elif reduction_type == "ma":
        return torch.mean(torch.abs(error), dim=-1)
    else:
        raise ValueError("invalid reduction type given. Can only be `ms` or `ma`.")


def binary_output_scores(
    target, output, switch, threshold, batch_size=5000, feature_indices=None
):
    """returns FPR, FNR, balanced accuracy, LR+ and LR-"""
    tp, fp, tn, fn = classify_binary_output(
        target, output, switch, threshold, batch_size, feature_indices
    )
    tp = tp.sum()
    fp = fp.sum()
    tn = tn.sum()
    fn = fn.sum()
    tpr = tp / (tp + fn)  # sensitivity
    tnr = tn / (tn + fp)  # specificity
    fpr = 1 - tnr
    fnr = 1 - tpr
    balanced_accuracy = (tpr + tnr) / 2
    positive_likelihood_ratio = tpr / fpr
    negative_likelihood_ratio = fnr / tnr

    return (
        tpr.item(),
        tnr.item(),
        balanced_accuracy.item(),
        positive_likelihood_ratio.item(),
        negative_likelihood_ratio.item(),
    )


def classify_binary_output(
    target, output, switch, threshold, batch_size=50, feature_indices=None
):
    """calculating true positives, false positives, true negatives and false negatives"""
    print("classifying binary output")

    n_samples = target.shape[0]

    x_accessibility = binarize(torch.Tensor(target)).int()
    y_accessibility = output
    if type(y_accessibility) is not torch.Tensor:
        if type(y_accessibility) == pd.core.frame.DataFrame:
            y_accessibility = torch.from_numpy(y_accessibility.values)
            y_accessibility = y_accessibility.detach().cpu()
    else:
        y_accessibility = y_accessibility.detach().cpu()
    y_accessibility = binarize(y_accessibility, threshold).int()
    if feature_indices is not None:
        x_accessibility = x_accessibility[:, feature_indices]
        y_accessibility = y_accessibility[:, feature_indices]
    p = x_accessibility == 1
    pp = y_accessibility == 1
    true_positives = torch.logical_and(p, pp).sum(-1).float()
    true_negatives = torch.logical_and(~p, ~pp).sum(-1).float()
    false_positives = (y_accessibility > x_accessibility).sum(-1).float()
    false_negatives = (y_accessibility < x_accessibility).sum(-1).float()

    return true_positives, false_positives, true_negatives, false_negatives


def binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x


if not os.path.exists(
    "../results/analysis/performance_evaluation/reconstruction/brain_test_counts_gex.npy"
):
    # load brain data
    import mudata as md
    data = md.read(data_dir+"human_brain.h5mu", backed=False)
    test_indices = test_indices = list(np.where(data.obs["train_val_test"] == "test")[0])
    # extract test set
    test_data = data[test_indices, :]
    # now save the test set by modality
    np.save(
        "../results/analysis/performance_evaluation/reconstruction/brain_test_counts_gex.npy",
        test_data["rna"].X.toarray(),
    )
    np.save(
        "../results/analysis/performance_evaluation/reconstruction/brain_test_counts_atac.npy",
        test_data["atac"].X.toarray(),
    )
    test_gex = test_data["rna"].X.toarray()
    test_atac = test_data["atac"].X.toarray()
else:
    print("data sets already generated")
    test_gex = np.load(
        "../results/analysis/performance_evaluation/reconstruction/brain_test_counts_gex.npy"
    )
    test_atac = np.load(
        "../results/analysis/performance_evaluation/reconstruction/brain_test_counts_atac.npy"
    )

n_samples = test_gex.shape[0]

#######################

random_seeds = [0, 37, 8790]

# multiDGD first
for count, seed in enumerate(random_seeds):
    print("seed: ", seed)

    recon_gex = np.asarray(
        mmread(
            "../results/other_models/scMM/scmm_counts_gex_rs"
            + str(seed)
            + ".mtx"
        ).todense()
    )
    # plot recon vs original
    import matplotlib.pyplot as plt

    plt.scatter(test_gex.flatten(), recon_gex.flatten(), s=1)
    plt.xlabel("original")
    plt.ylabel("reconstructed")
    plt.savefig(
        "../results/analysis/plots/performance_evaluation/scmm_counts_gex_rs"
        + str(seed)
        + ".png"
    )

    errors = compute_error_per_sample(
        torch.tensor(test_gex), torch.tensor(recon_gex), reduction_type="ms"
    )
    rmse = torch.sqrt(torch.mean(errors))
    print("RMSE: ", rmse.item())

    errors = compute_error_per_sample(
        torch.tensor(test_gex), torch.tensor(recon_gex), reduction_type="ma"
    )
    mae = torch.sqrt(torch.mean(errors))
    print("MAE: ", mae.item())

    recon_atac = np.asarray(
        mmread(
            "../results/other_models/scMM/scmm_counts_atac_rs"
            + str(seed)
            + ".mtx"
        ).todense()
    )
    threshold = 0.5
    (
        tpr,
        tnr,
        balanced_accuracy,
        positive_likelihood_ratio,
        negative_likelihood_ratio,
    ) = binary_output_scores(
        test_atac, torch.tensor(recon_atac), test_gex.shape[1], threshold
    )
    df_temp = pd.DataFrame(
        {
            "TPR (atac)": tpr,
            "TNR (atac)": tnr,
            "balanced accuracy": balanced_accuracy,
            "LR+": positive_likelihood_ratio,
            "LR-": negative_likelihood_ratio,
            "binary_threshold": threshold,
        },
        index=[0],
    )
    print(df_temp)

    df_temp["RMSE (rna)"] = rmse.item()
    df_temp["MAE (rna)"] = mae.item()
    df_temp["model"] = "scMM"
    df_temp["random_seed"] = seed
    if count == 0:
        metrics_df = df_temp
    else:
        metrics_df = metrics_df.append(df_temp)

metrics_df.to_csv(
    "../results/analysis/performance_evaluation/reconstruction/scMM_brain_recon_performance.csv"
)

print("done")
