import pandas as pd
import numpy as np

df_predictions = pd.read_csv(
    "../results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_prediction_errors_samplewise.csv"
)
df_predictions["model"] = "multiDGD"
df_predictions_mvi = pd.read_csv(
    "../results/analysis/modality_integration/mvi_l20_e2_d2_prediction_errors_samplewise.csv"
)
df_predictions_mvi["model"] = "MultiVI"
df_predictions = pd.concat([df_predictions, df_predictions_mvi])
df_predictions["model"] = df_predictions["model"].astype("category")
df_predictions["model"] = df_predictions["model"].cat.set_categories(["MultiVI", "multiDGD"])
df_abs_stats = df_predictions.groupby(["modality", "model"]).agg({"prediction": ["mean", "std", "count"], "reconstruction": ["mean", "std", "count"]})
df_abs_stats["SE_pred"] = df_abs_stats["prediction"]["std"] / np.sqrt(df_abs_stats["prediction"]["count"])
df_abs_stats["SE_rec"] = df_abs_stats["reconstruction"]["std"] / np.sqrt(df_abs_stats["reconstruction"]["count"])
print(df_abs_stats) # (table 5)
# export
df_abs_stats.to_csv("../results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_prediction_errors_samplewise_summary_abs.csv")

df_predictions["error_ratio"] = df_predictions["prediction"] / df_predictions["reconstruction"]

# print summary statistics (table 4)
df_stats = df_predictions.groupby(["modality", "model"]).agg({"error_ratio": ["mean", "std", "count"]})
df_stats["SE"] = df_stats["error_ratio"]["std"] / np.sqrt(df_stats["error_ratio"]["count"])
print(df_stats)
df_abs_stats.to_csv("../results/analysis/modality_integration/human_bonemarrow_l20_h2-3_rs0_unpaired0percent_prediction_errors_samplewise_summary.csv")