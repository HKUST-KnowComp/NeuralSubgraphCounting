import os
import sys
import json
import numpy as np
from utils import get_best_epochs, compute_mae, compute_rmse, compute_p_r_f1, compute_tp


if __name__ == "__main__":
    assert len(sys.argv) == 2
    model_dir = sys.argv[1]

    # get the best epoch
    if os.path.exists(os.path.join(model_dir, "finetune_log.txt")):
        best_epochs = get_best_epochs(os.path.join(model_dir, "finetune_log.txt"))
    elif os.path.exists(os.path.join(model_dir, "train_log.txt")):
        best_epochs = get_best_epochs(os.path.join(model_dir, "train_log.txt"))
    else:
        raise FileNotFoundError("finetune_log.txt and train_log.txt cannot be found in %s" % (os.path.join(model_dir)))
    print("retrieve the best epoch for training set ({:0>3d}), dev set ({:0>3d}), and test set ({:0>3d})".format(
        best_epochs["train"], best_epochs["dev"], best_epochs["test"]))

    with open(os.path.join(model_dir, "dev%d.json" % (best_epochs["dev"])), "r") as f:
        results = json.load(f)
        pred = np.array(results["data"]["pred"])
        counts = np.array(results["data"]["counts"])
        print("dev-RMSE: %.4f\tdev-MAE: %.4f\tdev-F1_Zero: %.4f\tdev-F1_NonZero: %.4f\tdev-Time: %.4f" % (
            compute_rmse(pred, counts), compute_mae(pred, counts),
            compute_p_r_f1(pred < 0.5, counts < 0.5)[2], compute_p_r_f1(pred > 0.5, counts > 0.5)[2],
            results["time"]["total"]))

    with open(os.path.join(model_dir, "test%d.json" % (best_epochs["dev"])), "r") as f:
        results = json.load(f)
        pred = np.array(results["data"]["pred"])
        counts = np.array(results["data"]["counts"])
        print("test-RMSE: %.4f\ttest-MAE: %.4f\ttest-F1_Zero: %.4f\ttest-F1_NonZero: %.4f\ttest-Time: %.4f" % (
            compute_rmse(pred, counts), compute_mae(pred, counts),
            compute_p_r_f1(pred < 0.5, counts < 0.5)[2], compute_p_r_f1(pred > 0.5, counts > 0.5)[2],
            results["time"]["total"]))