import os
import matplotlib.pyplot as plt

def plot_losses(loss_dict, save_dir, run_id, mode="source"):
    plot_dir = os.path.join(save_dir, "plot", mode)
    os.makedirs(plot_dir, exist_ok=True)

    for loss_name, values in loss_dict.items():
        plt.figure()
        plt.plot(values, label=loss_name)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"{loss_name} over training")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{loss_name}_run{run_id}.png"))
        plt.close()


def plot_metrics(acc_list, f1_list, save_dir, run_id, mode="target"):
    plot_dir = os.path.join(save_dir, "plot", mode)
    os.makedirs(plot_dir, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(acc_list, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"accuracy_run{run_id}.png"))
    plt.close()

    # F1
    plt.figure()
    plt.plot(f1_list, label="F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1-score over epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"f1_run{run_id}.png"))
    plt.close()
