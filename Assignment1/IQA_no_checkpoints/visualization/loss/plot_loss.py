from matplotlib import pyplot as plt
import pickle
import numpy as np

def plot_loss(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(title+".pdf")
    # plt.show()

if __name__ == "__main__":
    epoch_loss = pickle.load(
        open("/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/epoch_loss_for_all_iqa_epoch50.pkl", "rb"))
    loss = [np.mean(epoch_loss[i]) for i in range(len(epoch_loss))]
    plot_loss(loss, "training Loss For Metric Transformer")
