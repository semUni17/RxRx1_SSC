import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE


class TSNEVisualization:
    def __init__(self, classes):
        self.classes = classes

        self.labels = []
        self.embeddings = []

    def initialize(self):
        pass

    def update(self, embeddings, labels):
        self.embeddings.append(embeddings)
        self.labels.append(labels)

    def compute(self):
        self.embeddings = torch.cat(self.embeddings, dim=0).cpu()
        self.labels = torch.cat(self.labels, dim=0).cpu()
        # print(len(torch.unique(self.labels)), torch.unique(self.labels).tolist())

        tsne = TSNE()
        self.embeddings = tsne.fit_transform(self.embeddings)
        self.labels = [self.classes[l] for l in self.labels]

    def plot_embeddings(self):
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        sns.set_style("darkgrid")
        plt.legend(self.labels)
        sns.scatterplot(
            x=self.embeddings[:, 0], y=self.embeddings[:, 1],
            hue=self.labels,
            hue_order=self.labels,
            legend="full",
            palette=sns.color_palette("icefire", n_colors=len(self.labels))
        )
        # plt.savefig("src/results/tmp/embedding_default.png")
        plt.show()
