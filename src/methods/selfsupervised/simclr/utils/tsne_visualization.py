from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


class TSNEVisualization:
    def __init__(self, clases):
        self.classes = clases
        self.tsne = TSNE()

    def visualize(self, embedding, labels):
        embedding_tsne = self.tsne.fit_transform(embedding)
        labels = [self.classes[l] for l in labels]
        self.plot_vecs_n_labels(embedding_tsne, labels)

    def plot_vecs_n_labels(self, v, labels):
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        sns.set_style("darkgrid")
        plt.legend(self.classes)
        sns.scatterplot(
            x=v[:, 0], y=v[:, 1],
            hue=labels,
            hue_order=self.classes,
            legend="full",
            palette=sns.color_palette("bright", n_colors=len(self.classes)) # rocket
        )
        plt.show()
