import os

import hydra
import torch
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from pathlib import Path
from src.datasets.data_utils import parse_dataset_speakers, extract_speaker_id
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.colors as mcolors

def read_embeddings(directory) -> dict:

    embeddings = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            key = data['name']
            embedding = torch.nn.functional.normalize(data['embedding'], p=2, dim=0)
            embeddings[key] = embedding.cpu().numpy()
    return embeddings

def read_dataset_index(index_path) -> dict:
    return read_json(str(index_path))


def visualize_embeddings(embeddings_dict, labels_dict, title="Speaker Embeddings Visualization",
                         figsize=(20, 15), n_components=2, perplexity=30, n_iter=1000,
                         random_state=42, dpi=300, speakers=0):
    """
    Visualizes embeddings using t-SNE.

    Args:
    embeddings_dict: A dictionary of the form {model_name: embeddings}
    labels_dict: A dictionary of the form {model_name: labels}
    title: Title of the plot
    figsize: Size of the figure
    n_components: Number of components for t-SNE (2 or 3)
    perplexity: Perplexity parameter for t-SNE
    n_iter: Number of iterations for t-SNE
    random_state: Random state for reproducibility
    dpi: Image resolution
    """
    n_models = len(embeddings_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = axes.flatten()

    all_labels = np.concatenate([np.array(labels) for labels in labels_dict.values()])
    unique_labels = np.unique(all_labels)
    n_speakers = len(unique_labels)

    label_to_color_idx = {label: idx for idx, label in enumerate(unique_labels)}

    colors = sns.color_palette('husl', n_speakers)
    cmap = ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=n_speakers - 1)

    for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        labels = np.array(labels_dict[model_name])

        print(f"run t-SNE for {model_name}...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    n_iter=n_iter, random_state=random_state)
        embeddings_np = np.array(embeddings)
        embeddings_2d = tsne.fit_transform(embeddings_np)

        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'speaker': labels
        })

        color_indices = np.array([label_to_color_idx[label] for label in df['speaker']])

        ax = axes[i]
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
        ax.set_aspect('equal')

        scatter = ax.scatter(df['x'], df['y'], c=color_indices, cmap=cmap, norm=norm,
                             alpha=0.7, s=50, edgecolors='w', linewidths=0.5)

        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f8f9fa')

    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.92)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[label_to_color_idx[label]],
                                  markersize=10, label=f'Speaker {label}')
                       for label in unique_labels]

    if n_speakers <= speakers:
        fig.legend(handles=legend_elements,
                   loc='lower center', ncol=min(10, n_speakers),
                   bbox_to_anchor=(0.5, 0.01), fontsize=12)
    fig.subplots_adjust(top=0.9)

    return fig

@hydra.main(version_base=None, config_path="src/configs", config_name="t-sne")
def main(config):
    embeddings_paths = config.embeddings_paths
    names = config.names

    index_path = config.index_path
    index = read_dataset_index(index_path)
    set_of_speakers_id = set()

    for data in index:
        speaker_id = Path(*Path(data['name']).parts[0 + config.name_starts:1 + config.name_starts])
        if config.fix_id:
            speaker_id = f'id{speaker_id}'
        speaker_id = str(speaker_id)
        speaker_id = extract_speaker_id(speaker_id)
        set_of_speakers_id.add(speaker_id)

    model_embeddings = {}
    model_labels = {}

    for ind in range(len(embeddings_paths)):
        embeddings_path = embeddings_paths[ind]
        model_name = names[ind]


        embeddings = read_embeddings(embeddings_path)


        embeddings_vis = []
        labels_vis = []
        give_label = {}

        for name, embedding in embeddings.items():
            speaker_id = Path(*Path(name).parts[0 + config.name_starts:1 + config.name_starts])
            if config.fix_id:
                speaker_id = f'id{speaker_id}'
            speaker_id = str(speaker_id)
            speaker_id = extract_speaker_id(speaker_id)
            if speaker_id not in set_of_speakers_id:
                continue
            if speaker_id not in give_label:
                give_label[speaker_id] = len(give_label)

            embeddings_vis.append(embedding)
            labels_vis.append(give_label[speaker_id])
        model_embeddings[model_name] = np.array(embeddings_vis)
        model_labels[model_name] = np.array(labels_vis)

    fig = visualize_embeddings(
        model_embeddings,
        model_labels,
        title=config.title,
        figsize=(24, 12),
        perplexity=40,
        speakers=config.speakers
    )

    plt.savefig('speaker_embeddings_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
   main()






