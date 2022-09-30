import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from tqdm import tqdm

from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap


def save_correct_preds_and_attrs(datamodule, model, xai_models, label_list, activity='Walking', max_capacity=10):
    # use model in the evaluation mode
    model.eval()
    examples = []
    labels = []
    attributions = [] 
    # iterate over test dataloader
    for i, data in tqdm(enumerate(datamodule.test)):
        # read data
        example = data[0].permute(0, 2, 1).float()
        label = data[1]
        # generate predictions
        if label_list[label] != activity:
            continue
        out = model(example)
        pred = int(torch.argmax(out))
        if pred != label:
            continue 
        # generate attribution (explanation)
        num_channels = example.shape[1]
        
        curr_attrs = {}
        for name, xai_model in xai_models.items():
            attribution = xai_model.attribute(example, label)
            attribution = _adjust_input(attribution)
            if attribution.shape[0] == 1:
                attribution = np.tile(attribution, (num_channels, 1))
            curr_attrs[name] = attribution
            
        
        examples.append(_adjust_input(example))
        labels.append(label)
        attributions.append(curr_attrs)
        # save first (max_capacity) correctly predicted examples
        if max_capacity is not None:
            if len(examples) >= max_capacity:
                break
    return examples, labels, attributions


def produce_all_preds_and_attrs(datamodule, model, xai_model, batch_size):
    model.eval()
    examples = []
    labels = []
    preds = []
    attributions = [] 
    # iterate over test dataloader
    for i, data in tqdm(enumerate(datamodule.test)):
        # read data
        example_batch = data[0].permute(0, 2, 1).float()
        label_batch = data[1]
        # generate predictions
        out = model(example_batch)
        pred_batch = torch.argmax(out, dim=1)

        # generate attributions (explanation)
        attribution_batch = xai_model.attribute(example_batch, label_batch)
        
        examples.append(example_batch.detach().numpy())
        attributions.append(attribution_batch.detach().numpy())
        labels.append(label_batch.detach().numpy())
        preds.append(pred_batch.detach().numpy())

    examples = np.vstack(examples)
    attributions = np.vstack(attributions)
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    return examples, labels, preds, attributions


def plot_attribution_heatmap(attribution, activity='', xai_method='', yticks=None):
    sns.heatmap(attribution, yticklabels=yticks)
    if activity != '' and xai_method != '':
        plt.title('{}: {}'.format(xai_method, activity))
    plt.show()


def plot_ts(ts, figsize=(8, 10), subtitles=None):
    num_channels = ts.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=figsize) 
    for i in range(num_channels):
        ax = axes[i]
        sns.tsplot(ts[i], ax=ax)
        if subtitles is not None:
            axes[i].set_title(subtitles[i])
    fig.tight_layout()
    plt.show()


def _adjust_input(inp):
    if len(inp.shape) == 3:
        inp = inp.squeeze(0)
    if torch.is_tensor(inp):
        inp = inp.detach().numpy()
    return inp


def _normalize_attr_ts(values, percentile):
    attr_combined = np.abs(values)
    threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - percentile)
    return _normalize_scale(attr_combined, threshold)


def _normalize_scale(attr, scale_factor):
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values, percentile):
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def plot_attributions_ts(
    example, 
    attribution, 
    activity='',
    xai_method='',
    fig_size=(8, 10),
    cmap=None,
    outlier_perc=1,
    subtitles=None,
    chart='heatmap_per_channel',
    save_path=None
):
    
    example = _adjust_input(example)
    attribution = _adjust_input(attribution)
    
    num_channels = attribution.shape[0]
    ts_length = attribution.shape[1]
    
    num_subplots = num_channels
    x_values = np.arange(ts_length)
    
    norm_attr = _normalize_attr_ts(attribution, outlier_perc)

    default_cmap = cmap if cmap is not None else 'Blues'
    vmin, vmax = 0, 1

    cmap = cm.get_cmap(default_cmap)
    cm_norm = colors.Normalize(vmin, vmax)

    if 'per_channel' in chart:
        plt_fig, plt_axis = plt.subplots(
            figsize=fig_size, nrows=num_subplots, sharex=True
        )

        if chart == 'heatmap_per_channel':
            half_col_width = (x_values[1] - x_values[0]) / 2.0

            for chan in range(num_channels):
                plt_axis[chan].plot(x_values, example[chan,:], 'k')
                for icol, col_center in enumerate(x_values):
                    left = max(0, col_center - half_col_width)
                    right = min(col_center + half_col_width, ts_length)
                    plt_axis[chan].axvspan(
                        xmin=left,
                        xmax=right,
                        facecolor=(cmap(cm_norm(norm_attr[chan][icol]))),
                        edgecolor=None,
                        alpha=0.7,
                    )
                if subtitles is not None:
                    plt_axis[chan].set_title(subtitles[chan])
            if activity != '' and xai_method != '':
                plt.suptitle('{}: {}'.format(xai_method, activity))
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches = "tight")
            plt.show()
        
        elif chart == 'scatter_per_channel':
            # TODO: add line plot and annotations
            for chan in range(num_channels):
                plt_axis[chan].scatter(
                    x_values, 
                    example[chan, :], 
                    c=norm_attr[chan, :], 
                    cmap=cmap,
                    )
