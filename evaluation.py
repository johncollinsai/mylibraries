import traceback
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.utils.multiclass import unique_labels
import plotly.figure_factory as ff
from matplotlib.colors import LinearSegmentedColormap

def print_metric_results(accuracy, auc, cm):
    """Print accuracy, auc and cm with descriptions.

    Args:
        accuracy (float): Accuracy
        auc (float): AUC
        cm (numpy array): confusion matrix

    Return:
        None

    """
    print(f'Accuracy: {accuracy:.3f}')
    print(f'AUC: {auc:.3f}')
    print(f'Confusion Matrix:')
    print(cm)


def calculate_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=True)
    # method to determine the best threshold
    # now using sum of sensitivity and specificity
    score = tpr + (1 - fpr)
    best_threshold_id = np.argmax(score)
    best_threshold = thresholds[best_threshold_id]
    best_fpr = fpr[best_threshold_id]
    best_tpr = tpr[best_threshold_id]

    return best_fpr, best_tpr, best_threshold


def construct_plotly_graph(fpr1, tpr1, thresholds1, best_fpr1, best_tpr1, auc1,
                           fpr2, tpr2, thresholds2, best_fpr2, best_tpr2, auc2):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Train ROC curve, AUC = {auc1:.4f}', f'Val/Test ROC curve, AUC = {auc2:.4f}'),
    )

    # the first plot (train dataset)
    fig.add_trace(
        go.Scatter(
            name='',
            x=fpr1, y=tpr1,
            fill='tozeroy',
            text=thresholds1,
            hovertemplate='Threshold = %{text:.6f}',
        ),
        row=1, col=1,
    )

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=1
    )

    fig.add_annotation(
        x=best_fpr1,
        y=best_tpr1,
        text=f"Best threshold <br>({best_fpr1:.4f}, {best_tpr1:.4f})",
        row=1, col=1
    )

    # the second plot (test dataset)
    fig.add_trace(
        go.Scatter(
            name='',
            x=fpr2, y=tpr2,
            fill='tozeroy',
            text=thresholds2,
            hovertemplate='Threshold = %{text:.6f}',
        ),
        row=1, col=2,
    )

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=2
    )

    fig.add_annotation(
        x=best_fpr2,
        y=best_tpr2,
        text=f"Best threshold <br>({best_fpr2:.4f}, {best_tpr2:.4f})",
        row=1, col=2
    )

    # Final plot adjustments
    fig.update_layout(
        width=1000,
        height=550,
        showlegend=False,
    )
    fig.update_xaxes(title_text='False Positive Rate', range=[0, 1])
    fig.update_yaxes(title_text='True Positive Rate', range=[0, 1])

    fig.show()

    return fig


def construct_graph(fpr, tpr, best_fpr, best_tpr, auc, ax=None):
    ax = ax or plt.gca()
    ax.plot(fpr, tpr),
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.plot(best_fpr, best_tpr, marker='o', color='black')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curve, AUC = {auc:.4f}')
    return ax


def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob):
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''

    # Train stats
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label=True)
    best_fpr_train, best_tpr_train, best_threshold = calculate_best_threshold(y_train_true, y_train_prob)
    auc_train = roc_auc_score(y_train_true, y_train_prob)
    y_train_pred = y_train_prob > best_threshold

    print('Train results')
    print_metric_results(
        accuracy_score(y_train_true, y_train_pred),
        auc_train,
        confusion_matrix(y_train_true, y_train_pred)
    )

    # Test stats
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_true, y_test_prob, pos_label=True)
    auc_test = roc_auc_score(y_test_true, y_test_prob)
    y_test_pred = y_test_prob > best_threshold
    cm_test = confusion_matrix(y_test_true, y_test_pred)

    print('Valid/Test results')
    print_metric_results(
        accuracy_score(y_test_true, y_test_pred),
        auc_test,
        cm_test
    )

    best_tpr_test = recall_score(y_test_true, y_test_pred)
    best_fpr_test = float(cm_test[0][1]) / (cm_test[0][0] + cm_test[0][1])

    # try to plot using plotly first, if not successful, use matplolib
    try:
        fig = construct_plotly_graph(
            fpr_train, tpr_train, thresholds_train, best_fpr_train, best_tpr_train, auc_train,
            fpr_test, tpr_test, thresholds_test, best_fpr_test, best_tpr_test, auc_test
        )
    except Exception as e:
        traceback.print_exc()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1 = construct_graph(fpr_train, tpr_train, best_fpr_train, best_tpr_train, auc_train, ax=ax1)
        ax2 = construct_graph(fpr_test, tpr_test, best_fpr_test, best_tpr_test, auc_test, ax=ax2)
        plt.show()

    print(f'Best Threshold: {best_threshold}')

    return best_threshold


def plot_confusion_matrix(y_true, y_pred,
                          mode='plotly',
                          normalize=False,
                          title=None,
                          show=True,
                          color=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    assert mode in ['matplotlib', 'plotly']

    if not title:
        title = 'Confusion matrix'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    if mode == 'plotly':
        classes = ['<br>'.join(wrap(l, 20)) for l in classes]
        if not color:
            color = 'Reds'
        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in cm]

        # set up figure
        fig = ff.create_annotated_heatmap(
            cm, x=classes, y=classes, annotation_text=z_text, colorscale=color
        )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="grey", size=16),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted Label",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="grey", size=16),
                                x=-0.45,
                                y=0.5,
                                showarrow=False,
                                text="Actual Label",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # update x and y ticks fonts
        fig.update_xaxes(
            tickfont=dict(color='grey', size=16)
        )
        fig.update_yaxes(tickfont=dict(color='grey', size=16))

        # adjust margins to make room for yaxis title
        fig.update_layout(
            title_text=title,
            title_font_color="grey",
            font={'size': 16},
            width=800,
            height=680,
            margin=dict(t=150, l=250)
        )

        # reverse axis
        fig['layout']['yaxis']['autorange'] = "reversed"
        if show:
            fig.show()
        return fig

    elif mode == 'matplotlib':
        classes = ['\n'.join(wrap(l, 20)) for l in classes]
        if not color:
            color = LinearSegmentedColormap.from_list('name', ['white', 'red'])
        fig, ax = plt.subplots(figsize=(15.5, 12))
        im = ax.imshow(cm, interpolation='nearest', cmap=color)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=16)

        ax.set_title(title, fontsize=16, loc='center', fontweight='bold', color='grey')

        # We want to show all ticks...
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(classes, {'fontsize': 16, 'color': 'grey'}, rotation=0)
        ax.set_yticklabels(classes, {'fontsize': 16, 'color': 'grey'})
        ax.set_ylabel('True label', fontsize=16, color='grey')
        ax.set_xlabel('Predicted label', fontsize=16, color='grey')

        #     ax.xaxis.tick_top()

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16)
        fig.tight_layout()
        #     plt.savefig('confusion_matrix.png', dpi=500)
        if show:
            plt.show()
        return ax