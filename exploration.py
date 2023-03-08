import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from pandas_profiling import ProfileReport

def make_profile_report(df, save_dir):
    """Save the profile report of a dataframe in save_dir

    Args:
        df (DataFrame): The dataframe you want to save profile on
        save_dir (str or Path): The filename (ended with html) e.g. C:\brian\df_profile.html

    Return:
        None
    """

    if save_dir:
        save_dir = Path(save_dir)

    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(save_dir)
    print(f'Profile report saved as {save_dir}')



def describe_features(_series, logy=False, bins=20, figsize=(13, 7), **kwargs):
    """A method that will print summary statistics and plot box plot (Outliers = 1.5 * IQR) & histogram.

    Args:
        _series (Series): Data series to be plotted
        logx (bool): Log x axis for histogram if True (default False)
        logy (bool): Log y axis for histogram if True (default False)
        bins (float): Bin count for histogram
        figsize (float, float): figuresize of the histogram
        **kwargs: other parameters to pass to matplotlib.pyplot.figure()

    """
    sns.set_style("whitegrid", {
        'grid.linestyle': '--'
    })
    print('Range:', min(_series), max(_series))
    print('Mean:', _series.mean())
    print('Median:', _series.median())
    print('STD:', _series.std())
    print('IQR:', np.percentile(_series, 75) - np.percentile(_series, 25))
    print('Counts:', len(_series))
    nulls = _series.isnull().sum()
    print(f'Null Counts: {nulls} ({np.round(nulls / len(_series), 2)}%)')
    zeros = (_series == 0).astype(int).sum()
    print(f'Zero Counts: {zeros} ({np.round(zeros / len(_series), 2)}%)')
    fig = plt.figure(figsize=figsize, **kwargs)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
    ax1 = fig.add_subplot(gs[0])
    _series.plot(kind='box')
    ax2 = fig.add_subplot(gs[1])
    counts, bins, _ = plt.hist(_series.dropna(), log=logy, bins=bins, label=_series.name)
    # _series.plot.hist(logx=logx, logy=logy, bins=bins)
    ax2.set_xlabel(_series.name)
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    values_df = pd.DataFrame(columns=['counts', 'bins'])
    values_df['counts'] = counts
    values_df['bins'] = [(bins[ii] + bins[ii + 1]) / 2 for ii in np.arange(len(bins) - 1)]

    return values_df


def plot_feature_distribution(df, features, hue=''):
    """Density plots of the columns with hues

    Args:
        df (DataFrame): the data you want to plot
        features (list): list of column name that you want to plot distribution
        hue (str): column name of the column on labels you want to segregate data on

    """
    i = 0
    dim = math.ceil(len(features) / 3.)

    if hue:
        labels = list(df[hue].unique())
    else:
        labels = []

    plt.figure()
    fig, ax = plt.subplots(dim, 3, figsize=(18, 6 * dim))

    for feature in features:
        i += 1
        plt.subplot(dim, 3, i)
        try:
            if hue:
                for label in labels:
                    sns.kdeplot(df[df[hue] == label][feature], bw=0.5, label=label)
            else:
                sns.kdeplot(df[feature], bw=0.5, label=label)
            # sns.kdeplot(df1[feature], bw=0.5,label=label1)
            # sns.kdeplot(df2[feature], bw=0.5,label=label2)
            # sns.kdeplot(df3[feature], bw=0.5, label=label3)
        except Exception as e:
            pass
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=12)
        plt.tick_params(axis='y', which='major', labelsize=12)
    plt.show()


if __name__ == '__main__':
    pass
