import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def boxplot_experiment(df,
                       metric,
                       experiment_name,
                       logarithmic=False,
                       figsize=None,
                       path=None,
                       show=False):
    """
    Create a boxplot showing the values for a given metric for all functions in
    a specific experiment.

    Data for plotting should be provided as a Pandas DataFrame created with the
    `results.make_dataframe()` function.

    The plot can be saved to a file by setting the `path` argument to a valid
    file path. The plot can also be shown in a new window by setting the `show`
    method to `True`. If `path` is `None` however, the plot will be shown
    regardless of the `show` setting.

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        experiment_name: Name of the experiment to plot the results of. Name
            should match the name as it was given in the creation of the
            dataframe `df`.
        logarithmic: Boolean indicating if the y-axis of the plot should be
            logarithmic (default is `False`).
        figsize: Tuple containing the size of the figure in inches. If set to
            `None`, size is automatically determined. Default is `None`.
        path: Path to which the figure should be saved. If set to `None`, the
            figure will not be saved, but shown instead. Default is `None`.
        show: Boolean indicating if the created figure should be shown in
            a new window. Default is `False`, but this will be overwritten to
            `True` if `path` is set to `None`.
    """
    # Check if experiment_name is actually in df (and do same for metric)
    if experiment_name not in df['experiment'].unique():
        raise Exception(
            "Experiment '{}' not found in DataFrame.".format(experiment_name))
    if metric not in df.keys():
        raise Exception("Metric '{}' not found in DataFrame.".format(metric))
    # Create frame with only relevant information
    plot_df = df[df.experiment == experiment_name]
    # Get row and column labels
    function_labels = plot_df['function'].unique()
    function_labels.sort()
    # Make plot
    _, ax = plt.subplots(figsize=figsize)
    p = sns.boxplot(ax=ax, x="function", y=metric, data=plot_df)
    p.set_xticklabels(labels=function_labels, rotation=90)
    plt.title("{} for experiment '{}'".format(metric.title(), experiment_name))
    plt.tight_layout()
    if logarithmic:
        plt.yscale('log')
    # Show or store plot
    if path is not None: # pragma: no cover
        plt.savefig(path) # pragma: no cover
    if path is None or show: # pragma: no cover
        plt.show()  # pragma: no cover


def boxplot_function(df,
                     metric,
                     function_name,
                     logarithmic=False,
                     figsize=None,
                     path=None,
                     show=False):
    """
    Create a boxplot showing the values for a given metric for all experiments
    that ran a specific function.

    Data for plotting should be provided as a Pandas DataFrame created with the
    `results.make_dataframe()` function.

    The plot can be saved to a file by setting the `path` argument to a valid
    file path. The plot can also be shown in a new window by setting the `show`
    method to `True`. If `path` is `None` however, the plot will be shown
    regardless of the `show` setting.

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        function_name: Name of the function to plot the results of. Name
            should match the names of the functions in the dataframe `df`.
        logarithmic: Boolean indicating if the y-axis of the plot should be
            logarithmic (default is `False`).
        figsize: Tuple containing the size of the figure in inches. If set to
            `None`, size is automatically determined. Default is `None`.
        path: Path to which the figure should be saved. If set to `None`, the
            figure will not be saved, but shown instead. Default is `None`.
        show: Boolean indicating if the created figure should be shown in
            a new window. Default is `False`, but this will be overwritten to
            `True` if `path` is set to `None`.
    """
    # Check if function_name is actually in df (and do same for metric)
    if function_name not in df['function'].unique():
        raise Exception(
            "Function '{}' not found in DataFrame.".format(function_name))
    if metric not in df.keys():
        raise Exception("Metric '{}' not found in DataFrame.".format(metric))
    # Create frame with only relevant information
    plot_df = df[['experiment', 'function', 'run_number', metric]]
    plot_df = plot_df[df.function == function_name]
    # Get row and column labels
    experiment_names = plot_df['experiment'].unique()
    experiment_names.sort()
    # Make plot
    _, ax = plt.subplots(figsize=figsize)
    p = sns.boxplot(ax=ax, x="experiment", y=metric, data=plot_df)
    p.set_xticklabels(labels=experiment_names, rotation=90)
    plt.title("{} for function '{}'".format(metric.title(), function_name))
    plt.tight_layout()
    if logarithmic:
        plt.yscale('log')
    # Show or store plot
    if path is not None: # pragma: no cover
        plt.savefig(path) # pragma: no cover
    if path is None or show: # pragma: no cover
        plt.show()  # pragma: no cover


def histogram_experiment(df,
                         metric,
                         experiment_name,
                         aggregate='mean',
                         logarithmic=False,
                         figsize=None,
                         path=None,
                         show=False):
    """
    Create a histogram showing the aggregated values for a given metric for all
    functions in a specific experiment.

    Data for plotting should be provided as a Pandas DataFrame created with the
    `results.make_dataframe()` function.

    The values used in creating the histogram are aggregated from all runs for
    a specific function. Thorugh the `aggregate` argument the user can set how
    to aggregate these values. The histogram wil automatically be ordered in
    such a way that the values the bars represent are descending.

    The plot can be saved to a file by setting the `path` argument to a valid
    file path. The plot can also be shown in a new window by setting the `show`
    method to `True`. If `path` is `None` however, the plot will be shown
    regardless of the `show` setting.

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        experiment_name: Name of the experiment to plot the results of. Name
            should match the name as it was given in the creation of the
            dataframe `df`.
        aggregate: String indicating the function to use in order to aggregate
            the metric values. E.g. `'mean'`, `'max'`, `'min`. Default is
            `'mean'`.
        logarithmic: Boolean indicating if the x-axis of the plot should be
            logarithmic (default is `False`).
        figsize: Tuple containing the size of the figure in inches. If set to
            `None`, size is automatically determined. Default is `None`.
        path: Path to which the figure should be saved. If set to `None`, the
            figure will not be saved, but shown instead. Default is `None`.
        show: Boolean indicating if the created figure should be shown in
            a new window. Default is `False`, but this will be overwritten to
            `True` if `path` is set to `None`.
    """
    # Check if experiment_name is actually in df (and do same for metric)
    if experiment_name not in df['experiment'].unique():
        raise Exception(
            "Experiment '{}' not found in DataFrame.".format(experiment_name))
    if metric not in df.keys():
        raise Exception("Metric '{}' not found in DataFrame.".format(metric))
    # Create frame with only relevant information
    plot_df = df[['experiment', 'function', 'run_number', metric]]
    plot_df = plot_df[plot_df.experiment == experiment_name]
    # Aggregate data
    plot_df = plot_df.groupby(['function']).agg({
        metric: aggregate
    }).reset_index()
    plot_df = plot_df.sort_values(metric, ascending=False)
    # Make plot
    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(ax=ax, x=metric, y="function", data=plot_df)
    plt.title("{} {} for experiment '{}'".format(aggregate.title(), metric,
                                                 experiment_name))
    plt.tight_layout()
    if logarithmic:
        plt.xscale('log')
    # Show or store plot
    if path is not None: # pragma: no cover
        plt.savefig(path) # pragma: no cover
    if path is None or show: # pragma: no cover
        plt.show()  # pragma: no cover


def histogram_function(df,
                       metric,
                       function_name,
                       aggregate='mean',
                       logarithmic=False,
                       figsize=None,
                       path=None,
                       show=False):
    """
    Create a histogram showing the aggregated values for a given metric for all
    experiments that ran a specific function.

    Data for plotting should be provided as a Pandas DataFrame created with the
    `results.make_dataframe()` function.

    The values used in creating the histogram are aggregated from all runs for
    a specific function. Thorugh the `aggregate` argument the user can set how
    to aggregate these values. The histogram wil automatically be ordered in
    such a way that the values the bars represent are descending.

    The plot can be saved to a file by setting the `path` argument to a valid
    file path. The plot can also be shown in a new window by setting the `show`
    method to `True`. If `path` is `None` however, the plot will be shown
    regardless of the `show` setting.

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        function_name: Name of the function to plot the results of. Name
            should match the names of the functions in the dataframe `df`.
        aggregate: String indicating the function to use in order to aggregate
            the metric values. E.g. `'mean'`, `'max'`, `'min`. Default is
            `'mean'`.
        logarithmic: Boolean indicating if the x-axis of the plot should be
            logarithmic (default is `False`).
        figsize: Tuple containing the size of the figure in inches. If set to
            `None`, size is automatically determined. Default is `None`.
        path: Path to which the figure should be saved. If set to `None`, the
            figure will not be saved, but shown instead. Default is `None`.
        show: Boolean indicating if the created figure should be shown in
            a new window. Default is `False`, but this will be overwritten to
            `True` if `path` is set to `None`.
    """
    # Check if experiment_name is actually in df (and do same for metric)
    if function_name not in df['function'].unique():
        raise Exception(
            "Function '{}' not found in DataFrame.".format(function_name))
    if metric not in df.keys():
        raise Exception("Metric '{}' not found in DataFrame.".format(metric))
    # Create frame with only relevant information
    plot_df = df[['experiment', 'function', 'run_number', metric]]
    plot_df = plot_df[plot_df.function == function_name]
    plot_df = plot_df.rename(columns={'experiment': 'experiment'})
    # Aggregate data
    plot_df = plot_df.groupby(['experiment']).agg({
        metric: aggregate
    }).reset_index()
    plot_df = plot_df.sort_values(metric, ascending=False)
    # Make plot
    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(ax=ax, x=metric, y="experiment", data=plot_df)
    plt.title("{} {} for experiment '{}'".format(aggregate.title(), metric,
                                                 function_name))
    plt.tight_layout()
    if logarithmic:
        plt.xscale('log')
    # Show or store plot
    if path is not None: # pragma: no cover
        plt.savefig(path) # pragma: no cover
    if path is None or show: # pragma: no cover
        plt.show()  # pragma: no cover
