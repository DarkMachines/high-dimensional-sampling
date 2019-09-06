import numpy as np
from ..utils import require_extension


def tabulate_result(df, metric, experiment_name, functions=None, path=None):
    """
    Create a table containing the results of a specific experiment for a
    specific metric.

    All functions will get their own row in the table, each experimental run
    will get its own column.

    The table can be outputted to a file. The path to which the table is
    written determines what kind of formatting is used: currently supported
    are paths ending with `.csv` (creating a comma separated table) and `.tex`
    (creating a table that can be used in LaTeX documents).

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        experiment_name: Name of the experiment to tabulate the results of.
            Name should match the name as it was given in the creation of the
            dataframe `df`.
        functions: List of the functions that should be included in the table.
            If set to `None` (default), all functions will be included in the
            table.
        path: Path to which the table should be stored. This path should end
            with either `.csv` or `.tex`. If set to `None`, the table will not
            be formatted and stored.

    Returns:
        content: Numpy `nd.array` containing the contents of the table. The
            array has as shape (nFunctions, maxRuns).
        row_labels: Names of the functions included in the table, ordered 
            in such a way that they match the rows of the `content` array.
    
    Raises:
        Exception: Experiment '?' not found in provided DataFrame.
        Exception: Metric '?' not found in provided DataFrame.
    """
    # Check if experiment_name is actually in df (and do same for metric)
    if experiment_name not in df['experiment'].unique():
        raise Exception(
            "Experiment '{}' not found in provided DataFrame.".format(
                experiment_name))
    if metric not in df.keys():
        raise Exception(
            "Metric '{}' not found in provided DataFrame.".format(metric))
    # Create frame with only relevant information
    table_df = df[df.experiment == experiment_name]
    table_df = table_df[['function', 'run_number', metric]]
    # Get row and column labels
    if functions is None:
        row_labels = table_df['function'].unique()
        row_labels.sort()
    else:
        row_labels = functions
    n_columns = table_df.groupby('function').count().max()['run_number']
    # Fill table content
    content = np.ones((len(row_labels), n_columns)) * np.nan
    for i, row in enumerate(row_labels):
        metrics = table_df[table_df.function == row].sort_values(
            'run_number')[metric]
        for j, met in enumerate(metrics):
            content[i, j] = met
    content = np.round(content, 5)
    content = content.tolist()
    # Store table is requested
    if path is not None:
        # Check if provided path ends with proper extension
        extension = require_extension(path, ['csv', 'tex'])
        # Make table content and store it
        table_content = create_table_string(
            content, row_labels, None, extension,
            "{} for the run experiments (rows) for all runs (columns)".format(
                metric.title()))
        with open(path, 'w') as handle:
            handle.write(table_content)
    return (content, row_labels)


def tabulate_all_aggregated(df,
                            metric,
                            aggregate='mean',
                            experiment_names=None,
                            functions=None,
                            path=None):
    """
    Create a table containing the results of a specific metric (aggregated)
    for all experiments in a provided DataFrame.

    All functions will get their own row in the table, each experimental
    will get its own column.

    The values in the table will be aggregated over all experimental runs.
    Which function is used for this aggregation is configured through the 
    `aggregate` argument of this function.

    The table can be outputted to a file. The path to which the table is
    written determines what kind of formatting is used: currently supported
    are paths ending with `.csv` (creating a comma separated table) and `.tex`
    (creating a table that can be used in LaTeX documents).

    Args:
        df: Pandas DataFrame containing the data of all experiments, created
            with `results.make_dataframe()`.
        metric: Name of the metric to show as it is denoted in the provided
            dataframe `df`.
        aggregate: String indicating the function to use in order to aggregate
            the metric values. E.g. `'mean'`, `'max'`, `'min`. Default is
            `'mean'`.
        experiment_name: Name of the experiment to tabulate the results of.
            Name should match the name as it was given in the creation of the
            dataframe `df`.
        experiment_names: List of the experiments that should be included in
            the table. The names should match the names of the experiments in
            the provided DataFrame `df`. If set to `None` (default), all
            experiments will be included in the table.
        functions: List of the functions that should be included in the table.
            If set to `None` (default), all functions will be included in the
            table.            
        path: Path to which the table should be stored. This path should end
            with either `.csv` or `.tex`. If set to `None`, the table will not
            be formatted and stored.

    Returns:
        content: Numpy `nd.array` containing the contents of the table. The
            array has as shape (nFunctions, maxRuns).
        row_labels: Names of the functions included in the table, ordered 
            in such a way that they match the rows of the `content` array.
    
    Raises:
        Exception: Experiment '?' not found in provided DataFrame.
        Exception: Metric '?' not found in provided DataFrame.
    """
    # Check if experiment_names are actually in df (and do same for metric)
    allowed_experiment_names = df.experiment.unique()
    if experiment_names is not None:
        for experiment_name in experiment_names:
            if experiment_name not in allowed_experiment_names:
                raise Exception(
                    "Experiment '{}' not found in provided DataFrame.".format(
                        experiment_name))
    if metric not in df.keys():
        raise Exception(
            "Metric '{}' not found in provided DataFrame.".format(metric))
    # Create frame with only relevant information
    table_df = df[['experiment', 'function', 'run_number', metric]]
    # Aggregate data
    table_df = table_df.groupby(['function',
                                 'experiment']).agg({metric: aggregate})
    # Get row and column labels
    if functions is None:
        row_labels = table_df.index.levels[0].unique().sort_values().tolist()
    else:
        row_labels = functions
    if experiment_names is None:
        col_labels = table_df.index.levels[1].unique().sort_values().tolist()
    else:
        col_labels = experiment_names
    # Fill table content
    content = np.ones((len(row_labels), len(col_labels))) * np.nan
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            content[i, j] = table_df.loc[row.lower()].loc[col][metric]
    content = np.round(content, 5)
    content = content.tolist()
    # Store table is requested
    if path is not None:
        # Check if provided path ends with proper extension
        extension = require_extension(path, ['csv', 'tex'])
        # Make table content and store it
        content = create_table_string(
            content, row_labels, col_labels, extension,
            "{} {} for the run experiments (columns) on the selected test functions (rows)"
            .format(aggregate.title(), metric))
        with open(path, 'w') as handle:
            handle.write(content)
    return (content, row_labels, col_labels)


def create_table_string(content, row_labels, col_labels, format, caption):
    """
    Create a table following a specific formatting based on provided
    rows, columns and table entries.

    Args:
        content: Numpy `ndarray` of shape `(rows, columns)` containing the
            contents of the table.
        row_labels: List containing the labels that should be at the start of
            each row.
        col_labels: List containing the labels that should be at the top most
            row of the table.
        format: String indicating the formatting of the table. Currently
            supported are `csv` and `tex`.
        caption: Caption that should be used for the table. Currently this 
            option will only be used when creating a `tex` table.
    
    Returns:
        A string containing the table formatted with the rules of the user
        indicated format.
    
    Raises:
        Exception: Format '?' unknown for table creation. Currently supported
            are 'csv' and 'tex'.
    """
    # Create table following correct format
    output = ''
    if format == 'csv':
        # Create .csv file
        if col_labels is not None:
            line = [''] + col_labels
            output += ','.join(line) + "\n"
        for i, row in enumerate(row_labels):
            line = [row] + list(map(str, content[i]))
            output += ','.join(line) + "\n"
    elif format == 'tex':
        # Create .tex table
        output = '\\begin{table}\n\\centering\n\\begin{tabular}'
        output += "{" + ("l|" + "c" * len(content[0])) + "}\n"
        if col_labels is not None:
            line = [''] + col_labels
            output += ' & '.join(line) + '\\\\' + "\n"
            output += '\\hline \n'
        for i, row in enumerate(row_labels):
            line = [row] + list(map(str, content[i]))
            output += ' & '.join(line) + '\\\\' + "\n"
        output += "\\end{tabular}\n"
        output += "\\caption{" + caption + "}\n"
        output += "\\end{table}"
    else:
        raise Exception(
            "Format '{}' unknown for table creation. Currently supported are 'csv' and 'tex'."
            .format(format))
    return output
