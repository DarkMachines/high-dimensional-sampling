"""
This script shows how you can collect experimental results and easily create
some easy plots and tables with them. In order for this script to work, either
random_optimisation.py or rejection_sampling.py should have been run before.
"""
from high_dimensional_sampling import results

# Create a dataframe containing all data
# When multiple experiments are run (i.e. different folders), these can be
# added to the dictionary as new entries. Multiple runs of the same experiment
# should not be included here, as these will be automatically recognised by
# the results module as replication runs of the same experiment.
df = results.make_dataframe({'simple': './hds'})

# Create tables
content, row_labels = results.tabulate_result(df,
                                              'time',
                                              'simple',
                                              path='table.tex')
content, row_labels, col_labels = results.tabulate_all_aggregated(
    df, 'time', 'mean', path='table2.csv')

# Create boxplots
results.boxplot_experiment(df,
                           'time',
                           'simple',
                           logarithmic=True,
                           path='boxplot_experiment.png',
                           figsize=(10, 5),
                           show=False)
results.boxplot_function(df,
                         'time',
                         'ackley',
                         path='boxplot_function.png',
                         show=False)

# Crate histograms
results.histogram_experiment(df,
                             'time',
                             'simple',
                             aggregate='min',
                             path='histogram_experiment.png',
                             show=False)
results.histogram_function(df,
                           'time',
                           'beale',
                           aggregate='max',
                           path='histogram_function.png',
                           show=False)
