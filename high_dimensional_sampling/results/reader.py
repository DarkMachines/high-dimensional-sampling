import os
import pandas as pd
import yaml


def make_dataframe(result_paths):
    """
    Reads the results of all provided experiments and stores them in a pandas
    DataFrame.

    Internally this function uses the Result class implemented in this
    submodule, but under normal circumstances the user should not have to
    deal will this class directly.

    Args:
        result_paths: Dictionary. Keys indicate the name for the experiment,
            the values indicate the path to the output folder for said
            experiment.

    Returns:
        Pandas DataFrame containing the results of all experiments.
    """
    # Read dictionary result_paths into ResultFolders
    experiments = {}
    for k in result_paths:
        experiments[k] = Result(result_paths[k])
    # Read data from every experiment
    alldata = None
    for i in experiments:
        subdf = experiments[i].get_results()
        subdf['experiment'] = i
        if alldata is None:
            alldata = subdf
        else:
            alldata = alldata.append(subdf)
    return alldata


class Result:
    """
    Class with the functionality to read results from the output folder of an
    experiment and to store its results in a pandas DataFrame.

    Under normal circumstances the user should not have to deal with this class
    directly. Instead, the `make_dataframe` function should be used.

    Args:
        path: Path to the folder containing the results.
    """

    def __init__(self, path):
        self.path = None
        self.connect(path)

    def connect(self, path):
        """
        Connect the Results object to a specific folder.

        Args:
            path: Path to the folder containing the results

        Raises:
            Exception: Path provided to the Result does not seem to be a valid
                experiment folder (benchmarks.yaml is missing).
        """
        if not os.path.exists(path):
            raise Exception("Path '{}' not found.".format(path))
        if path[-1] is not os.sep:
            path = path + os.sep
        if not os.path.exists(path + 'benchmarks.yaml'):
            raise Exception(
                "Path provided to Result does not seem to be a valid"
                "experiment  folder (benchmarks.yaml is missing).")
        self.path = path

    def get_results(self):
        """
        Read the results from the folder to which the Results object is
        connected and return them.

        Returns:
            Pandas DataFrame containing the found results. Results will be
            ordered by function and run number.
        """
        results = None
        # Loop over directories in results folder
        directories = [
            obj for obj in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, obj))
        ]
        for d in directories:
            # Get function name and run number
            function_name, run_number = self.get_function_information(d)
            # Get metrics from experiment
            yaml_contents = self.read_experiment_yaml(d)
            metrics = self.extract_result_information(yaml_contents)
            # Get clean procedure time from procedurecalls.csv
            procedure_calls = self.read_procedure_calls(d)
            time_clean = self.extract_procedure_information(procedure_calls)
            # Create data frame is necessary
            if results is None:
                # Construct keys for columns
                default_keys = ['function', 'run_number']
                metric_keys = [k for k in metrics.keys()]
                keys = default_keys + metric_keys
                # Create dataframe
                results = pd.DataFrame(columns=keys)
            # Fill data frame with values
            results = results.append(
                {
                    'function': function_name,
                    'run_number': run_number,
                    'time_clean': time_clean,
                    **metrics
                },
                ignore_index=True)
        # Order by function_name and then by run_number
        return results.sort_values(['function', 'run_number'])

    def get_function_information(self, folder_name):
        """
        Extract the function name and the run number from the name of a folder
        in the experiment output folder.

        Args:
            folder_name: Name of the folder from which the function name and
                run number should be extracted.

        Returns:
            Tuple containing the function name [0] and run number [1].
        """
        parts = folder_name.split('_')
        run_number = 0
        function_name = '_'.join(parts[:-1])
        if len(parts) > 1:
            run_number = int(parts[-1])
        return function_name, run_number

    def read_procedure_calls(self, folder_name):
        """
        Read the procedurecalls.csv file from a folder and return its contents
        as a Pandas DataFrame.

        Args:
            folder_name: Name of the folder within the experiment output folder
                from which the procedurecalls.csv file should be read.

        Returns:
            Pandas DataFrame containing the content of the procedurecalls.csv
            file.
        """
        path = self.path + os.sep + folder_name
        return pd.read_csv(path + os.sep + 'procedurecalls.csv')

    def extract_procedure_information(self, procedure_calls):
        """
        Get the total time spend on procedure calls based on the DataFrame
        created from the procedurecalls.csv

        Args:
            procedure_calls: Pandas DataFrame of the procedurecalls.csv file.
                This object can be created with the `read_procedure_calls`
                method of this object.

        Returns:
            Total time speld on procedure calls (in seconds).
        """
        return procedure_calls['dt'].sum()

    def read_experiment_yaml(self, folder_name):
        """
        Read the contents of the `experiment.yaml` file from a specific
        folder within the experiment output folder.

        Args:
            folder_name: Name of the folder from which the experiment.yaml file
                should be read.

        Returns:
            Dictionary containing the contents of the experiment.yaml file.
        """
        path = self.path + os.sep + folder_name
        with open(path + os.sep + 'experiment.yaml', 'r') as stream:
            yaml_contents = yaml.safe_load(stream)
        return yaml_contents

    def extract_result_information(self, yaml_contents):
        """
        Get the relevant experimental result information out of a yaml
        dictionary.

        Returned information will be in the form of a dictionary containing
        the type of the experiment, the name of the run procedure and every
        metric in the `results` section of the experiment.yaml file.

        Args:
            yaml_contents: Dictionary with the contents of a experiment.yaml
                file. This dictionary can be created with the
                `read_experiment_yaml` method of this class.

        Returns:
            Dictionary with relevant result information.
        """
        metrics = yaml_contents['results']
        if 'best_value' in metrics:
            metrics['best_value'] = metrics['best_value'][0]
        experiment_type = yaml_contents['experiment']['type']
        procedure_name = yaml_contents['procedure']['name']
        return {
            'experiment_type': experiment_type,
            'procedure_name': procedure_name,
            **metrics
        }
