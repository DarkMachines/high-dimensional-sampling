# High-Dimensional Sampling Framework

Framework for the high-dimensional sampling challenge of DarkMachines.org. In
this framework sampling algorithms can be implemented, allowing for testing
against test functions and evaluating their performance.

## Installing

After having cloned or downloaded the repository, check your the version of
python that is run when calling python3

```
python3 --version
```

If this version is >=3.5, the package can be installed by running the following
command from the project folder:

```
pip3 install .
```

If you are running an older version of python than version 3.5 by default,
check if a newer version of python is installed on your machine. If not,
install this new version. Otherwise, run the pip3 install command from the
python command itself. For instance, for python version 3.7, you need to run

```
python3.7 -m pip install .
```

Both these installation methods install the package in the main python
enrivonment. If you don't have the rights to install in the main python package
directory, you can alternatively install the package in your user environment
by using the --user argument. For instance:

```
pip3 install --user .
```

## Examples

Examples can be found in the [examples](examples) folder.

## Contributing

The High Dimensional Sampling project is part of the Dark Machines research
collective. If you are interested in contributing to the project, please visit
[the Dark Machines website](http://www.darkmachines.org/) for more information
and the email addresses of the contact persons for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) for
more details.