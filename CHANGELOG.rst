###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

[Unreleased]
************

Added
-----
* A `get_ranges` method in the TestFunction class. It returns the ranges of the
  test function, taking a leeway parameter epsilon provided to the method into
  account: all minima will be raised by epsilon, all maxima will be reduced by
  it.
* The TestFunction `__call__` and `check_ranges` methods now also implement the
  epsilon argument (see first addition above).

Fixed
-----
* Ackley and Easom test functions returned data in an incorrect shape. This has
  been corrected
