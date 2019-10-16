import pytest
import re
import os
import time
import numpy as np
from high_dimensional_sampling import utils


def test_get_time():
    # Check if time is equal to or larger than a time calculated in this func
    t = int(round(time.time() * 1000.0))/1000.0
    assert t <= utils.get_time()

def test_datetime():
    pattern = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}.\d{3,6})$")
    assert pattern.match(utils.get_datetime())

def test_create_unique_folder():
    # Get current folder
    this_folder = os.getcwd()
    if this_folder[-1] == os.sep:
        this_folder = this_folder[:-1]
    # Check if created folders indeed have unique names
    name1 = utils.create_unique_folder(this_folder, "tmp")
    name2 = utils.create_unique_folder(this_folder, "tmp")
    assert name1 is not name2
    # Check if uniqueness is independent of closing seperator
    name3 = utils.create_unique_folder(this_folder+os.sep, "tmp")
    assert name1 is not name3
    assert name2 is not name3
    # Check if removing the folder indeeds opens up the folder name for a next
    # create_unique_folder call
    os.rmdir(name1)
    name4 = utils.create_unique_folder(this_folder+"/", "tmp")
    assert name1 == name4
    # Remove folders that were created
    os.rmdir(name2)
    os.rmdir(name3)
    os.rmdir(name4)

def test_benchmarks():
    # Check if benchmarks return a time larger than 0
    assert utils.benchmark_matrix_inverse() > 0
    assert utils.benchmark_sha_hashing() > 0

def test_require_extension():
    # Create temporary test file
    this_folder = os.getcwd()
    if this_folder[-1] != os.sep:
        this_folder = this_folder + os.sep
    tmp_file = this_folder + 'tmpfile.txt'
    # Perform checks on test test file extension
    assert utils.require_extension(tmp_file, ['txt'])
    assert utils.require_extension(tmp_file, ['txt', 'jpeg'])
    assert utils.require_extension(tmp_file, ['TXT'])
    assert utils.require_extension(tmp_file, ['tXt'])
    assert utils.require_extension(tmp_file, ['txt', 'TXT'])
    assert utils.require_extension(tmp_file, ['TXT', 'jPeG'])
    with pytest.raises(Exception):
        utils.require_extension(tmp_file, ['jpeg'])
    with pytest.raises(Exception):
        utils.require_extension(tmp_file, ['JPEG'])
