#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the high_dimensional_sampling module.
"""
import pytest

from high_dimensional_sampling import high_dimensional_sampling


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_high_dimensional_sampling(an_object):
    assert an_object == {}
