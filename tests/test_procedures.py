import pytest
from high_dimensional_sampling import procedures as proc


class TmpProcedureCorrect(proc.Procedure):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        return None

    def is_finished(self):
        return None

    def check_testfunction(self, function):
        return None

    def reset(self):
        return None


class TmpProcedureWrong1(proc.Procedure):
    def __call__(self, function):
        return None

    def is_finished(self):
        return None

    def check_testfunction(self, function):
        return None

    def reset(self):
        return None


class TmpProcedureWrong2(proc.Procedure):
    def __init__(self):
        self.store_parameters = []

    def is_finished(self):
        return None

    def check_testfunction(self, function):
        return None

    def reset(self):
        return None


class TmpProcedureWrong3(proc.Procedure):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        return None

    def check_testfunction(self, function):
        return None

    def reset(self):
        return None


class TmpProcedureWrong4(proc.Procedure):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        return None

    def is_finished(self):
        return None

    def reset(self):
        return None


class TmpProcedureWrong5(proc.Procedure):
    def __init__(self):
        self.store_parameters = []

    def __call__(self, function):
        return None

    def is_finished(self):
        return None

    def check_testfunction(self, function):
        return None


def test_procedure():
    with pytest.raises(TypeError):
        _ = proc.Procedure()  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpProcedureWrong1()  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpProcedureWrong2()  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpProcedureWrong3()  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpProcedureWrong4()  # pylint: disable=E0110
    with pytest.raises(TypeError):
        _ = TmpProcedureWrong5()  # pylint: disable=E0110
    _ = TmpProcedureCorrect()
