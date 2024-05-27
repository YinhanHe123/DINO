import abc
class BaseMeasure(abc.ABC):
    def __init__(self,measure_name):
        super().__init__()
        self.measure_name = measure_name
    @abc.abstractmethod
    def get_immunized_nodes(self, A, k):
        pass