import numpy as np
from abc import ABC, abstractmethod

from pyrado.logger.step import StepLogger


class BaseMetric(ABC):
    """ Base class for all metrics in Pyrado """

    def __init__(self, metric_fcn: callable, logger: StepLogger = None):
        """
        Constructor

        :param metric_fcn: function used to compute the metric
        :param logger: automatically log all computed values if not `None`
        """
        self._metric_fcn = metric_fcn
        self._logger = logger

    @abstractmethod
    @property
    def curr_value(self):
        return NotImplementedError

    @property
    def metric_fcn(self) -> callable:
        return self._metric_fcn

    def reset(self, *args, **kwargs):
        raise NotImplementedError


class SuccessMetric(BaseMetric):
    """ """

    def __init__(self, metric_fcn: callable, logger: StepLogger = None):
        """
        Constructor

        :param metric_fcn: function used to determine if the
        :param logger: automatically log all computed values if not `None`
        """
        super().__init__(metric_fcn, logger)

    @property
    def curr_value(self):
        return NotImplementedError

# succ = np.empty(len(self._metrics))
#
# for i in range(len(succ)):
#     succ[i] = self._metrics[i](data)
