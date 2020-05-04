from contextlib import contextmanager


class IterationTracker:
    """ Track the current iteration/step number on multiple levels (for meta-algorithms) """

    def __init__(self):
        """ Constructor """
        self._iter_stack = []

    def push(self, label: str, num: int):
        """
        Push an iteration scope.

        :param label: scope label
        :param num: iteration index
        """
        self._iter_stack.append((label, num))

    def pop(self) -> tuple:
        """ Remove the last iteration scope. """
        self._iter_stack.pop()

    @contextmanager
    def iteration(self, label: str, num: int):
        """
        Context with active iteration scope.

        :param label: scope label
        :param num: iteration index
        """
        self.push(label, num)
        yield
        self.pop()

    def get(self, label: str):
        """
        Get the iteration number for a labeled scope.

        :param label: scope label
        :return: iteration index
        """
        for l, n in self._iter_stack:
            if l == label:
                return n
        else:
            return None

    def __iter__(self):
        yield from self._iter_stack

    def format(self, scope_sep='-', label_num_sep='_'):
        """
        Format the current iteration stack into a string. Two parts can be customized:

        :param scope_sep: string separating the label and the number
        :param label_num_sep: string separating each label/number pair
        :return: string with custom separators
        """
        return scope_sep.join(l + label_num_sep + n for l, n in self._iter_stack)

    def __str__(self):
        """ Get an information string. """
        return self.format()
