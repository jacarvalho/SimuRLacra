def get_class_name(obj) -> str:
    """
    Get an arbitrary objects name.

    :param obj: any object
    :return: name of the class of the given object
    """
    return obj.__class__.__name__


def issequence(obj) -> bool:
    """
    Check if an object is a sequence / an iterable.

    .. note::
        Using against `isinstance(obj, collections.Sequence)` yields `False` for some types like `set` and `dict`.

    :param obj: any object
    :return: flag if the object is a sequence / an iterable
    """
    return hasattr(type(obj), '__iter__') and hasattr(type(obj), '__len__')
