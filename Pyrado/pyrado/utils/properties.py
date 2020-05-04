from functools import update_wrapper


class Delegate:
    """
    Delegate to a class member object.
    
    Example:
    class Foo:
        def __init__(self, bar):
            self.bar = bar
        baz = Delegate("bar")
    foo = Foo()
    foo.baz <=> foo.bar.baz
    """

    def __init__(self, delegate, attr = None):
        self.delegate = delegate
        self.attr = attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(getattr(instance, self.delegate), self.attr)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.delegate), self.attr, value)

    def __set_name__(self, owner, name):
        # Use outer name for inner name
        if self.attr is None:
            self.attr = name


class ReadOnlyDelegate:
    """
    Delegate to a class member object in a read-only way.
    
    Example:
    class Foo:
        def __init__(self, bar):
            self.bar = bar
        baz = Delegate("bar")
    foo = Foo()
    foo.baz <=> foo.bar.baz
    """

    def __init__(self, delegate, attr = None):
        self.delegate = delegate
        self.attr = attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(getattr(instance, self.delegate), self.attr)

    def __set_name__(self, owner, name):
        # Use outer name for inner name
        if self.attr is None:
            self.attr = name


class cached_property:
    """
    Decorator that turns a function into a cached property.
    When the property is first accessed, the function is called to compute the value. Later calls use the cached value.
    .. note:: Requires a `__dict__` field, so it won't work on named tuples.
    """

    def __init__(self, func):
        self._func = func
        self._name = func.__name__
        update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Create it
        res = self._func(instance)
        # Store in dict, subsequent queries will use that value
        instance.__dict__[self._name] = res
        return res
