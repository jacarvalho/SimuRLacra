import inspect
import os.path as osp


class BaseErr(Exception):
    """ Base class for exceptions in Pyrado """

    @staticmethod
    def retrieve_var_name(var, stack_level=2) -> str:
        """
        Get the name of the given variable by searching through the locals and globals.

        :param var: variable to look for
        :param stack_level: number of frames to go up. 1 is the caller, 2 (default) is the caller's caller.
        :return: name of the first matching variable
        """

        # Helper
        def first_match_name(items):
            # Get the first variable that points to obj
            return next((k for k, v in items if v is var), None)

        # Check locals from stack
        frame = inspect.currentframe()
        for _ in range(stack_level):
            frame = frame.f_back
        name = first_match_name(frame.f_locals.items())

        if name is None:
            # Check fields if method
            m_self = frame.f_locals.get("self", None)
            if m_self:
                # Use generator to avoid errors
                # noinspection PyBroadException
                def list_attrs():
                    for attr in dir(m_self):
                        # try to access attribute
                        try:
                            yield attr, getattr(m_self, attr)
                        except Exception:
                            # Simply skip errors here, we only do this for error reporting
                            pass

                name = first_match_name(list_attrs())

        if name is None:
            # Check globals
            name = first_match_name(globals().items())

        if name is None:
            # General purpose return
            return 'the input'

        return name


class TypeErr(BaseErr):
    """ Class for exceptions raised by passing wrong data types """

    def __init__(self, *,
                 given=None,
                 given_name: str = None,
                 expected_type: [type, list] = None,
                 msg: str = None):
        """
        Constructor

        :param given: input which caused the error
        :param given_name: explicitly pass the name of the variable for the error message
        :param expected_type: type or types that should have been passed
        :param msg: offers possibility to override the error message
        """
        if given is None and msg is None:
            super().__init__("Either specify an input for the error message using the 'given' arg, or set a custom"
                             "message via the 'msg' arg!")
        elif msg is None:
            self.given_name = given_name if given_name is not None else BaseErr.retrieve_var_name(given)
            self.given_type = type(given)
            self.expected_types = expected_type
            # Default error message
            msg = f'Expected the type of {self.given_name} to be'
            if isinstance(expected_type, list):
                for i, t in enumerate(expected_type):
                    if i == 0:
                        msg += ' ' + t.__name__
                    else:
                        msg += ' or ' + t.__name__
            else:
                msg += ' ' + expected_type.__name__
            msg += f' but received {self.given_type.__name__}!'
        # Pass to Python Exception
        super().__init__(msg)


class ValueErr(BaseErr):
    """ Class for exceptions raised by passing wrong values """

    def __init__(self, *,
                 given=None,
                 given_name: str = None,
                 eq_constraint=None,
                 l_constraint=None,
                 le_constraint=None,
                 g_constraint=None,
                 ge_constraint=None,
                 msg: str = None):
        """
        Constructor

        :param given: input which caused the error
        :param given_name: explicitly pass the name of the variable for the error message
        :param eq_constraint: violated equality constraint
        :param l_constraint: violated less than constraint
        :param le_constraint: violated less or equal than constraint
        :param g_constraint: violated greater than constraint
        :param ge_constraint: violated greater or equal than constraint
        :param msg: offers possibility to override the error message
        """
        if given is None and msg is None:
            super().__init__("Either specify an input for the error message using the 'given' arg, or set a custom"
                             "message via the 'msg' arg!")
        if msg is None:
            # If the default error message is used
            assert not (eq_constraint is None and l_constraint is None and le_constraint is None and
                        g_constraint is None and ge_constraint is None), 'Specify at least one constraint!'
        self.given_name = given_name if given_name is not None else BaseErr.retrieve_var_name(given)
        self.given_str = str(given)
        self.eq_constraint_str = str(eq_constraint)
        self.l_constraint_str = str(l_constraint)
        self.le_constraint_str = str(le_constraint)
        self.g_constraint_str = str(g_constraint)
        self.ge_constraint_str = str(ge_constraint)
        if msg is None:
            # Default error message
            msg = f'The value of {self.given_name} should be '
            if eq_constraint is not None:
                msg += f'equal to {self.eq_constraint_str} '
            if l_constraint is not None:
                msg += f'smaller than {self.l_constraint_str} '
            if le_constraint is not None:
                msg += f'smaller or equal than {self.le_constraint_str} '
            if g_constraint is not None:
                msg += f'greater than {self.g_constraint_str} '
            if ge_constraint is not None:
                msg += f'greater or equal than {self.ge_constraint_str} '
            msg += f'but it is {self.given_str}!'
        # Pass to Python Exception
        super().__init__(msg)


class ShapeErr(BaseErr):
    """ Class for exceptions raised by passing wrong values """

    @staticmethod
    def get_shape_and_name(obj, var):
        if isinstance(obj, tuple):
            # Shape passed explicitly
            return obj, "shape"
        shape = getattr(obj, "shape", None)
        if shape is not None:
            return shape, "shape"
        elif hasattr(obj, '__len__'):
            return len(obj), "length"
        raise AttributeError(f"{var} must have either a shape attribute or support len()!")

    def __init__(self, *,
                 given=None,
                 given_name: str = None,
                 expected_match=None,
                 msg: str = None):
        """
        Constructor

        :param given: input which caused the error
        :param expected_match: object which's shape is the one the input should have
        :param given_name: explicitly pass the name of the variable for the error message
        :param msg: offers possibility to override the error message
        """
        if given is None and msg is None:
            super().__init__("Either specify an input for the error message using the 'given' arg, or set a custom"
                             "message via the 'msg' arg!")
        elif msg is None:
            self.given_name = given_name if given_name is not None else BaseErr.retrieve_var_name(given)
            self.given_shape, gsn = ShapeErr.get_shape_and_name(given, "given")
            self.expected_shape, esn = ShapeErr.get_shape_and_name(expected_match, "expected_match")
            self.attributes = (gsn, esn)

            # Default error message
            msg = f'The {self.attributes[0]} of {self.given_name} should match the {self.attributes[1]} ' \
                f'{self.expected_shape} but it is {self.given_shape}!'
        # Pass to Python Exception
        super().__init__(msg)


class PathErr(BaseErr):
    """ Class for exceptions raised by passing wrong paths to folders or files """

    def __init__(self, *, given: str = None, msg: str = None):
        """
        Constructor

        :param given: input which caused the error
        :param msg: offers possibility to override the error message
        """
        if given is None and msg is None:
            super().__init__("Either specify an input for the error message using the 'given' arg, or set a custom"
                             "message via the 'msg' arg!")
        elif msg is None:
            self.is_dir = osp.isdir(given)
            self.is_file = osp.isfile(given)
            # Default error message
            msg = f'The given path {given} '
            if not self.is_dir and not self.is_file:
                msg += 'is neither a directory nor a file!'
            if not self.is_dir and self.is_file:
                msg += 'is not a directory but a file!'
            if not self.is_dir and self.is_file:
                msg += 'is a directory but not a file!'
            # Pass to Python Exception
        super().__init__(msg)
