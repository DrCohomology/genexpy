import re

from genexpy.kernels import utils

def find_subclass(superclass, subclass_name: str):
    # Find the subclass with the given name
    for subclass in utils.all_subclasses(superclass):
        if subclass.__name__ == subclass_name:
            return subclass
    else:
        raise ValueError(f"No subclass named '{subclass_name}' found for {superclass.__name__}")


class Kernel:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def _find_subclass(cls, subclass_name: str):
        return find_subclass(cls, subclass_name)

    @classmethod
    def from_string(cls, s: str):
        """
        Parse a string like 'MallowsKernel(nu="auto")' and return
        an instance of the corresponding subclass.
        The string must be a valid kernel with already initizliaed parameters (for instance, in the case of
            kernels.rankings.MallowsKernel, nu must be a number).
            - Working: Kernel.from_string("MallowsKernel(nu=0.04)")
            - Not working: Kernel.from_string("MallowsKernel(nu='auto')")
        """
        # Parse class name and arguments using regex
        match = re.fullmatch(r"(\w+)\((.*)\)", s.strip())
        if not match:
            raise ValueError(f"Invalid kernel string: {s}")

        class_name, args_str = match.groups()
        target_class = cls._find_subclass(class_name)

        kwargs = {}
        if args_str:
            try:
                # Only allow a restricted environment for safety
                kwargs = eval(f"dict({args_str})", {"__builtins__": {}, "dict": dict})
            except Exception as e:
                raise ValueError(f"Invalid argument list '{args_str}': {e}")

        return target_class(**kwargs)

    @classmethod
    def from_name_and_parameters(cls, kernel_name: str, **kernel_args):
        """
        Example usage:
        Kernel.from_name_and_parameters(kernel_name="MallowsKernel", **{'nu':0.04})

        Parameters
        ----------
        kernel_name :
        kernel_args :

        Returns
        -------

        """
        kernel_cls = cls._find_subclass(kernel_name)
        return kernel_cls(**kernel_args)

    def mmd_distribution_many_n(self, sample, nmin, nmax, step, rep, disjoint, replace, method, N,
                                use_cached_support_matrix):
        pass

    def get_eps(self, delta, na):
        pass

    def latex_str(self):
        return r"$\kappa$"
