import inspect

def print_module_methods(module):
    """ Prints signature and formal parameters for each method in the module.
    Args:
        module: A module that contains methods.
    """
    for fun in [x[1] for x in inspect.getmembers(module)][:]:
        try:
            print(fun.__name__+str(inspect.signature(fun)), end="\n\n")
        except:
            pass
