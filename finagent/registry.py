# Registry for managing different modules in the project

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, force=False):
        def _register(cls):
            if not force and cls.__name__ in self._module_dict:
                raise KeyError(f'{cls.__name__} is already registered in {self._name}')
            self._module_dict[cls.__name__] = cls
            return cls
        return _register

# Define registries for different components
DOWNLOADER = Registry('downloader')
CLEANER = Registry('cleaner')  # Registry for data cleaners