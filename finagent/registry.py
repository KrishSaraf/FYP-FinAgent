# from mmengine.registry import Registry

# DATASET = Registry('data', locations=['finagent.data'])
# PROMPT = Registry('prompt', locations=['finagent.prompt'])
# AGENT = Registry('agent', locations=['finagent.agent'])
# PROVIDER = Registry('provider', locations=['finagent.provider'])
# DOWNLOADER = Registry('downloader', locations=['finagent.downloader'])
# PROCESSOR = Registry('processor', locations=['finagent.processor'])
# ENVIRONMENT = Registry('environment', locations=['finagent.environment'])
# MEMORY = Registry('memory', locations=['finagent.memory'])
# PLOTS = Registry('plot', locations=['finagent.plots'])

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

DOWNLOADER = Registry('downloader')