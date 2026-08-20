def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class OpManager:
    def __init__(self):
        self.registry = {}

    def register(self, *names):
        """Decorator to register a class with a specific name."""

        def decorator(cls):
            for name in names:
                if name in self.registry:
                    # print(
                    #     f"WARN: op name [{name}] 's cost model [{self.registry[name].__name__}] "
                    #     f"is replaced by new one [{cls.__name__}].",
                    #     flush=True,
                    # )
                    pass

                self.registry[name] = cls
            return cls

        return decorator

    def create_instance(self, name, *args, **kwargs):
        """Create an instance of a registered class by its name."""
        cls = self.registry.get(name, None)
        if cls is not None:
            return cls(*args, **kwargs)
        else:
            raise ValueError(f"op name [{name}] 's cost model is not registered.")

    def predict_single_op_pref(self, name, op, hardware):
        return self.create_instance(name, op, hardware)()


op_manager = OpManager()
