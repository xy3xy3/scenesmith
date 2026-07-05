from omegaconf import OmegaConf


def register_resolvers():
    def _register(name, resolver):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)

    _register("not", lambda boolean: not boolean)

    _register("equal", lambda arg1, arg2: arg1 == arg2)

    def conditional_resolver(condition, arg1, arg2):
        return arg1 if condition else arg2

    _register("ifelse", conditional_resolver)
