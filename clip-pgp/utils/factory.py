from methods.tprompt import TPrompts


def get_model(model_name, args):
    name = model_name.lower()
    options = {"tprompts": TPrompts}
    return options[name](args)
