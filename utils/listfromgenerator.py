def generate_list_functions_from_generator_functions(f):
    def f_(*args):
        return list(f(*args))

    return f_