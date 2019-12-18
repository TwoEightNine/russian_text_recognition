def find_optimum(min_func, start_value, init_step, min_step):
    step = init_step
    value = start_value
    while step > min_step:
        values = [value - step, value, value + step]
        mins = [min_func(i) for i in values]

        value = values[mins.index(min(mins))]
        step /= 2
    return value
