def iso_code_to_int(code):
    out = 0
    for i, c in enumerate(code):
        num = ord(c)
        power = len(code) - (i + 1)
        out += num * (256 ** (power))
    return out