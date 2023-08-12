def set_latex_font():
    #try to set the nice latex font on the latex figures
    from matplotlib import rc
    try:
        rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
        rc('text', usetex=True)
    except:
        print("Could not set matplotlib to use latex font ")
    return None


def si_fm(number):
    # Display a value to 3dp in scientific forma using SI prefix
    # e.g. 1234 is 1.23k

    prefixes = {
        24: 'Y',  # yotta
        21: 'Z',  # zetta
        18: 'E',  # exa
        15: 'P',  # peta
        12: 'T',  # tera
        9: 'G',   # giga
        6: 'M',   # mega
        3: 'k',   # kilo
        0: '',    # (no prefix)
        -3: 'm',  # milli
        -6: 'µ',  # micro
        -9: 'n',  # nano
        -12: 'p', # pico
        -15: 'f', # femto
        -18: 'a', # atto
        -21: 'z', # zepto
        -24: 'y'  # yocto
    }

    # Find the appropriate prefix for the number
    for exp, prefix in prefixes.items():
        if number >= 10 ** exp:
            break

    
    value = round(number / (10 ** exp), 3)

    # Return the formatted string
    return f"{value}{prefix}"