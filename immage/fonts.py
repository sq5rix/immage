import matplotlib.font_manager as fm

def list_available_fonts():
    """
    Lists all available TrueType fonts on the system.

    Returns:
    - A list of font file paths.
    """
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in font_list:
        print(font)
    return font_list


