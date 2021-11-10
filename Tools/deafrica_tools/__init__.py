__version__ = "0.1.0"

__locales__ = __path__[0] + '/locales'


def set_lang(lang=None):
    if lang is not None:
        lang = [lang]
    
    import gettext
    try:
        translation = gettext.translation(
            'deafrica_tools', 
            localedir=__locales__, 
            languages=lang,
            fallback=True
        )
        translation.install()
        
    except FileNotFoundError:
        print(f'Could not load lang={lang}')
