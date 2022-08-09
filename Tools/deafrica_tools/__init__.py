__version__ = "0.1.10"

__locales__ = __path__[0] + '/locales'


def set_lang(lang=None):
    if lang is None:
        import os
        os_lang = os.getenv('LANG')
        
        # Just take the first 2 letters: 'fr' not 'fr_FR.UTF-8'
        if os_lang is not None and len(os_lang) >=2:
            lang = [os_lang[:2]]
    else:
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
