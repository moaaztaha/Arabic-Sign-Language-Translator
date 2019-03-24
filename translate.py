
# -*- coding: UTF-8 -*- 
def translate(text):
    return{
        "ain": 'ع',
        "al": 'ال',
        "aleff": 'أ',
        "bb": 'ب',
        "dal": 'د',
        "dha": 'ضا',
        "dhad": 'ض',
        "fa": 'ف',
        "gaaf": 'ق',
        "ghain": 'غ',
        "ha": 'ها',
        "haa": 'ه',
        "jeem": 'ج',
        "kaaf": 'ك',
        "khaa": 'كا',
        "la": 'لا',
        "laam": 'ل',
        "meem": 'م',
        "nun": 'ن',
        "ra": 'ر',
        "saad": 'ص',
        "seen": 'س',
        "sheen": 'ش',
        "ta": 'ت',
        "taa": 'تا',
        "thaa": 'ذا',
        "thal": 'ذ',
        "toot": 'ت',
        "waw": 'ع',
        "ya": 'ع',
        "yaa": 'ع',
        "zay": 'ع',
    }[text]


text = "dal"
x = translate(text)
print(x)
