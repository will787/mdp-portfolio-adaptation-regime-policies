from pathlib import Path
import sys
import os

def try_read_dir():
    try:
        BASE_DIR = Path(__file__).resolve().parents[2]
    except NameError:
        diretorio_atual =  Path.cwd()

        while_raiz = diretorio_atual
        while not (while_raiz / 'src').exists() and while_raiz.parent != while_raiz:
            while_raiz = while_raiz.parent
        
        BASE_DIR = while_raiz

    return BASE_DIR
