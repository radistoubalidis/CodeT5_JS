import argparse
import esprima

from modules import Esprima

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--js_file')
    return parser.parse_args()

def readJS(path):
    with open(path, 'r') as f:
        code_str = f.read()
    return code_str


def main():
    args = getArgs()
    JS_FILE = args.js_file
    code_str = readJS(JS_FILE)
    ast = Esprima.read(code_str)
    if ast.body != None:
        for node in ast.body:
            if node.type == 'FunctionDeclaration':
                Esprima.max_params()
    
    

if __name__ == '__main__':
    main()

