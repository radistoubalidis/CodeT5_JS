const esprima = require('esprima');
const fs = require('fs');
const escodegen = require('escodegen');

const filePath = process.argv[2];
const code = fs.readFileSync(filePath, 'utf-8');

if (!filePath){
    console.error('Usage: node your_script.js <path_to_js_file>');
    process.exit(1);
}

// Parse the code into an AST
const ast = esprima.parseScript(code, { loc: true });

// Function to recursively traverse the AST and transform function parameters
function transformFunctionParams(node) {
  if (node.type === 'FunctionDeclaration') {
    // Found a function declaration
    const paramObject = esprima.parseScript(`({${node.params.map(param => param.name).join(', ')}})`);

    // Replace the original parameters with the object parameter
    node.params = [paramObject];
  }

  // Recursively traverse child nodes
  for (const key in node) {
    if (node[key] && typeof node[key] === 'object') {
      transformFunctionParams(node[key]);
    }
  }
}

// Start the traversal from the root of the AST
transformFunctionParams(ast);

// Convert the modified AST back to code
const modifiedCode = escodegen.generate(ast);

console.log(modifiedCode);
