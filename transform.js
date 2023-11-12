/**
 * jscodeshift script to log fixes for max-params rule by transforming functions with more than 3 parameters.
 *
 * Usage:
 * jscodeshift -t transform.js your_file.js
 */

module.exports = function(file, api) {
  const j = api.jscodeshift;
  const root = j(file.source);

  // Find all function declarations or expressions with more than 3 parameters
  root.find(j.FunctionDeclaration, { params: { length: { $gt: 3 } } })
    .forEach(path => {
      // Create an object with property names as param names and values as param values
      const paramObject = j.objectExpression(
        path.value.params.map(param => j.property('init', param, param))
      );
      // Log the proposed changes
      console.log(`Function "${path.value.id.name}" has more than 3 parameters. Proposed changes:`);
      console.log(`Replace:`);
      console.log(path.value.params);
      console.log(`with:`);
      console.log(`{ ${path.value.params.map(param => param.name).join(', ')} }`);
      console.log('---');
    });

  return root.toSource();
};
