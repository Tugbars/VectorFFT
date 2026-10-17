// Build-time math: TeX -> KaTeX HTML (+ MathML), once per build; the pages carry no JavaScript.
//
// usage: node katex_render.js <katex bundle> < requests.json > rendered.json
//   requests.json: [{"tex": "...", "display": true|false}, ...]
//   rendered.json: {"version": "0.16.21", "html": ["<span class=\"katex\">...", ...]}   (same order)
//   On a TeX error: exit 1, and stdout is {"errors": [{"index": i, "message": "..."}, ...]}.
//
// The vendored KaTeX (vendor/katex-0.16.21/katex.js) is the copy VS Code ships inside its
// built-in markdown-math extension (KaTeX 0.16.21, MIT), an esbuild ES module that holds
// KaTeX's UMD factory; a plain UMD katex.js or katex.min.js from the npm package also loads.
'use strict';
const fs = require('fs');
const vm = require('vm');

function loadKatex(bundlePath) {
  let src = fs.readFileSync(bundlePath, 'utf8');
  if (/^\s*(\/\*[\s\S]*?\*\/\s*)?\(function webpackUniversalModuleDefinition|typeof exports/.test(src.slice(0, 600))
      && !/export\{/.test(src.slice(-200))) {
    return require(bundlePath);                      // a plain UMD katex.js / katex.min.js
  }
  // VS Code bundle: find the lazy-CommonJS wrapper that holds KaTeX's UMD factory.
  const m = src.match(/var (\w+)=\w+\(\(\w+,\w+\)=>\{\(function\(\w+,\w+\)\{typeof \w+=="object"&&typeof \w+=="object"\?\w+\.exports=\w+\(\)/);
  if (!m) throw new Error('KaTeX UMD wrapper not found in ' + bundlePath);
  src = src.replace(/import\.meta\.url/g, JSON.stringify('file:///katex.js'))
           .replace(/export\{[^}]*\};?\s*$/, '');
  const sandbox = { console };
  vm.createContext(sandbox);
  vm.runInContext(src + `\n;globalThis.__katex = ${m[1]}();`, sandbox, { filename: bundlePath });
  const k = sandbox.__katex;
  return (k && k.renderToString) ? k : k.default;
}

const katex = loadKatex(process.argv[2]);
const reqs = JSON.parse(fs.readFileSync(0, 'utf8'));
const errors = [];
const out = reqs.map((r, i) => {
  try {
    return katex.renderToString(r.tex, {
      displayMode: !!r.display,
      throwOnError: true,        // a TeX error fails the build instead of shipping red text
      output: 'htmlAndMathml',   // HTML for the eye, MathML for assistive technology
      strict: 'error',
      trust: false,
    });
  } catch (e) {
    errors.push({ index: i, message: String(e && e.message || e) });
    return '';
  }
});
if (errors.length) {
  process.stdout.write(JSON.stringify({ errors }));
  process.exit(1);
}
process.stdout.write(JSON.stringify({ version: katex.version, html: out }));
