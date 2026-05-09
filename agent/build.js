/**
 * CAFE-u Agent Build Script
 * Minifies the agent JS for production deployment.
 * Output: dist/cafeu.min.js
 */

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, 'src', 'cafeu.js');
const DIST = path.join(__dirname, 'dist');
const OUT = path.join(DIST, 'cafeu.min.js');

// Simple minification (removes comments, extra whitespace)
function minify(code) {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, '')       // Remove block comments
    .replace(/\/\/[^\n]*/g, '')              // Remove line comments
    .replace(/^\s*\/\/.*$/gm, '')            // Remove comment-only lines
    .replace(/\n\s*\n/g, '\n')               // Remove blank lines
    .replace(/^\s+/gm, '')                   // Remove leading whitespace
    .replace(/\s+$/gm, '')                   // Remove trailing whitespace
    .replace(/\s{2,}/g, ' ')                 // Collapse multiple spaces
    .replace(/\n/g, '')                      // Remove newlines
    .trim();
}

// Build
fs.mkdirSync(DIST, { recursive: true });

const source = fs.readFileSync(SRC, 'utf-8');
const minified = minify(source);
const header = `/* CAFE-u Agent v0.1.0 | MIT | https://github.com/AkhiChalasani/cafe-u */\n`;

fs.writeFileSync(OUT, header + minified);

const srcSize = Buffer.byteLength(source, 'utf-8');
const distSize = Buffer.byteLength(header + minified, 'utf-8');
const ratio = ((1 - distSize / srcSize) * 100).toFixed(1);

console.log(`CAFE-u Agent build complete:`);
console.log(`  Source: ${(srcSize / 1024).toFixed(1)} KB`);
console.log(`  Minified: ${(distSize / 1024).toFixed(1)} KB  (${ratio}% reduction)`);
console.log(`  Output: ${OUT}`);
