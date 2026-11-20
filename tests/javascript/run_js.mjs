import { readFileSync } from "node:fs";
import * as extracted from "./extracted_from_html.mjs";
import * as exports from "./exports.mjs";

// Merge both modules so functions from both can be called
const crossovers = { ...extracted, ...exports };

// Input format:
// {
//   "function": "crossover",
//   "args": [[[1, 2], [3, 4]]]
// }
//
// `args` is a list because this also supports functions with multiple
// arguments.

const request = JSON.parse(readFileSync(0, "utf8"));

const func = crossovers[request.function];

if (typeof func !== "function") {
    throw new Error(`Unknown JavaScript function: ${request.function}`);
}

const result = func(...request.args);

process.stdout.write(JSON.stringify(result));