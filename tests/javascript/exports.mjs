/**
 * Test wrapper functions for exported JavaScript functionality
 * These functions wrap the core functionality from extracted_from_html.mjs
 * to provide a clean testing interface
 */

import { parseCSV, getDistances, calcInterferenceLength, calcCoC, EXAMPLE_POSITIONS, EXAMPLE_LENGTH } from "./extracted_from_html.mjs";

export function getExampleData() {
    const positions = parseCSV(EXAMPLE_POSITIONS.replaceAll(";", "\n"), 1);
    const distances = getDistances(positions);
    return {
        positions,
        distances,
        length: EXAMPLE_LENGTH
    };
}

export function calcInterferenceLengthSimple(positions, distances, length) {
    return calcInterferenceLength(positions, distances, length, 1);
}

export function getDistancesExport(positions) {
    return getDistances(positions);
}

export function calcCoCMwhite(positions, length, numberOfIntervals) {
    const [coc_x, coc_y] = calcCoC(positions, length, numberOfIntervals, "mwhite", false);
    return { 
        coc_x: Array.from(coc_x), 
        coc_y: Array.from(coc_y).map(v => isNaN(v) ? null : v) 
    };
}

export function calcCoCTotalAverage(positions, length, numberOfIntervals) {
    const [coc_x, coc_y, ...rest] = calcCoC(positions, length, numberOfIntervals, "total_average", false);
    return { 
        coc_x: Array.from(coc_x), 
        coc_y: Array.from(coc_y).map(v => isNaN(v) ? null : v) 
    };
}

export function calcCoCMernst(positions, length, numberOfIntervals) {
    const [coc_x, coc_y] = calcCoC(positions, length, numberOfIntervals, "mernst", false);
    return { 
        coc_x: Array.from(coc_x), 
        coc_y: Array.from(coc_y).map(v => isNaN(v) ? null : v) 
    };
}


