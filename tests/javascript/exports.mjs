/**
 * Test wrapper functions for exported JavaScript functionality
 * These functions wrap the core functionality from extracted_from_html.mjs
 * to provide a clean testing interface
 */

import { parseCSV, getDistances, calcInterferenceLength, EXAMPLE_POSITIONS, EXAMPLE_LENGTH } from "./extracted_from_html.mjs";

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

