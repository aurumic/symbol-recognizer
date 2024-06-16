import { writable } from 'svelte/store';

export const predictions = writable([
    { character: "H", probability: 0.6509329676628113 },
    { character: "M", probability: 0.22670942544937134 },
    { character: "4", probability: 0.033812087029218674 },
    { character: "g", probability: 0.022895483300089836 },
    { character: "U", probability: 0.016536856070160866 }
]);