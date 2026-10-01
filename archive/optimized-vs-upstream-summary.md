# optimized vs upstream - sumo 227k paired sensors

Six counterbalanced process runs per workload. Warm index. Lower is better except QPS.

- Capture: `archive/paired-sensors-c227k.jsonl` (48 rows)
- Subjects: optimized, upstream
- Checksum gate: **24/24 pairs match**

## Medians

| workload | metric | upstream | optimized | ratio (up/opt) |
|---|---|---:|---:|---:|
| file_picker_ 1t | wall us | 124,197 | 129,332 | 0.96x |
| file_picker_ 1t | cpu us | 122,135 | 129,427 | 0.94x |
| file_picker_ 1t | cycles | 355,981,656 | 379,493,683 | 0.94x |
| file_picker_ 1t | read B/op | 99,087 | 103,586 | 0.96x |
| file_picker_ 1t | faults/100k | 108,454,667 | 131,328,333 | 0.83x |
| file_picker_ 1t | peak RSS MB | 76.8 | 79.5 | 0.97x |
| file_picker_ 1t | QPS | 8.06 | 7.79 | 0.97x |
| file_picker_ 4t | wall us | 42,423 | 40,924 | 1.04x |
| file_picker_ 4t | cpu us | 142,292 | 148,281 | 0.96x |
| file_picker_ 4t | cycles | 411,507,164 | 433,656,854 | 0.95x |
| file_picker_ 4t | read B/op | 1,634,619 | 1,645,640 | 0.99x |
| file_picker_ 4t | faults/100k | 151,639,000 | 169,888,667 | 0.89x |
| file_picker_ 4t | peak RSS MB | 76.6 | 79.2 | 0.97x |
| file_picker_ 4t | QPS | 23.89 | 24.76 | 1.04x |
| score 1t | wall us | 124,515 | 119,751 | 1.04x |
| score 1t | cpu us | 124,609 | 135,078 | 0.92x |
| score 1t | cycles | 365,160,554 | 394,007,536 | 0.93x |
| score 1t | read B/op | 119,445 | 91,643 | 1.30x |
| score 1t | faults/100k | 519,366,000 | 560,621,000 | 0.93x |
| score 1t | peak RSS MB | 76.8 | 79.5 | 0.96x |
| score 1t | QPS | 8.04 | 8.36 | 1.04x |
| score 4t | wall us | 46,249 | 42,762 | 1.08x |
| score 4t | cpu us | 162,500 | 159,531 | 1.02x |
| score 4t | cycles | 473,207,572 | 470,137,527 | 1.01x |
| score 4t | read B/op | 2,178,199 | 1,901,096 | 1.15x |
| score 4t | faults/100k | 673,703,500 | 716,875,000 | 0.94x |
| score 4t | peak RSS MB | 76.8 | 79.5 | 0.97x |
| score 4t | QPS | 21.65 | 23.39 | 1.08x |

## QPS

QPS = 1e6 / us_per_op (sustained single-client). Higher is better; ratio is optimized/upstream.

| workload | upstream QPS | optimized QPS | ratio |
|---|---:|---:|---:|
| file_picker_ 1t | 8.06 | 7.79 | 0.97x |
| file_picker_ 4t | 23.89 | 24.76 | 1.04x |
| score 1t | 8.04 | 8.36 | 1.04x |
| score 4t | 21.65 | 23.39 | 1.08x |

## Outputs

- `archive/optimized-vs-upstream-atlas.html`
- `archive/charts/optimized-vs-upstream-atlas.png`
- `archive/optimized-vs-upstream-summary.md`

