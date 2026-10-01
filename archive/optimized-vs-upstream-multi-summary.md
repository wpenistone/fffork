# optimized vs upstream - multi-corpus paired sensors

Three counterbalanced blocks per workload. Warm index. Lower is better except QPS.

- Capture: `archive/paired-sensors-multi.jsonl` (96 rows)
- Subjects: upstream, optimized
- Checksum gate: **42/48 pairs match -- DIFFERENCES FOUND**

Corpus shapes: sumo tiny-files (227k), linux medium-source (100k), pager flat (51k), faces large-media (19k).

## CHECKSUM DIFFERENCES (DO NOT IGNORE)

- ('linux-100k', 'score', 1, 1): [('upstream', 4072000), ('optimized', 4073750)]
- ('linux-100k', 'score', 1, 2): [('optimized', 4073750), ('upstream', 4072000)]
- ('linux-100k', 'score', 1, 3): [('upstream', 4072000), ('optimized', 4073750)]
- ('linux-100k', 'score', 4, 1): [('upstream', 4072000), ('optimized', 4073750)]
- ('linux-100k', 'score', 4, 2): [('optimized', 4073750), ('upstream', 4072000)]
- ('linux-100k', 'score', 4, 3): [('upstream', 4072000), ('optimized', 4073750)]

## Medians

| corpus | workload | metric | upstream | optimized | ratio (up/opt) |
|---|---|---|---:|---:|---:|
| sumo-227k | file_picker_ 1t | wall us | 65,496 | 63,668 | 1.03x |
| sumo-227k | file_picker_ 1t | cpu us | 69,844 | 68,906 | 1.01x |
| sumo-227k | file_picker_ 1t | cycles | 205,347,402 | 207,777,838 | 0.99x |
| sumo-227k | file_picker_ 1t | read B/op | 73,413 | 172,117 | 0.43x |
| sumo-227k | file_picker_ 1t | faults/100k | 110,375,000 | 118,600,000 | 0.93x |
| sumo-227k | file_picker_ 1t | peak RSS MB | 76.6 | 78.6 | 0.97x |
| sumo-227k | file_picker_ 1t | QPS | 15.27 | 15.71 | 1.03x |
| sumo-227k | file_picker_ 4t | wall us | 22,732 | 22,075 | 1.03x |
| sumo-227k | file_picker_ 4t | cpu us | 98,281 | 102,500 | 0.96x |
| sumo-227k | file_picker_ 4t | cycles | 286,386,022 | 294,612,844 | 0.97x |
| sumo-227k | file_picker_ 4t | read B/op | 1,997,767 | 1,789,373 | 1.12x |
| sumo-227k | file_picker_ 4t | faults/100k | 146,303,000 | 161,683,000 | 0.90x |
| sumo-227k | file_picker_ 4t | peak RSS MB | 76.7 | 78.6 | 0.98x |
| sumo-227k | file_picker_ 4t | QPS | 43.99 | 45.30 | 1.03x |
| sumo-227k | score 1t | wall us | 57,595 | 56,174 | 1.03x |
| sumo-227k | score 1t | cpu us | 74,479 | 82,812 | 0.90x |
| sumo-227k | score 1t | cycles | 215,430,543 | 256,279,221 | 0.84x |
| sumo-227k | score 1t | read B/op | 1,293,329 | 1,760,615 | 0.73x |
| sumo-227k | score 1t | faults/100k | 516,791,667 | 523,280,000 | 0.99x |
| sumo-227k | score 1t | peak RSS MB | 76.7 | 78.5 | 0.98x |
| sumo-227k | score 1t | QPS | 17.36 | 17.80 | 1.03x |
| sumo-227k | score 4t | wall us | 28,575 | 24,533 | 1.16x |
| sumo-227k | score 4t | cpu us | 96,615 | 108,073 | 0.89x |
| sumo-227k | score 4t | cycles | 283,866,196 | 297,979,269 | 0.95x |
| sumo-227k | score 4t | read B/op | 2,612,145 | 1,980,578 | 1.32x |
| sumo-227k | score 4t | faults/100k | 682,180,000 | 679,861,667 | 1.00x |
| sumo-227k | score 4t | peak RSS MB | 76.9 | 78.6 | 0.98x |
| sumo-227k | score 4t | QPS | 35.00 | 40.76 | 1.16x |
| linux-100k | file_picker_ 1t | wall us | 10,951 | 10,151 | 1.08x |
| linux-100k | file_picker_ 1t | cpu us | 22,070 | 10,352 | 2.13x |
| linux-100k | file_picker_ 1t | cycles | 63,020,652 | 30,587,749 | 2.06x |
| linux-100k | file_picker_ 1t | read B/op | 3,640 | 4,815 | 0.76x |
| linux-100k | file_picker_ 1t | faults/100k | 670,000 | 0 | n/a |
| linux-100k | file_picker_ 1t | peak RSS MB | 125.0 | 37.2 | 3.36x |
| linux-100k | file_picker_ 1t | QPS | 91.32 | 98.51 | 1.08x |
| linux-100k | file_picker_ 4t | wall us | 3,674 | 3,631 | 1.01x |
| linux-100k | file_picker_ 4t | cpu us | 16,406 | 9,961 | 1.65x |
| linux-100k | file_picker_ 4t | cycles | 46,644,409 | 34,518,901 | 1.35x |
| linux-100k | file_picker_ 4t | read B/op | 1,419 | 1,123 | 1.26x |
| linux-100k | file_picker_ 4t | faults/100k | 5,113,750 | 2,835,000 | 1.80x |
| linux-100k | file_picker_ 4t | peak RSS MB | 110.4 | 37.2 | 2.97x |
| linux-100k | file_picker_ 4t | QPS | 272.19 | 275.39 | 1.01x |
| linux-100k | score 1t | wall us | 16,258 | 14,636 | 1.11x |
| linux-100k | score 1t | cpu us | 32,500 | 20,938 | 1.55x |
| linux-100k | score 1t | cycles | 92,443,519 | 60,917,073 | 1.52x |
| linux-100k | score 1t | read B/op | 7,312 | 6,246 | 1.17x |
| linux-100k | score 1t | faults/100k | 168,716,000 | 192,080,000 | 0.88x |
| linux-100k | score 1t | peak RSS MB | 133.7 | 37.1 | 3.60x |
| linux-100k | score 1t | QPS | 61.51 | 68.32 | 1.11x |
| linux-100k | score 4t | wall us | 8,418 | 7,324 | 1.15x |
| linux-100k | score 4t | cpu us | 24,375 | 21,562 | 1.13x |
| linux-100k | score 4t | cycles | 78,654,153 | 67,888,746 | 1.16x |
| linux-100k | score 4t | read B/op | 3,830 | 3,625 | 1.06x |
| linux-100k | score 4t | faults/100k | 167,074,000 | 245,338,000 | 0.68x |
| linux-100k | score 4t | peak RSS MB | 136.2 | 38.2 | 3.57x |
| linux-100k | score 4t | QPS | 118.79 | 136.54 | 1.15x |
| pager-flat-51k | file_picker_ 1t | wall us | 1,933 | 1,923 | 1.01x |
| pager-flat-51k | file_picker_ 1t | cpu us | 2,148 | 1,953 | 1.10x |
| pager-flat-51k | file_picker_ 1t | cycles | 5,844,484 | 5,835,938 | 1.00x |
| pager-flat-51k | file_picker_ 1t | read B/op | 69 | 67 | 1.03x |
| pager-flat-51k | file_picker_ 1t | faults/100k | 0 | 0 | n/a |
| pager-flat-51k | file_picker_ 1t | peak RSS MB | 21.3 | 50.5 | 0.42x |
| pager-flat-51k | file_picker_ 1t | QPS | 517.41 | 520.03 | 1.01x |
| pager-flat-51k | file_picker_ 4t | wall us | 817 | 853 | 0.96x |
| pager-flat-51k | file_picker_ 4t | cpu us | 1,953 | 2,539 | 0.77x |
| pager-flat-51k | file_picker_ 4t | cycles | 7,117,265 | 7,204,746 | 0.99x |
| pager-flat-51k | file_picker_ 4t | read B/op | 27 | 29 | 0.93x |
| pager-flat-51k | file_picker_ 4t | faults/100k | 2,457,500 | 2,025,000 | 1.21x |
| pager-flat-51k | file_picker_ 4t | peak RSS MB | 21.2 | 50.5 | 0.42x |
| pager-flat-51k | file_picker_ 4t | QPS | 1224.30 | 1172.46 | 0.96x |
| pager-flat-51k | score 1t | wall us | 1,416 | 1,326 | 1.07x |
| pager-flat-51k | score 1t | cpu us | 1,562 | 1,250 | 1.25x |
| pager-flat-51k | score 1t | cycles | 4,290,324 | 4,016,032 | 1.07x |
| pager-flat-51k | score 1t | read B/op | 46 | 47 | 0.98x |
| pager-flat-51k | score 1t | faults/100k | 1,064,000 | 1,786,000 | 0.60x |
| pager-flat-51k | score 1t | peak RSS MB | 21.2 | 50.5 | 0.42x |
| pager-flat-51k | score 1t | QPS | 706.18 | 754.34 | 1.07x |
| pager-flat-51k | score 4t | wall us | 844 | 803 | 1.05x |
| pager-flat-51k | score 4t | cpu us | 2,188 | 2,500 | 0.88x |
| pager-flat-51k | score 4t | cycles | 5,576,695 | 5,310,141 | 1.05x |
| pager-flat-51k | score 4t | read B/op | 48 | 29 | 1.63x |
| pager-flat-51k | score 4t | faults/100k | 5,082,000 | 2,298,000 | 2.21x |
| pager-flat-51k | score 4t | peak RSS MB | 21.2 | 50.5 | 0.42x |
| pager-flat-51k | score 4t | QPS | 1184.70 | 1245.43 | 1.05x |
| faces-19k | file_picker_ 1t | wall us | 886 | 840 | 1.05x |
| faces-19k | file_picker_ 1t | cpu us | 781 | 781 | 1.00x |
| faces-19k | file_picker_ 1t | cycles | 3,424,443 | 2,443,314 | 1.40x |
| faces-19k | file_picker_ 1t | read B/op | 83,308 | 0 | n/a |
| faces-19k | file_picker_ 1t | faults/100k | 0 | 0 | n/a |
| faces-19k | file_picker_ 1t | peak RSS MB | 16.4 | 25.9 | 0.63x |
| faces-19k | file_picker_ 1t | QPS | 1128.51 | 1190.58 | 1.05x |
| faces-19k | file_picker_ 4t | wall us | 463 | 849 | 0.54x |
| faces-19k | file_picker_ 4t | cpu us | 1,302 | 781 | 1.67x |
| faces-19k | file_picker_ 4t | cycles | 4,752,220 | 2,462,418 | 1.93x |
| faces-19k | file_picker_ 4t | read B/op | 402,909 | 0 | n/a |
| faces-19k | file_picker_ 4t | faults/100k | 2,410,000 | 0 | n/a |
| faces-19k | file_picker_ 4t | peak RSS MB | 16.4 | 25.6 | 0.64x |
| faces-19k | file_picker_ 4t | QPS | 2161.97 | 1177.47 | 0.54x |
| faces-19k | score 1t | wall us | 2,301 | 2,312 | 1.00x |
| faces-19k | score 1t | cpu us | 3,125 | 4,297 | 0.73x |
| faces-19k | score 1t | cycles | 9,530,487 | 10,680,785 | 0.89x |
| faces-19k | score 1t | read B/op | 272,234 | 0 | n/a |
| faces-19k | score 1t | faults/100k | 38,275,000 | 38,945,000 | 0.98x |
| faces-19k | score 1t | peak RSS MB | 16.3 | 25.9 | 0.63x |
| faces-19k | score 1t | QPS | 434.50 | 432.55 | 1.00x |
| faces-19k | score 4t | wall us | 1,829 | 2,506 | 0.73x |
| faces-19k | score 4t | cpu us | 5,469 | 2,344 | 2.33x |
| faces-19k | score 4t | cycles | 11,086,255 | 12,263,887 | 0.90x |
| faces-19k | score 4t | read B/op | 211,282 | 0 | n/a |
| faces-19k | score 4t | faults/100k | 40,417,500 | 39,427,500 | 1.03x |
| faces-19k | score 4t | peak RSS MB | 18.1 | 25.8 | 0.70x |
| faces-19k | score 4t | QPS | 546.72 | 398.99 | 0.73x |

## QPS (higher is better)

QPS = 1e6 / us_per_op (sustained single-client). Ratio is optimized/upstream.

| corpus | workload | upstream QPS | optimized QPS | ratio |
|---|---|---:|---:|---:|
| sumo-227k | file_picker_ 1t | 15.27 | 15.71 | 1.03x |
| sumo-227k | file_picker_ 4t | 43.99 | 45.30 | 1.03x |
| sumo-227k | score 1t | 17.36 | 17.80 | 1.03x |
| sumo-227k | score 4t | 35.00 | 40.76 | 1.16x |
| linux-100k | file_picker_ 1t | 91.32 | 98.51 | 1.08x |
| linux-100k | file_picker_ 4t | 272.19 | 275.39 | 1.01x |
| linux-100k | score 1t | 61.51 | 68.32 | 1.11x |
| linux-100k | score 4t | 118.79 | 136.54 | 1.15x |
| pager-flat-51k | file_picker_ 1t | 517.41 | 520.03 | 1.01x |
| pager-flat-51k | file_picker_ 4t | 1224.30 | 1172.46 | 0.96x |
| pager-flat-51k | score 1t | 706.18 | 754.34 | 1.07x |
| pager-flat-51k | score 4t | 1184.70 | 1245.43 | 1.05x |
| faces-19k | file_picker_ 1t | 1128.51 | 1190.58 | 1.05x |
| faces-19k | file_picker_ 4t | 2161.97 | 1177.47 | 0.54x |
| faces-19k | score 1t | 434.50 | 432.55 | 1.00x |
| faces-19k | score 4t | 546.72 | 398.99 | 0.73x |

## Outputs

- `archive/optimized-vs-upstream-multi-atlas.html`
- `archive/charts/optimized-vs-upstream-multi-atlas.png`
- `archive/optimized-vs-upstream-multi-summary.md`
