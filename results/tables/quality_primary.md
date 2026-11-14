| Dataset     | Method                    | Macro-F1 % [95% CI]   | Δ vs wrong labels (pp)   |
|:------------|:--------------------------|:----------------------|:-------------------------|
| HAPT        | Source only               | 81.05 [77.79, 84.00]  | +3.13 [-0.04, 6.23]      |
| HAPT        | Keep wrong labels         | 77.92 [75.16, 80.56]  | +0.00 [0.00, 0.00]       |
| HAPT        | Add corrected labels only | 79.22 [76.53, 81.74]  | +1.29 [0.34, 2.28]       |
| HAPT        | Remove source prior       | 78.10 [74.37, 81.51]  | +0.18 [-2.42, 2.88]      |
| HAPT        | 8-block edit history      | 78.61 [76.01, 81.13]  | +0.68 [-0.19, 1.57]      |
| HAPT        | 2-second edit grid        | 79.69 [76.91, 82.29]  | +1.76 [-0.11, 3.90]      |
| HAPT        | Full-vector reference     | 80.15 [77.30, 82.96]  | +2.22 [-0.31, 4.78]      |
| HAPT        | Unpacked summaries        | 80.15 [77.30, 82.96]  | +2.22 [-0.31, 4.78]      |
| HAPT        | Packed journal            | 80.15 [77.30, 82.96]  | +2.22 [-0.31, 4.78]      |
| MotionSense | Source only               | 77.18 [69.19, 84.20]  | -6.68 [-13.95, -0.12]    |
| MotionSense | Keep wrong labels         | 83.86 [80.29, 87.28]  | +0.00 [0.00, 0.00]       |
| MotionSense | Add corrected labels only | 86.40 [83.01, 89.67]  | +2.54 [0.90, 4.09]       |
| MotionSense | Remove source prior       | 87.92 [84.40, 91.12]  | +4.06 [0.63, 7.25]       |
| MotionSense | 8-block edit history      | 87.30 [84.54, 90.01]  | +3.44 [1.73, 5.21]       |
| MotionSense | 2-second edit grid        | 87.15 [83.92, 90.30]  | +3.29 [1.08, 5.46]       |
| MotionSense | Full-vector reference     | 88.69 [85.62, 91.62]  | +4.83 [2.30, 7.50]       |
| MotionSense | Unpacked summaries        | 88.69 [85.62, 91.62]  | +4.83 [2.30, 7.50]       |
| MotionSense | Packed journal            | 88.69 [85.62, 91.62]  | +4.83 [2.30, 7.50]       |
