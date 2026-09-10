# 连夜重跑汇总（新数据 vs 旧报告）
生成时间: 2026-09-10 00:05:45

## exp13_arch
> 对照: 报告 2026-06-18: V12 力矩 85.7% (n=98), 位置 23.7%

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ballspeed7__noplot__positionmode__servebox | 199 | 47 | 23.6 | 0.1132 | 3 |
| ballspeed7__noplot__servebox | 198 | 169 | 85.4 | 0.0992 | 3 |

**小计**: 397 runs, 216 hits (54.4%), 6 errors

## exp15_speed_v2
> 对照: 报告 2026-06-22: 7→78% 9→90% 12→86% 15→58% (n=50)

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ballspeed10__noplot__servebox | 197 | 176 | 89.3 | 0.0966 | 3 |
| ballspeed11__noplot__servebox | 198 | 166 | 83.8 | 0.1006 | 3 |
| ballspeed12__noplot__servebox | 198 | 168 | 84.8 | 0.11 | 2 |
| ballspeed13__noplot__servebox | 199 | 132 | 66.3 | 0.1136 | 1 |
| ballspeed14__noplot__servebox | 199 | 146 | 73.4 | 0.1193 | 1 |
| ballspeed15__noplot__servebox | 200 | 121 | 60.5 | 0.1191 | 1 |
| ballspeed7__noplot__servebox | 198 | 169 | 85.4 | 0.0992 | 3 |
| ballspeed8__noplot__servebox | 198 | 188 | 94.9 | 0.0941 | 2 |
| ballspeed9__noplot__servebox | 197 | 186 | 94.4 | 0.0972 | 3 |

**小计**: 1784 runs, 1452 hits (81.4%), 19 errors

## exp16_limits_v2
> 对照: 报告 2026-07-09: real(TCP1.0) 30%, sim(TCP1.8) 78%

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ballspeed7__limitsconfigconfigs_real_robot.yaml__noplot__servebox | 198 | 72 | 36.4 | 0.1109 | 3 |
| ballspeed7__noplot__servebox | 199 | 170 | 85.4 | 0.0992 | 3 |

**小计**: 397 runs, 242 hits (61.0%), 6 errors

## exp14_pd_v2
> 对照: 报告 2026-06-22: 最优 Kp=500 Kr=0.15 → 80.0% (n=30)

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ballspeed7__kd10.0__kp100__noplot__positionmode__servebox | 47 | 29 | 61.7 | 0.0651 | 3 |
| ballspeed7__kd10.0__kp200__noplot__positionmode__servebox | 47 | 24 | 51.1 | 0.0621 | 3 |
| ballspeed7__kd100.0__kp1000__noplot__positionmode__servebox | 47 | 38 | 80.9 | 0.0579 | 3 |
| ballspeed7__kd112.5__kp750__noplot__positionmode__servebox | 47 | 40 | 85.1 | 0.0495 | 3 |
| ballspeed7__kd12.0__kp150__noplot__positionmode__servebox | 47 | 31 | 66.0 | 0.0789 | 3 |
| ballspeed7__kd15.0__kp100__noplot__positionmode__servebox | 47 | 29 | 61.7 | 0.0746 | 3 |
| ballspeed7__kd15.0__kp150__noplot__positionmode__servebox | 47 | 21 | 44.7 | 0.0689 | 3 |
| ballspeed7__kd15.0__kp300__noplot__positionmode__servebox | 48 | 20 | 41.7 | 0.0779 | 3 |
| ballspeed7__kd150.0__kp1000__noplot__positionmode__servebox | 48 | 41 | 85.4 | 0.0481 | 3 |
| ballspeed7__kd16.0__kp200__noplot__positionmode__servebox | 47 | 27 | 57.4 | 0.082 | 3 |
| ballspeed7__kd2.5__kp50__noplot__positionmode__servebox | 48 | 28 | 58.3 | 0.0784 | 3 |
| ballspeed7__kd20.0__kp200__noplot__positionmode__servebox | 47 | 21 | 44.7 | 0.0745 | 3 |
| ballspeed7__kd22.5__kp150__noplot__positionmode__servebox | 47 | 26 | 55.3 | 0.0823 | 3 |
| ballspeed7__kd24.0__kp300__noplot__positionmode__servebox | 47 | 20 | 42.6 | 0.074 | 3 |
| ballspeed7__kd25.0__kp500__noplot__positionmode__servebox | 47 | 17 | 36.2 | 0.0743 | 3 |
| ballspeed7__kd30.0__kp200__noplot__positionmode__servebox | 47 | 30 | 63.8 | 0.0744 | 3 |
| ballspeed7__kd30.0__kp300__noplot__positionmode__servebox | 47 | 29 | 61.7 | 0.0789 | 3 |
| ballspeed7__kd37.5__kp750__noplot__positionmode__servebox | 47 | 24 | 51.1 | 0.0931 | 3 |
| ballspeed7__kd4.0__kp50__noplot__positionmode__servebox | 47 | 36 | 76.6 | 0.0794 | 3 |
| ballspeed7__kd40.0__kp500__noplot__positionmode__servebox | 47 | 31 | 66.0 | 0.0881 | 3 |
| ballspeed7__kd45.0__kp300__noplot__positionmode__servebox | 47 | 35 | 74.5 | 0.0766 | 3 |
| ballspeed7__kd5.0__kp100__noplot__positionmode__servebox | 47 | 20 | 42.6 | 0.0616 | 3 |
| ballspeed7__kd5.0__kp50__noplot__positionmode__servebox | 47 | 37 | 78.7 | 0.071 | 3 |
| ballspeed7__kd50.0__kp1000__noplot__positionmode__servebox | 47 | 36 | 76.6 | 0.0842 | 3 |
| ballspeed7__kd50.0__kp500__noplot__positionmode__servebox | 47 | 38 | 80.9 | 0.0723 | 3 |
| ballspeed7__kd60.0__kp750__noplot__positionmode__servebox | 47 | 38 | 80.9 | 0.0703 | 3 |
| ballspeed7__kd7.5__kp150__noplot__positionmode__servebox | 47 | 32 | 68.1 | 0.0802 | 3 |
| ballspeed7__kd7.5__kp50__noplot__positionmode__servebox | 47 | 40 | 85.1 | 0.0702 | 3 |
| ballspeed7__kd75.0__kp500__noplot__positionmode__servebox | 47 | 43 | 91.5 | 0.054 | 3 |
| ballspeed7__kd75.0__kp750__noplot__positionmode__servebox | 47 | 39 | 83.0 | 0.0576 | 3 |
| ballspeed7__kd8.0__kp100__noplot__positionmode__servebox | 47 | 24 | 51.1 | 0.0779 | 3 |
| ballspeed7__kd80.0__kp1000__noplot__positionmode__servebox | 47 | 40 | 85.1 | 0.065 | 3 |

**小计**: 1507 runs, 984 hits (65.3%), 96 errors

## exp17a_noise
> 对照: 新实验（V12 首测）; V11 旧值仅参考: σ0.02 → ~5%

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ablationfull__ballspeed12__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 98 | 65 | 66.3 | 0.1046 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 98 | 67 | 68.4 | 0.0976 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 57 | 58.2 | 0.0943 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 50 | 51.0 | 0.0921 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 98 | 41 | 41.8 | 0.1041 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 98 | 25 | 25.5 | 0.087 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 98 | 21 | 21.4 | 0.1082 | 2 |
| ablationfull__ballspeed12__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 98 | 4 | 4.1 | 0.1166 | 2 |
| ablationfull__ballspeed12__noplot__servebox | 98 | 84 | 85.7 | 0.1084 | 2 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 97 | 58 | 59.8 | 0.0845 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 97 | 48 | 49.5 | 0.0867 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 97 | 42 | 43.3 | 0.0933 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 97 | 27 | 27.8 | 0.089 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 97 | 25 | 25.8 | 0.0878 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 97 | 10 | 10.3 | 0.1124 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 97 | 10 | 10.3 | 0.1149 | 3 |
| ablationfull__ballspeed7__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 97 | 1 | 1.0 | 0.237 | 3 |
| ablationfull__ballspeed7__noplot__servebox | 99 | 85 | 85.9 | 0.1021 | 3 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 98 | 68 | 69.4 | 0.0908 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 98 | 54 | 55.1 | 0.0819 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 53 | 54.1 | 0.0842 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 40 | 40.8 | 0.0882 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 98 | 38 | 38.8 | 0.0964 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 98 | 19 | 19.4 | 0.104 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 98 | 14 | 14.3 | 0.1088 | 2 |
| ablationfull__ballspeed9__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 98 | 1 | 1.0 | 0.0793 | 2 |
| ablationfull__ballspeed9__noplot__servebox | 98 | 94 | 95.9 | 0.0964 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 98 | 50 | 51.0 | 0.0873 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 98 | 48 | 49.0 | 0.0907 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 46 | 46.9 | 0.0869 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 40 | 40.8 | 0.0925 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 98 | 39 | 39.8 | 0.0981 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 98 | 22 | 22.4 | 0.0909 | 2 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 98 | 23 | 23.5 | 0.0854 | 4 |
| ablationnone__ballspeed12__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 97 | 3 | 3.1 | 0.0933 | 3 |
| ablationnone__ballspeed12__noplot__servebox | 98 | 52 | 53.1 | 0.0912 | 2 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 97 | 59 | 60.8 | 0.0864 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 97 | 49 | 50.5 | 0.0928 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 44 | 44.9 | 0.0875 | 4 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 97 | 29 | 29.9 | 0.0944 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 97 | 29 | 29.9 | 0.0956 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 97 | 9 | 9.3 | 0.1188 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 97 | 8 | 8.2 | 0.1134 | 3 |
| ablationnone__ballspeed7__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 97 | 1 | 1.0 | 0.2516 | 3 |
| ablationnone__ballspeed7__noplot__servebox | 97 | 79 | 81.4 | 0.0877 | 3 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.005__obsnoisevel0.05__obsusekf__servebox | 98 | 65 | 66.3 | 0.0839 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.005__obsnoisevel0.05__servebox | 98 | 55 | 56.1 | 0.0736 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 49 | 50.0 | 0.0788 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 35 | 35.7 | 0.0816 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.02__obsnoisevel0.2__obsusekf__servebox | 98 | 36 | 36.7 | 0.0811 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.02__obsnoisevel0.2__servebox | 98 | 20 | 20.4 | 0.0911 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.05__obsnoisevel0.5__obsusekf__servebox | 98 | 13 | 13.3 | 0.0828 | 2 |
| ablationnone__ballspeed9__noplot__obsnoisepos0.05__obsnoisevel0.5__servebox | 98 | 1 | 1.0 | 0.106 | 2 |
| ablationnone__ballspeed9__noplot__servebox | 98 | 78 | 79.6 | 0.082 | 2 |

**小计**: 5276 runs, 2083 hits (39.5%), 130 errors

## exp17b_perturb
> 对照: 新实验（V12 首测）; 无旧对照

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0 | 98 | 88 | 89.8 | 0.0939 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 87 | 88.8 | 0.0936 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 98 | 85 | 86.7 | 0.093 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 98 | 84 | 85.7 | 0.0929 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 89 | 90.8 | 0.0961 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0 | 98 | 85 | 86.7 | 0.0855 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 81 | 82.7 | 0.0959 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 98 | 80 | 81.6 | 0.0928 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 99 | 86 | 86.9 | 0.0948 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 83 | 84.7 | 0.0956 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0 | 98 | 75 | 76.5 | 0.0792 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 63 | 64.3 | 0.0848 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 98 | 61 | 62.2 | 0.0774 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 98 | 63 | 64.3 | 0.0838 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 60 | 61.2 | 0.0804 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms10 | 98 | 92 | 93.9 | 0.0951 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms100 | 98 | 90 | 91.8 | 0.096 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms25 | 98 | 90 | 91.8 | 0.0922 | 2 |
| ablationfull__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms50 | 98 | 90 | 91.8 | 0.0922 | 2 |
| ablationfull__ballspeed9__noplot__servebox | 99 | 95 | 96.0 | 0.0961 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0 | 98 | 79 | 80.6 | 0.0839 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 78 | 79.6 | 0.0829 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 98 | 80 | 81.6 | 0.0769 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 98 | 83 | 84.7 | 0.0833 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.05__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 82 | 83.7 | 0.0782 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0 | 98 | 81 | 82.7 | 0.0778 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 70 | 71.4 | 0.0811 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 98 | 68 | 69.4 | 0.0726 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 98 | 75 | 76.5 | 0.0798 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.1__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 72 | 73.5 | 0.0762 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0 | 98 | 70 | 71.4 | 0.0715 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms10 | 98 | 62 | 63.3 | 0.0859 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms100 | 99 | 63 | 63.6 | 0.0813 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms25 | 98 | 65 | 66.3 | 0.0831 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__spaceperturbm0.2__spaceperturbminm0.0__timeperturbminms0__timeperturbms50 | 98 | 60 | 61.2 | 0.0749 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms10 | 98 | 79 | 80.6 | 0.084 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms100 | 98 | 79 | 80.6 | 0.0732 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms25 | 98 | 82 | 83.7 | 0.0848 | 2 |
| ablationnone__ballspeed9__noplot__perturbsignrandom__randomperturb__servebox__timeperturbminms0__timeperturbms50 | 98 | 87 | 88.8 | 0.0809 | 2 |
| ablationnone__ballspeed9__noplot__servebox | 98 | 78 | 79.6 | 0.082 | 2 |

**小计**: 3923 runs, 3120 hits (79.5%), 80 errors

## exp17c_obsfreq
> 对照: 新实验; V11 旧值仅参考: off 模式 200→10Hz 零退化

| config | n | hits | 命中率% | hit误差m | err |
|---|---|---|---|---|---|
| ablationfull__ballspeed9__noplot__obsfreq10__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 24 | 24.5 | 0.0909 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq10__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 34 | 34.7 | 0.081 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq10__servebox | 98 | 94 | 95.9 | 0.0964 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq15__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 32 | 32.7 | 0.0782 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq15__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 33 | 33.7 | 0.0821 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq15__servebox | 98 | 90 | 91.8 | 0.0854 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq200__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 53 | 54.1 | 0.0868 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq200__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 40 | 40.8 | 0.0882 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq200__servebox | 99 | 95 | 96.0 | 0.0961 | 3 |
| ablationfull__ballspeed9__noplot__obsfreq30__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 42 | 42.9 | 0.0809 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq30__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 43 | 43.9 | 0.0836 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq30__servebox | 98 | 92 | 93.9 | 0.0912 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq60__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 51 | 52.0 | 0.0829 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq60__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 37 | 37.8 | 0.0936 | 2 |
| ablationfull__ballspeed9__noplot__obsfreq60__servebox | 98 | 94 | 95.9 | 0.096 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq10__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 99 | 29 | 29.3 | 0.0831 | 3 |
| ablationnone__ballspeed9__noplot__obsfreq10__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 34 | 34.7 | 0.087 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq10__servebox | 98 | 78 | 79.6 | 0.082 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq15__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 35 | 35.7 | 0.083 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq15__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 36 | 36.7 | 0.0924 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq15__servebox | 98 | 86 | 87.8 | 0.0699 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq200__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 49 | 50.0 | 0.0773 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq200__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 35 | 35.7 | 0.0816 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq200__servebox | 98 | 78 | 79.6 | 0.082 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq30__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 40 | 40.8 | 0.0829 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq30__obsnoisepos0.01__obsnoisevel0.1__servebox | 99 | 40 | 40.4 | 0.0964 | 3 |
| ablationnone__ballspeed9__noplot__obsfreq30__servebox | 98 | 81 | 82.7 | 0.0733 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq60__obsnoisepos0.01__obsnoisevel0.1__obsusekf__servebox | 98 | 45 | 45.9 | 0.0818 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq60__obsnoisepos0.01__obsnoisevel0.1__servebox | 98 | 34 | 34.7 | 0.082 | 2 |
| ablationnone__ballspeed9__noplot__obsfreq60__servebox | 98 | 78 | 79.6 | 0.082 | 2 |

**小计**: 2943 runs, 1632 hits (55.5%), 63 errors
