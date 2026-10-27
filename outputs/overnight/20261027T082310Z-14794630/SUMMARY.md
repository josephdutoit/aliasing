# Overnight experiment results

Started (UTC): 2026-10-27T08:23:10+00:00  
Updated (UTC): 2026-10-27T09:07:50+00:00  
Git revision: `d3b4d4f187e2c4648cf017b413cffe97e26d0fb7`  
Profile: bounded overnight survey. Reduced trials, epochs and grid sizes; see `results.json`.

| Experiment | Status | Minutes | Results | Log |
| --- | --- | ---: | --- | --- |
| 04 | completed | 0.1 | [04](04/) | [log](04/stdout.log) |
| 05 | completed | 0.1 | [05](05/) | [log](05/stdout.log) |
| 06 | timed_out | 25.0 | [06](06/) | [log](06/stdout.log) |
| 07 | completed | 1.6 | [07](07/) | [log](07/stdout.log) |
| 08 | failed | 7.8 | [08](08/) | [log](08/stdout.log) |
| 09 | completed | 0.2 | [09](09/) | [log](09/stdout.log) |
| 10 | completed | 0.1 | [10](10/) | [log](10/stdout.log) |
| 11 | completed | 0.1 | [11](11/) | [log](11/stdout.log) |
| 12 | completed | 0.1 | [12](12/) | [log](12/stdout.log) |
| 01 | completed | 0.2 | [01](01/) | [log](01/stdout.log) |
| 02 | failed | 0.1 | [02](02/) | [log](02/stdout.log) |
| 03 | completed | 0.9 | [03](03/) | [log](03/stdout.log) |
| glm-samplewise | completed | 1.6 | [glm-samplewise](glm-samplewise/) | [log](glm-samplewise/stdout.log) |
| glm-featurewise | completed | 1.6 | [glm-featurewise](glm-featurewise/) | [log](glm-featurewise/stdout.log) |
| frozen-superposition | completed | 0.2 | [frozen-superposition](frozen-superposition/) | [log](frozen-superposition/stdout.log) |
| learned-superposition | completed | 0.6 | [learned-superposition](learned-superposition/) | [log](learned-superposition/stdout.log) |
| gaussian-superposition | completed | 0.5 | [gaussian-superposition](gaussian-superposition/) | [log](gaussian-superposition/stdout.log) |
| no-bias-superposition | completed | 0.5 | [no-bias-superposition](no-bias-superposition/) | [log](no-bias-superposition/stdout.log) |
| embedded-complexity | completed | 0.5 | [embedded-complexity](embedded-complexity/) | [log](embedded-complexity/stdout.log) |
| embedded-mechanics | completed | 0.6 | [embedded-mechanics](embedded-mechanics/) | [log](embedded-mechanics/stdout.log) |
| embedded-regularization | completed | 0.5 | [embedded-regularization](embedded-regularization/) | [log](embedded-regularization/stdout.log) |
| embedded-heatmap | completed | 0.5 | [embedded-heatmap](embedded-heatmap/) | [log](embedded-heatmap/stdout.log) |
| legacy-exponential_regression | completed | 0.4 | [legacy-exponential_regression](legacy-exponential_regression/) | [log](legacy-exponential_regression/stdout.log) |
| legacy-logisitic_regression | completed | 0.4 | [legacy-logisitic_regression](legacy-logisitic_regression/) | [log](legacy-logisitic_regression/stdout.log) |
| legacy-poisson_regression | completed | 0.5 | [legacy-poisson_regression](legacy-poisson_regression/) | [log](legacy-poisson_regression/stdout.log) |
| 13 | skipped | 0.0 | — | — |

## Notes

- **06:** exceeded 25.0 minutes
- **08:** Traceback (most recent call last):  |   File "/home/jcdutoit/aliasing/experiments/08_trained_superposition_aliasing.py", line 234, in <module>  |     if __name__ == "__main__": main()  |                                ~~~~^^  |   File "/home/jcdutoit/aliasing/experiments/08_trained_superposition_aliasing.py", line 227, in main  |     if sel: ax.plot([r["sample_size"] for r in sel],[r[key] for r in sel],marker="o",label=task)  |                                                      ~^^^^^  | KeyError: 'probe_test_mse' 
- **02:**                                      [--sample-sizes SAMPLE_SIZES]  |                                      [--modeled-features MODELED_FEATURES]  |                                      [--full-features FULL_FEATURES]  |                                      [--input-dim INPUT_DIM]  |                                      [--n-train N_TRAIN] [--n-test N_TEST]  |                                      [--noise-std NOISE_STD]  |                                      --output-dir OUTPUT_DIR  | 02_random_feature_aliasing.py: error: the following arguments are required: --sweep 
- **13:** requires GAD_FEATURES_NPZ or GAD_TEXT_FILE (plus a cached model)

[Open the figure gallery](GALLERY.md). Source scripts record their own mathematical caveats.
