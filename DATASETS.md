# Datasets and provenance

The raw archives are downloaded from their original publishers. The repository contains derived result tables and aggregate figures.

| Dataset | Original source | SHA-256 of the expected archive |
|---|---|---|
| HAPT v2.1 | [UCI Machine Learning Repository](https://doi.org/10.24432/C54G7M) | `4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09` |
| MotionSense | [Official project](https://github.com/mmalekzadeh/motion-sense/tree/2a0c72755f898ae509e5cb3a1cd58f8c55db5499) | `bc632abddbb1e92db31daa7b740710bc8f756f90febe385e6f1dc32ff8d92285` |

For offline use, place the exact publisher archives at `data/hapt.zip` and `data/motionsense.zip`. The ingestion code verifies their hashes. A different directory can be supplied using `--data /path/to/data` for both experiment commands. The notebooks use the default data directory.

**HAPT.** Reyes-Ortiz, J. L., Anguita, D., Oneto, L., and Parra, X. (2015). *Smartphone-Based Recognition of Human Activities and Postural Transitions*. UCI Machine Learning Repository. The repository lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We use the accelerometer and gyroscope readings plus the six basic activity annotations. The parser uses the original 1-based inclusive annotation endpoints.

**MotionSense.** Malekzadeh, M., Clegg, R. G., Cavallaro, A., and Haddadi, H. (2019). *Mobile Sensor Data Anonymization*. ACM/IEEE IoTDI. [Paper DOI](https://doi.org/10.1145/3302505.3310068). We use the official DeviceMotion archive, adding gravity and user acceleration to reconstruct the acceleration channels and keeping the rotation-rate channels. Refer to the official repository for the authors' data and citation terms.

**Related methods.** [BayaHAR](https://arxiv.org/abs/2606.04798) uses closed-form prototypes for user adaptation. [Stojchevska et al., *From Lab to Real World*](https://doi.org/10.3390/s23104606) documents field annotation edits. Renode's [profiling documentation](https://renode.readthedocs.io/en/latest/execution-tracing/metrics-and-profiling.html) describes its instruction-based measurement facilities.
