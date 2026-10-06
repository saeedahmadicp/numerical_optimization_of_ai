### Q1 rate fits (ln E = α − C·g(n); RMS residual in decades)

| function | approximant | window n | algebraic C (1st half, 2nd half) | RMS | root-exp C (1st half, 2nd half) | RMS |
|---|---|---|---|---|---|---|
| |x| on [-1, 1] | chebyshev | 10–1000 | 1.00 (1.10, 1.01) | 0.11 | 0.17 (0.46, 0.12) | 0.21 |
| |x| on [-1, 1] | aaa clustered/columns | 6–92 | 9.66 (7.84, 12.53) | 0.81 | 3.38 (3.41, 3.05) | 0.71 |
| |x| on [-1, 1] | aaa clustered/none | 6–95 | 9.86 (8.68, 13.46) | 0.82 | 3.39 (3.74, 3.20) | 0.77 |
| |x| on [-1, 1] | aaa equispaced/columns | 6–64 | 4.13 (5.77, -0.66) | 0.85 | 1.54 (2.82, -0.20) | 0.93 |
| √x on [0, 1] | chebyshev | 10–1000 | 0.98 (0.96, 0.99) | 0.01 | 0.17 (0.39, 0.12) | 0.17 |
| √x on [0, 1] | aaa clustered/columns | 6–45 | 9.87 (8.39, 12.98) | 0.30 | 4.47 (4.55, 4.41) | 0.11 |
| √x on [0, 1] | aaa clustered/none | 6–24 | 8.46 (7.98, 9.05) | 0.10 | 4.69 (5.11, 4.14) | 0.11 |
| √x on [0, 1] | aaa equispaced/columns | 6–19 | 0.84 (0.94, 0.72) | 0.02 | 0.49 (0.62, 0.36) | 0.03 |

### Q1 AAA runs (final step and best step on the test grid)

| function | Z | scaling | M | m | converged | final sample err | final test err | best test err (n) | real poles in interval (final) | doublets (final) | unresolved pole estimates (final) | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |x| on [-1, 1] | equispaced | columns | 20000 | 65 | True | 2.5e-14 | 1.9e-05 | 1.9e-05 (64) | 1 | 1 | 0 | 2.8 |
| |x| on [-1, 1] | equispaced | none | 20000 | 61 | True | 3.8e-14 | 2.1e-05 | 2.0e-05 (59) | 1 | 1 | 0 | 1.9 |
| |x| on [-1, 1] | clustered | columns | 5998 | 100 | False | 3.7e-12 | 3.7e-12 | 3.7e-12 (99) | 1 | 29 | 3 | 1.3 |
| |x| on [-1, 1] | clustered | none | 5998 | 100 | False | 4.1e-12 | 4.1e-12 | 4.1e-12 (99) | 1 | 29 | 1 | 1.5 |
| √x on [0, 1] | equispaced | columns | 20000 | 20 | True | 4.9e-14 | 3.2e-04 | 3.2e-04 (19) | 0 | 0 | 0 | 0.1 |
| √x on [0, 1] | equispaced | none | 20000 | 20 | True | 9.4e-14 | 3.2e-04 | 3.2e-04 (19) | 0 | 0 | 0 | 0.1 |
| √x on [0, 1] | clustered | columns | 5999 | 65 | True | 8.1e-14 | 8.4e-14 | 8.4e-14 (64) | 0 | 33 | 27 | 0.8 |
| √x on [0, 1] | clustered | none | 5999 | 100 | False | 3.9e-02 | 2.6e+00 | 2.9e-08 (24) | 82 | 7 | 2 | 3.8 |
| tanh(50x) on [-1, 1] | equispaced | columns | 2000 | 25 | True | 2.2e-14 | 2.3e-14 | 2.3e-14 (24) | 0 | 0 | 0 | 0.0 |
| tanh(50x) on [-1, 1] | equispaced | none | 2000 | 25 | True | 3.9e-14 | 3.9e-14 | 3.9e-14 (24) | 0 | 0 | 0 | 0.0 |
| 1/(1+25x²) on [-1, 1] | equispaced | columns | 2000 | 3 | True | 1.1e-15 | 1.4e-15 | 1.4e-15 (2) | 0 | 0 | 0 | 0.0 |
| 1/(1+25x²) on [-1, 1] | equispaced | none | 2000 | 3 | True | 1.0e-15 | 1.0e-15 | 1.0e-15 (2) | 0 | 0 | 0 | 0.0 |

Unresolved pole estimate: relative residual |d(λ)|/Σ|w_j/(λ - z_j)| > 1e-08. Real poles in the interval: certified zeros of d (sign changes), positions:
* |x| on [-1, 1], equispaced/columns: -0.055
* |x| on [-1, 1], equispaced/none: -0.000935
* |x| on [-1, 1], clustered/columns: -3.99e-17
* |x| on [-1, 1], clustered/none: -9e-16
* √x on [0, 1], clustered/none: 8.72e-06, 0.000123, 0.000857, 0.00246, 0.00405, 0.0134 (+76 more)

### Q1 root-exponential fit sensitivity (AAA, clustered Z, columns)

| function | window n | root-exp C (1st half, 2nd half) | RMS |
|---|---|---|---|
| |x| on [-1, 1] | 6–92 | 3.38 (3.41, 3.05) | 0.71 |
| |x| on [-1, 1] | 10–60 | 3.45 (3.62, 4.37) | 0.67 |
| |x| on [-1, 1] | 6–99 | 3.37 (3.44, 3.10) | 0.69 |
| √x on [0, 1] | 6–45 | 4.47 (4.55, 4.41) | 0.11 |
| √x on [0, 1] | 10–60 | 4.19 (4.45, 3.34) | 0.40 |
| √x on [0, 1] | 6–64 | 4.21 (4.52, 3.37) | 0.39 |

### Q1 AAA error / best-approximation asymptotics 8e^{-C√n} (clustered Z, columns)

* |x| on [-1, 1] (C = 3.1416): n=10: 521.8, n=15: 23.9, n=20: 11.0, n=25: 15.8, n=30: 21.5, n=35: 20.7, n=40: 272.3, n=45: 19.7, n=50: 12.5, n=55: 15.3, n=60: 12.8, n=65: 104.2, n=70: 13.5, n=75: 63.9, n=80: 3672.8, n=85: 23.0, n=90: 15.2, n=95: 35.4, n=99: 17.2
* √x on [0, 1] (C = 4.4429): n=10: 11.6, n=15: 6.0, n=20: 6.3, n=25: 6.7, n=30: 6.1, n=35: 6.3, n=40: 8.8, n=45: 10.6, n=50: 7.7, n=55: 10.3, n=60: 18.1, n=64: 28.8

### Q1 √x clustering depth (Z = 4000 equispaced ∪ 2000 log-spaced in [10^-depth, 1]; test grid down to 1e-32)

| depth | scaling | m | converged | final sample err | final test err | best test err (n) | √(smallest sample) |
|---|---|---|---|---|---|---|---|
| 1e-10 | columns | 38 | True | 4.3e-14 | 3.7e-07 | 3.7e-07 (36) | 1.0e-05 |
| 1e-10 | none | 100 | False | 1.6e-09 | 4.3e-07 | 4.3e-07 (66) | 1.0e-05 |
| 1e-15 | columns | 49 | True | 9.3e-14 | 1.4e-09 | 1.4e-09 (48) | 3.2e-08 |
| 1e-15 | none | 100 | False | 1.8e-09 | 7.0e-07 | 3.2e-09 (28) | 3.2e-08 |
| 1e-20 | columns | 57 | True | 7.5e-14 | 2.3e-08 | 6.7e-12 (55) | 1.0e-10 |
| 1e-20 | none | 100 | False | 2.0e-04 | 3.3e-05 | 2.0e-08 (24) | 1.0e-10 |
| 1e-25 | columns | 65 | True | 5.4e-14 | 2.1e-13 | 1.1e-13 (61) | 3.2e-13 |
| 1e-25 | none | 100 | False | 9.6e-04 | 6.6e-03 | 1.9e-08 (24) | 3.2e-13 |
| 1e-30 | columns | 65 | True | 8.1e-14 | 8.4e-14 | 8.4e-14 (64) | 1.0e-15 |
| 1e-30 | none | 100 | False | 3.9e-02 | 2.6e+00 | 2.9e-08 (24) | 1.0e-15 |

### Q1 Chebyshev interpolation, test error at selected degrees

| function | n=10 | n=20 | n=40 | n=100 | n=200 | n=398 | n=1000 |
|---|---|---|---|---|---|---|---|
| |x| on [-1, 1] | 5.5e-02 | 2.8e-02 | 1.5e-02 | 5.9e-03 | 3.0e-03 | 1.5e-03 | 6.0e-04 |
| √x on [0, 1] | 4.6e-02 | 2.4e-02 | 1.2e-02 | 5.0e-03 | 2.5e-03 | 1.3e-03 | 5.0e-04 |
| tanh(50x) on [-1, 1] | 7.7e-01 | 6.3e-01 | 4.2e-01 | 8.8e-02 | 4.4e-03 | 9.0e-06 | 3.8e-13 |
| 1/(1+25x²) on [-1, 1] | 1.1e-01 | 1.5e-02 | 2.9e-04 | 1.9e-09 | 2.5e-14 | 5.7e-14 | 1.6e-13 |

### Q1 smallest degree n with test error ≤ level (— = never, within n ≤ 99 for AAA, n ≤ 1000 for Chebyshev; every n checked)

| function | level | Chebyshev | AAA clustered/columns | AAA clustered/none | AAA equispaced/columns | AAA equispaced/none |
|---|---|---|---|---|---|---|
| |x| on [-1, 1] | 1e-06 | — | 39 | 36 | — | — |
| |x| on [-1, 1] | 1e-10 | — | 78 | 79 | — | — |
| |x| on [-1, 1] | 1e-13 | — | — | — | — | — |
| √x on [0, 1] | 1e-06 | — | 17 | 17 | — | — |
| √x on [0, 1] | 1e-10 | — | 38 | — | — | — |
| √x on [0, 1] | 1e-13 | — | 64 | — | — | — |
| tanh(50x) on [-1, 1] | 1e-06 | 447 | n/a | n/a | 15 | 15 |
| tanh(50x) on [-1, 1] | 1e-10 | 741 | n/a | n/a | 20 | 20 |
| tanh(50x) on [-1, 1] | 1e-13 | 961 | n/a | n/a | 24 | 24 |
| 1/(1+25x²) on [-1, 1] | 1e-06 | 70 | n/a | n/a | 2 | 2 |
| 1/(1+25x²) on [-1, 1] | 1e-10 | 116 | n/a | n/a | 2 | 2 |
| 1/(1+25x²) on [-1, 1] | 1e-13 | 150 | n/a | n/a | 2 | 2 |

Chebyshev scan check (every n ≤ 1000; first degree of the log grid in brackets):
* |x| on [-1, 1]: 1e-06: None (None), 1e-10: None (None), 1e-13: None (None); DCT vs numopt coefficients max |diff| 3.1e-14; full-grid evaluations 0
* √x on [0, 1]: 1e-06: None (None), 1e-10: None (None), 1e-13: None (None); DCT vs numopt coefficients max |diff| 1.4e-14; full-grid evaluations 0
* tanh(50x) on [-1, 1]: 1e-06: 447 (501), 1e-10: 741 (794), 1e-13: 961 (None); DCT vs numopt coefficients max |diff| 3.2e-14; full-grid evaluations 3
* 1/(1+25x²) on [-1, 1]: 1e-06: 70 (79), 1e-10: 116 (126), 1e-13: 150 (158); DCT vs numopt coefficients max |diff| 9.7e-15; full-grid evaluations 3

Chebyshev geometric fit tanh50: ρ_fit = 1.0310, ρ_theory = 1.0319, n = 10–794, RMS = 0.13 decades

Chebyshev geometric fit runge: ρ_fit = 1.2213, ρ_theory = 1.2198, n = 10–126, RMS = 0.15 decades

### Q2 Runge on n + 1 equispaced points: max error

| n | fh3 | fh4 | fh5 | fh6 | fh7 | fh8 | spline | pchip | barycentric | lagrange | chebyshev_points |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 6.9e-02 | 1.3e-01 | 2.4e-01 | 4.7e-01 | 8.7e-01 | 1.4e+00 | 2.2e-02 | 1.8e-02 | 1.9e+00 | 1.9e+00 | 1.1e-01 |
| 20 | 2.8e-03 | 5.3e-03 | 9.9e-03 | 1.9e-02 | 3.6e-02 | 7.0e-02 | 3.2e-03 | 1.3e-02 | 6.0e+01 | 6.0e+01 | 1.5e-02 |
| 40 | 4.3e-06 | 9.9e-06 | 1.8e-05 | 3.5e-05 | 6.7e-05 | 1.3e-04 | 2.8e-04 | 4.2e-03 | 1.0e+05 | 1.0e+05 | 2.9e-04 |
| 80 | 5.1e-08 | 6.0e-09 | 8.0e-10 | 2.7e-10 | 2.0e-10 | 4.5e-10 | 1.6e-05 | 1.1e-03 | 6.3e+14 | 5.5e+11 | 1.0e-07 |
| 160 | 3.0e-09 | 1.7e-10 | 1.1e-11 | 9.1e-13 | 8.7e-14 | 2.2e-14 | 9.7e-07 | 2.9e-04 | non-finite | — | 2.5e-14 |
| 320 | 1.8e-10 | 5.0e-12 | 1.7e-13 | 1.0e-14 | 8.0e-15 | 1.7e-14 | 6.0e-08 | 7.2e-05 | overflow | — | 4.2e-14 |
| 640 | 1.1e-11 | 1.5e-13 | 5.3e-15 | 5.3e-15 | 1.3e-14 | 2.1e-14 | 3.7e-09 | 1.8e-05 | non-finite | — | 1.1e-13 |
| 1280 | 7.0e-13 | 6.0e-15 | 4.8e-15 | 7.3e-15 | 1.6e-14 | 2.4e-14 | 2.3e-10 | 4.5e-06 | overflow | — | 2.3e-13 |

### Q2 observed order log2(e_n / e_2n), n → 2n

| n → 2n | fh3 | fh4 | fh5 | fh6 | fh7 | fh8 | spline | pchip |
|---|---|---|---|---|---|---|---|---|
| 10→20 | 4.6 | 4.6 | 4.6 | 4.6 | 4.6 | 4.3 | 2.8 | 0.5 |
| 20→40 | 9.4 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 3.5 | 1.6 |
| 40→80 | 6.4 | 10.7 | 14.5 | 17.0 | 18.3 | 18.1 | 4.1 | 1.9 |
| 80→160 | 4.1 | 5.2 | 6.1 | 8.2 | 11.2 | 14.3 | 4.1 | 2.0 |
| 160→320 | 4.0 | 5.1 | 6.1 | 6.4 | 3.4 | 0.4 | 4.0 | 2.0 |
| 320→640 | 4.0 | 5.0 | 5.0 | 1.0 | -0.7 | -0.4 | 4.0 | 2.0 |
| 640→1280 | 4.0 | 4.7 | 0.1 | -0.5 | -0.3 | -0.2 | 4.0 | 2.0 |

### Q2 Lebesgue constant Λ (FH, equispaced)

| n | Λ d=3 | Λ d=4 | Λ d=5 | Λ d=6 | Λ d=7 | Λ d=8 |
|---|---|---|---|---|---|---|
| 10 | 3.7 | 5.4 | 7.9 | 11.6 | 16.9 | 23.6 |
| 20 | 4.7 | 7.2 | 11.5 | 18.8 | 31.5 | 53.3 |
| 40 | 5.6 | 8.9 | 14.7 | 25.1 | 43.8 | 77.8 |
| 80 | 6.4 | 10.5 | 17.8 | 31.1 | 55.4 | 100.4 |
| 160 | 7.3 | 12.1 | 20.8 | 36.9 | 66.6 | 122.1 |
| 320 | 8.1 | 13.7 | 23.8 | 42.6 | 77.6 | 143.5 |
| 640 | 9.0 | 15.3 | 26.8 | 48.3 | 88.5 | 164.7 |
| 1280 | 9.8 | 16.8 | 29.8 | 54.0 | 99.4 | 185.8 |

### Q2 rounding floor at n = 1280: error vs ε·Λ·max|f|

| d | error | ε·Λ·max|f| | error / (ε·Λ·max|f|) |
|---|---|---|---|
| 3 | 7.0e-13 | 2.2e-15 | 318.68 |
| 4 | 6.0e-15 | 3.7e-15 | 1.60 |
| 5 | 4.8e-15 | 6.6e-15 | 0.72 |
| 6 | 7.3e-15 | 1.2e-14 | 0.60 |
| 7 | 1.6e-14 | 2.2e-14 | 0.71 |
| 8 | 2.4e-14 | 4.1e-14 | 0.58 |

Λ growth for n = 160, d = 10..30: log2 Λ grows by 0.95 per unit d (Λ = 424 at d = 10, 2.8e+05 at d = 20, 2.18e+08 at d = 30)

### FH 2007 Table 2, n = 160, d = 10, 1/(1+x²) on [-5, 5]: paper 1.3e-15; Λ = 424.1, ε·Λ = 9.4e-14; long double eps 1.9e-34

| equispaced test points | shared with nodes | float64 error | same weights in long double (truncation) |
|---|---|---|---|
| 101 | 21 | 7.8e-16 | 4.5e-17 |
| 201 | 41 | 8.1e-15 | 7.3e-17 |
| 501 | 21 | 1.5e-14 | 3.3e-16 |
| 1001 | 41 | 1.5e-14 | 3.3e-16 |
| 2001 | 81 | 1.6e-14 | 3.3e-16 |
| 10001 | 81 | 4.1e-14 | 3.3e-16 |
| 100001 | 161 | 6.0e-14 | 3.3e-16 |

### FH 2007 Table 3 spline column (1/(1+x²) on [-5, 5], 100 001 test points)

| n | paper (clamped) | clamped (SciPy) | not-a-knot (numopt) |
|---|---|---|---|
| 10 | 2.2e-02 | 2.2e-02 | 2.2e-02 |
| 20 | 3.2e-03 | 3.2e-03 | 3.2e-03 |
| 40 | 2.8e-04 | 2.8e-04 | 2.8e-04 |
| 80 | 1.6e-05 | 1.6e-05 | 1.6e-05 |
| 160 | 9.5e-07 | 9.7e-07 | 9.7e-07 |
| 320 | 5.9e-08 | 6.0e-08 | 6.0e-08 |
| 640 | 3.7e-09 | 3.7e-09 | 3.7e-09 |

Diagnostics: √x clustered, unscaled AAA at m = 24: column-norm ratio 9.2e+13, κ(A) = 4.7e+21, κ(AD) = 5.2e+08. tanh(50x) poles (24), relative error vs long-double Newton roots: numpy route max 6.3e-10 / median 2.1e-11; QZ max 2.8e-09 / median 3.1e-10. Degree-80 equispaced polynomial for Runge in long double: max error 5.46e+11.

### Q2 best d (d ≤ min(n, 30))

| n | best d | error | Λ at best d | error at d = 3 | error at d = 8 |
|---|---|---|---|---|---|
| 10 | 0 | 3.6e-02 | 2.3 | 6.9e-02 | 1.4e+00 |
| 20 | 1 | 1.5e-03 | 2.7 | 2.8e-03 | 7.0e-02 |
| 40 | 3 | 4.3e-06 | 5.6 | 4.3e-06 | 1.3e-04 |
| 80 | 7 | 2.0e-10 | 55.4 | 5.1e-08 | 4.5e-10 |
| 160 | 8 | 2.2e-14 | 122.1 | 3.0e-09 | 2.2e-14 |
