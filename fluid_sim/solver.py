# numba-accelerated solver based on Stam's paper
# Fluid solver based on Jos Stam's "Real-Time Fluid Dynamics for Games" (2003).
# add source -> diffuse -> advect (convey by horizontal movement of fluid) -> project

import numpy as np
from numba import njit

# boundary conditions
# b = type of field (b=1 -> x-velocity, b=2 -> y-velocity, b=0 -> pressure or dye)
# when b = 1 (x-velocity): fluid cannot pass thru left and right walls -> normal velocity = 0
@njit(cache = True)
def set_boundary(b, x, N):
    for i in range(1, N + 1):
        x[0,     i] = -x[1,     i] if b == 1 else x[1,     i]
        x[N + 1, i] = -x[N,     i] if b == 1 else x[N,     i]
        x[i,     0] = -x[i,     1] if b == 2 else x[i,     1]
        x[i, N + 1] = -x[i,     N] if b == 2 else x[i,     N]

    # Corners — average of neighbours
    x[0,     0    ] = 0.5 * (x[1,     0    ] + x[0,     1    ])
    x[0,     N + 1] = 0.5 * (x[1,     N + 1] + x[0,     N    ])
    x[N + 1, 0    ] = 0.5 * (x[N,     0    ] + x[N + 1, 1    ])
    x[N + 1, N + 1] = 0.5 * (x[N,     N + 1] + x[N + 1, N    ])

# linear solver - Gauss-Seidel relaxation
@njit(cache = True)
def lin_solve(b, x, x0, a, c, N, iterations = 20):
    """
    Solve  (I - a * Laplacian) x = x0  iteratively.
    Used for both diffusion and pressure projection.
    """
    inv_c = 1.0 / c
    for _ in range(iterations):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                x[i, j] = (x0[i, j] + a * (
                    x[i - 1, j] + x[i + 1, j] +
                    x[i,     j - 1] + x[i,     j + 1]
                )) * inv_c
        set_boundary(b, x, N)