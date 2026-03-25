# pygame loop, input handling, and rendering
# Based on Stam (2003) with vorticity confinement from Fedkiw et al. (2001)

"""
Controls:
  Left mouse drag   — inject dye + velocity
  Right mouse drag  — inject velocity only (invisible wind)
  Scroll wheel      — adjust brush size
  R                 — reset simulation
  V                 — toggle velocity field overlay
  C                 — cycle dye colour mode
  +/-               — adjust vorticity confinement strength
  P                 — pause / resume
  ESC               — quit
"""

import sys
import math
import numpy as np
import pygame
from solver import step

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N        = 256          # Grid resolution (N×N interior cells). 128 = fast, 256 = detailed
DT       = 0.12         # Timestep
DIFF     = 0.0          # Dye diffusion (0 = sharp dye edges)
VISC     = 0.0000001    # Fluid viscosity (water-like)
FORCE    = 300.0        # Velocity injection strength
SOURCE   = 80.0         # Dye injection strength
VORT_EPS = 6.0          # Vorticity confinement strength (0 = off)

