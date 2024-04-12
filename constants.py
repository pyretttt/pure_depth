from dataclasses import dataclass

@dataclass
class Namespace:
  name: str

# CONSTANTS
SEED = 42
EPSILON = 1e-8

SCALE_INVARIANCE_ALPHA = 0.5


RGB = Namespace(name='RGB')
RGB.mean = (0.4644, 0.3905, 0.3726) # Derived with utils.py
RGB.std = (0.2412, 0.2432, 0.2532) # Derived with utils.py

NYU_V2 = Namespace(name='NYU_V2')
NYU_V2.max = 10.0

NYU_V2_SAFE_DIM = (448, 608)
NYU_V2_SOURCE_DIM = (480, 640)
INPUT_SIZE=(224, 320)