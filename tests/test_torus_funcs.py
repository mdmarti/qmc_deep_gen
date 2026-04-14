from analysis.model_helpers import torus_forward, torus_reverse
import numpy as np


ds = range(1, 11)
gen = np.random.default_rng()
n_samples = 100

for d in ds:
    latent_samples = gen.random(size=(n_samples, d))

    embedded = torus_forward(latent_samples)
    unbedded = torus_reverse(embedded)

    assert np.all(np.isclose(latent_samples, unbedded))


print("All samples close!")
