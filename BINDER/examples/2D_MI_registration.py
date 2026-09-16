import numpy as np
from BINDER import FastRegistration
from PIL import Image
import time
import os
from BINDER import BINDERDIR

fixed = np.array(Image.open(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "2DTestImages", "shapes.jpg")).convert('L'))
moving = np.array(Image.open(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "2DTestImages", "shapesWarped.jpg")).convert('L'))
num_bins = 32

fixed = np.array(fixed, dtype=float)
moving = np.array(moving, dtype=float)
resolutions = [8, 4, 2, 1]
n_iterations = 100
n_burnins = 0
n_samples = 1000
gamma = 1e-4
visualizer = True
threads = 8

registrator = FastRegistration(voxels=fixed,
                               nodes=moving,
                               num_voxel_intesity_levels=num_bins,
                               num_node_intensity_levels=num_bins,
                               gamma=gamma,
                               visualizer=visualizer,
                               threads=threads,
                               spline_order=3,
                               metric='MI',
                               smooth=False,
                               topology_correction=False,
                               )

print("Start")
t = time.time()
_ = registrator.run(resolutions=resolutions, max_number_of_iterations=n_iterations)
elapsed = time.time() - t
print("Time for EM: " + str(elapsed))

_, _, = registrator.sample_param(n_burnins, n_samples, sample_gamma=True)
