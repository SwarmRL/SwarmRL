import h5py as hf
import numpy as np
import znvis as vis

if __name__ == "__main__":
    """
    Run the simple spheres example.
    """
    material_1 = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.6)
    # Define the first particle.
    with hf.File("trajectory.hdf5") as db:
        trajectory = db["colloids"]["Unwrapped_Positions"][:]

    print(trajectory.shape)

    mesh = vis.Sphere(radius=20.0, resolution=5, material=material_1)
    particle = vis.Particle(
        name="Blue", mesh=mesh, position=trajectory, smoothing=False
    )

    # Construct the visualizer and run
    visualizer = vis.Visualizer(particles=[particle], frame_rate=80)
    visualizer.run_visualization()
