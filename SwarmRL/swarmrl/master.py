from engine import espresso
from models import dummy_models
import pint
import numpy as np

if __name__ == '__main__':
    com_force_rule = dummy_models.ToCenterMass(1000, vision_angle=np.pi / 2.)

    ureg = pint.UnitRegistry()
    params = espresso.MDParams(n_colloids=10,
                               ureg=ureg,
                               colloid_radius=ureg.Quantity(1, 'micrometer'),
                               fluid_dyn_viscosity=ureg.Quantity(8.9, 'pascal * second'),
                               WCA_epsilon=ureg.Quantity(1e-20, 'joule'),
                               colloid_density=ureg.Quantity(2.65, 'gram / centimeter**3'),
                               temperature=ureg.Quantity(300, 'kelvin'),
                               box_length=ureg.Quantity(100, 'micrometer'),
                               time_step=ureg.Quantity(0.05, 'second'),
                               time_slice=ureg.Quantity(0.1, 'second'),
                               write_interval=ureg.Quantity(0.5, 'second'))

    output_folder = '../test_sim/'

    system_runner = espresso.EspressoMD(params, seed=42, out_folder=output_folder, write_chunk_size=1000)
    system_runner.setup_simulation()

    for i in range(10):
        system_runner.integrate(500, com_force_rule)
        data_for_ML_trainer = system_runner.get_particle_data()
        com_force_rule.forward(None, None)
