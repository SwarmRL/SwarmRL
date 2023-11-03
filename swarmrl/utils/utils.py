"""
Utils for the SwarmRL package.
"""

import logging
import os
import pickle
import shutil
import typing

import jax.numpy as jnp
import numpy as np
import pint

import swarmrl
from swarmrl.models.interaction_model import Colloid


def get_random_angles(rng: np.random.Generator):
    # https://mathworld.wolfram.com/SpherePointPicking.html
    return np.arccos(2.0 * rng.random() - 1), 2.0 * np.pi * rng.random()


def vector_from_angles(theta, phi):
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def angles_from_vector(director):
    director /= np.linalg.norm(director)
    theta = np.arccos(director[2])
    phi = np.arctan2(director[1], director[0])
    return theta, phi


def write_params(
    folder_name: str,
    sim_name: str,
    params: typing.Any,
    write_espresso_version: bool = False,
):
    """
    Writes parameters human-readable and to pickle

    Parameters
    ----------
    folder_name
        Folder to save to
    sim_name
        Name of the simulation, used to create the two file names
    params : dict
        The parameters to be saved. Should have a string representation for txt output
         and be serializable by pickle
    write_espresso_version : bool
        If True, the espresso version will be printed to the txt file

    """

    fname_base = f"{folder_name}/params_{sim_name}"
    with open(fname_base + ".txt", "w") as txt_file:
        if write_espresso_version:
            from espressomd import version

            txt_file.write(
                f"Espresso version {version.friendly()} branch {version.git_branch()} "
                f"at commit {version.git_commit()}\n"
            )
        txt_file.write(str(params))

    with open(fname_base + ".pick", "wb") as pick_file:
        pickle.dump(params, pick_file)


def setup_sim_folder(
    outfolder_base: str,
    name: str,
    ask_if_exists: bool = True,
    delete_existing: bool = True,
):
    """
    Create a simulation folder.
    Depending on flags, delete previous folders of the same name
    Parameters
    ----------
    outfolder_base
        Folder in which to create the new simulation folder
    name
        Name of the new folder
    ask_if_exists
        Flag to determine if the program stops to await user input on
        how to handle if the folder already exists (true, default)
        or to just delete it (false)
    delete_existing
        Whether to delete the existing folder

    Returns
    -------

    """
    folder_name = f"{outfolder_base}/{name}"
    if os.path.isdir(folder_name):
        if (
            ask_if_exists
            and input(
                f"Directory for sim '{name}' already exists in '{outfolder_base}'. "
                "Delete previous and create new? (yes/N) "
            )
            != "yes"
        ):
            print("aborting")
            exit()
        elif delete_existing:
            shutil.rmtree(folder_name)
            print(f"removed {folder_name} and all its contents")

    os.makedirs(folder_name, exist_ok=True)
    print(f"outdir {folder_name} created")

    return folder_name


def setup_swarmrl_logger(
    filename: str,
    loglevel_terminal: typing.Union[int, str] = logging.INFO,
    loglevel_file: typing.Union[int, str] = logging.DEBUG,
) -> logging.Logger:
    """
    Configure the swarmrl logger. This logger is used internally for logging,
    but you can also use it for your own log messages.
    Parameters
    ----------
    filename
        Name of the file where logs get written to
    loglevel_terminal
        Loglevel of the terminal output. The values correspond to
        https://docs.python.org/3/library/logging.html#logging-levels.
        You can pass an integer (or logging predefined values such as logging.INFO)
        or a string that corresponds to the loglevels of the link above.
    loglevel_file
        Loglevel of the file output.
    Returns
    -------
        The logger
    """

    def get_numeric_level(loglevel: typing.Union[int, str]):
        if isinstance(loglevel, str):
            numeric_level = getattr(logging, loglevel.upper(), None)
        elif isinstance(loglevel, int):
            numeric_level = loglevel
        else:
            raise ValueError(
                f"Invalid log level: {loglevel}. Must be either str or int"
            )
        return numeric_level

    logger = logging.getLogger(swarmrl._ROOT_NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(levelname)-10s] %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(get_numeric_level(loglevel_file))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(get_numeric_level(loglevel_terminal))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def gather_n_dim_indices(reference_array: np.ndarray, indices: np.ndarray):
    """
    Gather entries from an n_dim array using an n_dim index array

    Parameters
    ----------
    reference_array : np.ndarray
            Array that you want to gather the indices of.
    indices : np.ndarray
            Indices in the same shape as the array without the last dimension.

    Returns
    -------
    reduced_array : np.ndarray
            Shape is the reference array with the last dimension reduced to 1.
            This array is the initial reference array with the desired indices chosen
            out.
    """
    indices = indices.astype(int)
    reference_shape = reference_array.shape

    multiplier = (
        np.linspace(0, len(indices.flatten()) - 1, len(indices.flatten()), dtype=int)
        * reference_shape[-1]
    )

    indices = indices.flatten() + multiplier

    gathered_array = reference_array.flatten()[indices]

    return gathered_array.reshape(reference_shape[0], reference_shape[1])


def record_trajectory(
    particle_type: str,
    features: np.ndarray,
    actions: np.ndarray,
    log_probs: np.ndarray,
    rewards: np.ndarray,
):
    """
    Record trajectory if required.

    Parameters
    ----------
    particle_type : str
            Type of the particle saved. Important for the multi-species training.
    rewards : np.ndarray (n_timesteps, n_particles, 1)
            Rewards collected during the simulation to be used in training.
    log_probs : np.ndarray (n_timesteps, n_particles, 1)
            log_probs used for debugging.
    features : np.ndarray (n_timesteps, n_particles, n_dimensions)
            Features to store in the array.
    actions : np.ndarray (n_timesteps, n_particles, 1)
            A numpy array of actions

    Returns
    -------
    Dumps a hidden file to disc which is often removed after reading.
    """
    try:
        data = np.load(f".traj_data_{particle_type}.npy", allow_pickle=True)
        feature_data = data.item().get("features")
        action_data = data.item().get("actions")
        log_probs_data = data.item().get("log_probs")
        reward_data = data.item().get("rewards")

        feature_data = np.append(feature_data, np.array([features]), axis=0)
        action_data = np.append(action_data, np.array([actions]), axis=0)
        log_probs_data = np.append(log_probs_data, np.array([log_probs]), axis=0)
        reward_data = np.append(reward_data, np.array([rewards]), axis=0)

        os.remove(f".traj_data_{particle_type}.npy")

    except FileNotFoundError:
        feature_data = np.array([features])
        action_data = np.array([actions])
        log_probs_data = np.array([log_probs])
        reward_data = np.array([rewards])

    np.save(
        f".traj_data_{particle_type}.npy",
        {
            "features": feature_data,
            "actions": action_data,
            "log_probs": log_probs_data,
            "rewards": reward_data,
        },
        allow_pickle=True,
    )


def save_memory(memory: dict):
    """
    Records the training data if required.

    Parameters:
    ----------
    memory : a dictionary containing the data from the method where it is called from.
        The data is specified in the method.
        It has to contain a key "file_name" which is the name of the file to be saved.
        To handle multiple particle types: one can specify the file name in the
        initialisation of the method.

    Returns
    -------
    Dumps a  file to disc to evaluate training.
    """
    empty_memory = {key: [] for key in memory.keys()}
    empty_memory["file_name"] = memory["file_name"]
    try:
        reloaded_dict = np.load(memory["file_name"], allow_pickle=True).item()
        for key, _ in reloaded_dict.items():
            reloaded_dict[key].append(memory[key])
        np.save(memory["file_name"], reloaded_dict, allow_pickle=True)
    except FileNotFoundError:
        for key, _ in empty_memory.items():
            empty_memory[key].append(memory[key])
        np.save(memory["file_name"], empty_memory, allow_pickle=True)
    return empty_memory


def calc_signed_angle_between_directors(
    my_director: np.ndarray, other_director: np.ndarray
) -> float:
    """
    In 2D compare two different normalized
    directors to determine the angle between them

    Parameters
    ----------
    my_director : np.ndarray
            Normalized director in 3D.
    other_director : np.ndarray
            Normalized director in 3D.
    Returns
    ----------
    signed_angle : float
        signed float which represents the signed angle of my_director to other_director
        with the mathematical sign convention.
    """

    # Assert if the directors were really normalized
    my_director /= jnp.linalg.norm(my_director)
    other_director /= jnp.linalg.norm(other_director)

    # calculate the angle in which the my_colloid is looking
    angle = jnp.arccos(jnp.clip(jnp.dot(other_director, my_director), -1.0, 1.0))
    # use the director in orthogonal direction to determine sign
    orthogonal_dot = jnp.dot(
        other_director,
        jnp.array([-my_director[1], my_director[0], my_director[2]]),
    )
    # don't use np.sign instead use np.where because
    # np.sign(0) => 0 is not what we want
    angle *= jnp.where(orthogonal_dot >= 0, 1, -1)

    return angle


def create_colloids(
    n_cols: int,
    type_: int = 0,
    center: np.array = np.array([500, 500, 0]),
    dist: float = 200.0,
    face_middle: bool = False,
):
    """
    Create a number of colloids in a circle around the center of the box.
    This method is primarily used for writing tests. It is not used in the
    actual simulation. The colloids are created in a circle around the center
    of the box.

    Parameters
    ----------
    n_cols : int
            Number of colloids to create.
    type_ : int, optional
            Type of the colloids to create.
    center : np.ndarray, optional
            Center of the circle in which the colloids are created.
    dist : float, optional
            Distance of the colloids to the center.
    face_middle : bool, optional
            If True, the colloids face the center of the circle.

    Returns
    -------
    colloids : list(Colloid)
            List of colloid.
    """
    cols = []
    for i in range(n_cols):
        theta = np.random.random(1)[0] * 2 * np.pi
        position = center + dist * np.array([np.cos(theta), np.sin(theta), 0])
        if face_middle:
            direction = np.array(center - position)
        else:
            direction = np.random.random(3)
        direction[-1] = 0
        direction = direction / np.linalg.norm(direction)
        cols.append(Colloid(pos=position, director=direction, type=type_, id=i))
    return cols


def calc_ellipsoid_friction_factors_translation(
    axial_semiaxis, equatorial_semiaxis, dynamic_viscosity
):
    """
    https://link.springer.com/article/10.1007/BF02838005

    Returns
    -------

    gamma_ax, gamma_eq
        The friction coefficient for dragging the ellipsoid along its symmetry axis
        and along one of the equatorial axes

    """
    if axial_semiaxis > equatorial_semiaxis:
        # prolate spheroid
        a = axial_semiaxis
        b = equatorial_semiaxis
        e = np.sqrt(1 - b**2 / a**2)
        ll = np.log((1 + e) / (1 - e))
        gamma_ax = 16 * np.pi * dynamic_viscosity * a * e**3 / ((1 + e**2) * ll - 2 * e)
        gamma_eq = (
            32 * np.pi * dynamic_viscosity * a * e**3 / (2 * e + (3 * e**2 - 1) * ll)
        )
    else:
        # oblate spheroid
        b = axial_semiaxis
        a = equatorial_semiaxis
        e = np.sqrt(1 - b**2 / a**2)
        gamma_ax = (
            8
            * np.pi
            * dynamic_viscosity
            * a
            * e**3
            / (e * np.sqrt(1 - e**2) - (1 - 2 * e**2) * np.arcsin(e))
        )
        gamma_eq = (
            16
            * np.pi
            * dynamic_viscosity
            * a
            * e**3
            / (-e * np.sqrt(1 - e**2) + (1 + 2 * e**2) * np.arcsin(e))
        )

    return gamma_ax, gamma_eq


def calc_ellipsoid_friction_factors_rotation(
    axial_semiaxis, equatorial_semiaxis, dynamic_viscosity
):
    """
    https://en.wikipedia.org/wiki/Perrin_friction_factors

    Returns
    -------

    gamma_ax, gamma_eq:
        The friction factors for rotation around the axial (symmetry) axis
        and for rotation around one of the equatorial axes
    """
    p = axial_semiaxis / equatorial_semiaxis
    xi = np.sqrt(np.abs(p**2 - 1)) / p

    if p > 1:
        S = 2 * np.arctanh(xi) / xi
    else:
        S = 2 * np.arctan(xi) / xi

    f_ax = 4.0 / 3.0 * (p**2 - 1) / (2 * p**2 - S)
    f_eq = 4.0 / 3.0 * (p**-2 - p**2) / (2 - S * (2 - p**-2))

    gamma_sphere = (
        8 * np.pi * dynamic_viscosity * axial_semiaxis * equatorial_semiaxis**2
    )

    return gamma_sphere * f_ax, gamma_sphere * f_eq


def convert_array_of_pint_to_pint_of_array(array_of_pint, ureg: pint.UnitRegistry):
    units = [val.units for val in array_of_pint]
    # np.unique doesn't work so we have to do it manually
    unit = units[0]
    for u in units:
        if u != unit:
            raise ValueError(f"The values in the array have different units: {units}")

    return ureg.Quantity(np.array([val.m_as(unit) for val in array_of_pint]), unit)
