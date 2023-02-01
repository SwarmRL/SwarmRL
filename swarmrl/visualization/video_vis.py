import os
import pickle
import random

import h5py
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

# derivative of modified bessel Function of second kind of real order
# modivied bessel function of second kind of real order v
from scipy.special import kv, kvp


def angle_from_vector(vec) -> float:
    return np.arctan2(vec[1], vec[0])


def calc_chemical_potential(chemical_pos, measure_pos):
    """
    This function implements the solution to the 2D diffusion differential equation
    with an additional decay term. Because of and constant source in the origin
    a decay of the density is needed to yield a steady state in the
    diffusion equation. The solution is the modified bessel function of second kind.


    Parameters
    ----------
    chemical_pos : np.array (2)
            x and y coordinate of chemical source

    col_pos : np.array (n_position , 2)
            x and y coordinate of measure point. Simultaneously evalute n_positions.


    Returns
    -------
    chemical_magnitude : np.array (n_position , 1)
            magnitude of the chemical potential

    chemical_gradient : np.array (n_position , 2)
            gradient of the chemical potential in length units defined by input values
    """
    distance = np.linalg.norm(np.array(chemical_pos) - np.array(measure_pos), axis=-1)
    distance = np.where(distance == 0, 5, distance)
    # prevent zero division
    direction = (chemical_pos - measure_pos) / np.stack([distance] * 2, axis=-1)
    #  parameters with arbitrary units currently set to aim amplitude \approx 1
    diffusion_coefficient = 670  # glucose in water in mum^2/s
    density_decay_per_second = 0.4
    amount_per_second_from_source = 1800  # chemics
    rescale_fac = np.sqrt(density_decay_per_second / diffusion_coefficient)

    # const = -2 * np.pi * diffusion_coefficient * radius
    # * rescale_fac * kvp(0, rescale_fac * radius )
    # reuse already calculated value
    const = 4182.925639571625
    # amount_per_second_from_source =const * amplitude
    # the const sums over the chemics that flow throw an imaginary boundary at radius
    # Amplitude of the potential i.e. the chemical density
    # (google "first ficks law" for theoretical info)
    amplitude = amount_per_second_from_source / const

    rescaled_distance = rescale_fac * distance
    chemical_magnitude = amplitude * kv(0, rescaled_distance)
    chemical_gradient = (
        -np.stack([amplitude * rescale_fac * kvp(0, rescaled_distance)] * 2, axis=-1)
        * direction
    )
    return chemical_magnitude, chemical_gradient


class Animations:
    def __init__(
        self,
        fig,
        ax,
        positions,
        directors,
        times,
        ids,
        types,
        vision_cone_boolean,
        cone_radius,
        n_cones,
        cone_half_angle,
        trace_boolean,
        trace_fade_boolean,
        eyes_boolean,
        radius_col,
        schmell_boolean,
        schmell_ids,
        maze_boolean,
    ):
        self.color_index = None
        self.x_1 = None
        self.x_0 = None
        self.y_1 = None
        self.y_0 = None
        self.fig = fig
        self.ax = ax
        self.written_info_data = None
        self.positions = positions
        self.directors = directors
        self.times = times
        self.ids = ids
        self.types = types
        self.vision_cone_data = None
        self.vision_cone_data_frame = None

        self.vision_cone_boolean = [False] * len(self.ids)
        self.cone_radius = [0] * len(self.ids)
        self.n_cones = 1
        self.cone_half_angle = [0] * len(self.ids)
        self.trace_boolean = [False] * len(self.ids)
        self.trace_fade_boolean = [False] * len(self.ids)
        self.eyes_boolean = [False] * len(self.ids)
        self.radius_col = [0] * len(self.ids)
        # schmell means as much as  chemical potential
        self.schmell_boolean = [False] * len(self.ids)
        self.maze_boolean = False
        self.maze_points = {}
        self.wall_thickness = 42
        self.maze_walls = []

        self.trace = [0] * len(self.ids)

        self.part_body = [0] * len(self.ids)
        self.part_lefteye = [0] * len(self.ids)
        self.part_righteye = [0] * len(self.ids)
        self.time_annotate = [0]
        self.schmell = [0]
        self.written_info = [0]
        self.maze = []
        self.set_gif_options(
            vision_cone_boolean,
            cone_radius,
            n_cones,
            cone_half_angle,
            trace_boolean,
            trace_fade_boolean,
            eyes_boolean,
            radius_col,
            schmell_boolean,
            schmell_ids,
            maze_boolean,
        )

    def set_gif_options(
        self,
        vision_cone_boolean,
        cone_radius,
        n_cones,
        cone_half_angle,
        trace_boolean,
        trace_fade_boolean,
        eyes_boolean,
        radius_col,
        schmell_boolean,
        schmell_ids,
        maze_boolean,
    ):
        # Adjust in for loop what you want (default is False)
        # and place parameters that you have
        possible_types = list(dict.fromkeys(self.types))
        for i in range(len(self.ids)):
            if self.types[i] == possible_types[0]:
                # type=0 correspond to normal colloids
                # print(possible_types[0])
                self.vision_cone_boolean[i] = vision_cone_boolean[self.types[i]]
                self.cone_radius[i] = cone_radius
                self.n_cones = n_cones
                #  different Cone numbers for different particles is not yet supported
                self.cone_half_angle[i] = cone_half_angle
                self.trace_boolean[i] = trace_boolean[self.types[i]]
                self.trace_fade_boolean[i] = trace_fade_boolean[self.types[i]]
                self.eyes_boolean[i] = eyes_boolean[self.types[i]]
                self.radius_col[i] = radius_col[self.types[i]]
            elif self.types[i] == possible_types[1]:
                # type=1 correspond to rod colloids
                # print(possible_types[1])
                self.vision_cone_boolean[i] = vision_cone_boolean[self.types[i]]
                self.trace_boolean[i] = trace_boolean[self.types[i]]
                self.trace_fade_boolean[i] = trace_fade_boolean[self.types[i]]
                self.eyes_boolean[i] = eyes_boolean[self.types[i]]
                self.radius_col[i] = radius_col[self.types[i]]
                if int(self.ids[i]) in schmell_ids:
                    # those ids correspond to chemical emitting colloids
                    self.schmell_boolean[i] = schmell_boolean
            elif self.types[i] == possible_types[2]:
                # type=3 correspond to whatever colloids
                # print(possible_types[2])
                self.vision_cone_boolean[i] = vision_cone_boolean[self.types[i]]
                self.trace_boolean[i] = trace_boolean[self.types[i]]
                self.trace_fade_boolean[i] = trace_fade_boolean[self.types[i]]
                self.eyes_boolean[i] = eyes_boolean[self.types[i]]
            else:
                raise Exception("unknown colloid type in visualisation")

        self.maze_boolean = maze_boolean  # type=2 corresponds to wall particles

    def animation_plt_init(self):
        # calc figure limits
        Delta_max_x = np.max(self.positions[:, :, 0].magnitude) - np.min(
            self.positions[:, :, 0].magnitude
        )
        mean_x = (
            np.min(self.positions[:, :, 0].magnitude)
            + np.max(self.positions[:, :, 0].magnitude)
        ) / 2
        delta_max_y = np.max(self.positions[:, :, 1].magnitude) - np.min(
            self.positions[:, :, 1].magnitude
        )
        mean_y = (
            np.min(self.positions[:, :, 1].magnitude)
            + np.max(self.positions[:, :, 1].magnitude)
        ) / 2
        max_region = max(Delta_max_x, delta_max_y) * 1.2
        self.x_0 = mean_x - max_region / 2
        self.x_1 = mean_x + max_region / 2
        self.y_0 = mean_y - max_region / 2
        self.y_1 = mean_y + max_region / 2
        self.ax.set_xlim(self.x_0, self.x_1)
        self.ax.set_ylim(self.y_0, self.y_1)
        l_units = self.positions.units
        self.ax.set_xlabel(f"x-position in ${l_units:~L}$")
        self.ax.set_ylabel(f"y-position in ${l_units:~L}$")
        self.ax.grid(True)

        if self.written_info_data is not None:
            if len(self.written_info_data) != len(self.times):
                raise Exception(
                    "In Visualization written_info_data is "
                    + str(len(self.written_info_data))
                    + " long and self.times is "
                    + str(len(self.times))
                )

        # prepare colors for trace fade and schmell
        self.color_names = []
        for color_name, color in colors.TABLEAU_COLORS.items():
            self.color_names.append(color_name)
        norm = plt.Normalize(0, 1)
        self.color_index = [random.randint(1, 9) for _ in range(len(self.ids))]
        self.mycolor = [0] * len(self.ids)

        for i in range(len(self.ids)):
            cfade = colors.to_rgb(self.color_names[self.color_index[i]]) + (0.0,)
            self.mycolor[i] = colors.LinearSegmentedColormap.from_list(
                "my", [cfade, self.color_names[self.color_index[i]]]
            )

        # prepare  schmell
        self.schmell_N = 50
        cfade = colors.to_rgb(self.color_names[4]) + (0.0,)
        self.schmellcolor = colors.LinearSegmentedColormap.from_list(
            "my", [cfade, self.color_names[4]]
        )

        # prepare vision cones
        if self.n_cones < 1:  # default =1
            print(
                "You have choosen zero vision cones (self.n_cones<1), I made"
                " self.vision_cone_boolean[i]=False because no vision is what you"
                " want.  "
            )
            self.vision_cone_boolean = [False] * len(self.ids)
            self.n_cones = 1
        self.cone = [0] * (self.n_cones * len(self.ids))

        if (
            self.vision_cone_boolean != [False] * len(self.ids)
            and self.vision_cone_data is None
        ):
            print(
                "You haven't set self.vision_cone_boolean[i] !=0 for some i, i.e. you"
                " want vision cones but no self.vision_cone_data in the options are"
                " provided. Then default is created"
            )
            self.vision_cone_data = [
                [
                    [id] + [0.2 for _ in range(self.n_cones)]
                    for id in range(len(self.ids))
                ]
                for _ in range(len(self.times))
            ]

        if (
            self.vision_cone_boolean != [False] * len(self.ids)
            and self.vision_cone_data is not None
        ):
            if len(self.vision_cone_data) != len(self.times):
                raise Exception(
                    "vision_cone_data is "
                    + str(len(self.vision_cone_data))
                    + " long and self.times is "
                    + str(len(self.times))
                )
            if len(self.vision_cone_data[0]) >= len(self.ids):
                raise Exception(
                    "vision_cone_data[0] is "
                    + str(len(self.vision_cone_data[0]))
                    + " long and the number of colloids is "
                    + str(len(self.ids))
                )
            if len(self.vision_cone_data[0][0]) != 2:
                raise Exception(
                    "vision_cone_data[0][0] is "
                    + str(len(self.vision_cone_data[0][0]))
                    + " long and  it should be 2 containing the colloid.id and the"
                    " vision data per frame, per colloid in a np.array of all types"
                    " visible."
                )
            if len(self.vision_cone_data[0][0][1]) != self.n_cones:
                raise Exception(
                    "vision_cone_data[0][0][1] is "
                    + str(len(self.vision_cone_data[0][0][1]))
                    + " long and the number of colloids is "
                    + str(self.n_cones)
                )

        # expand vision cone data
        if self.vision_cone_boolean != [False] * len(self.ids):
            self.vision_cone_data_frame = [
                [[0 for _ in range(self.n_cones)] for _ in range(len(self.ids))]
                for _ in range(len(self.times))
            ]
            maximum = 0
            for frame in range(
                len(self.times)
            ):  # not correct for self.ids with holes,...
                for c_id in range(len(self.ids)):
                    for given_c_id in range(len(self.vision_cone_data[frame])):
                        if c_id == self.vision_cone_data[frame][given_c_id][0]:
                            if maximum < max(
                                self.vision_cone_data[frame][given_c_id][1][:, 1]
                            ):
                                maximum
                            self.vision_cone_data_frame[frame][
                                c_id
                            ] = self.vision_cone_data[frame][given_c_id][1][:, 1]
                            # change the last 1 to 0 will select
                            # the vision of particles of type 0
            if np.max(self.vision_cone_data_frame) != 0:
                self.vision_cone_data_frame /= 2 * np.max(
                    self.vision_cone_data_frame
                )  # color adjustment
                self.vision_cone_data_frame += 0.1

        partN = len(self.ids)
        for i in range(partN):
            self.part_body[i] = self.ax.add_patch(
                patches.Circle(
                    xy=(0, 0),
                    radius=self.radius_col[i],
                    alpha=0.7,
                    color="g",
                    zorder=partN + partN * self.n_cones + i,
                )
            )
            if self.trace_boolean[i] and not self.trace_fade_boolean[i]:
                (self.trace[i],) = self.ax.plot([], [], zorder=i + 1)
            elif self.trace_boolean[i] and self.trace_fade_boolean[i]:
                lc = LineCollection(
                    [[[0, 0], [0, 0]]], norm=norm, cmap=self.mycolor[i], zorder=i + 1
                )
                self.trace[i] = self.ax.add_collection(lc)
            elif not self.trace_boolean[i]:
                (self.trace[i],) = self.ax.plot([], [], zorder=i + 1, alpha=0)
            else:
                raise Exception("self.trace_boolean is neither True nor False")

            if self.vision_cone_boolean[i]:
                for j in range(self.n_cones):
                    self.cone[i * self.n_cones + j] = self.ax.add_patch(
                        patches.Wedge(
                            center=(0, 0),
                            r=self.cone_radius[i],
                            theta1=0,
                            theta2=0,
                            alpha=0.2,
                            color="r",
                            zorder=partN + i * self.n_cones + j,
                        )
                    )
            else:
                for j in range(self.n_cones):
                    self.cone[i * self.n_cones + j] = self.ax.add_patch(
                        patches.Wedge(center=(0, 0), r=42, theta1=0, theta2=0, alpha=0)
                    )
            if self.eyes_boolean[i]:
                self.part_lefteye[i] = self.ax.add_patch(
                    patches.Circle(
                        xy=(0, 0),
                        radius=self.radius_col[i] / 5,
                        alpha=0.7,
                        color="k",
                        zorder=partN * 2 + partN * self.n_cones + 2 * i,
                    )
                )
                self.part_righteye[i] = self.ax.add_patch(
                    patches.Circle(
                        xy=(0, 0),
                        radius=self.radius_col[i] / 5,
                        alpha=0.7,
                        color="k",
                        zorder=partN * 2 + partN * self.n_cones + 1 + 2 * i,
                    )
                )
            else:
                self.part_lefteye[i] = self.ax.add_patch(
                    patches.Circle(xy=(0, 0), radius=42, alpha=0)
                )
                self.part_righteye[i] = self.ax.add_patch(
                    patches.Circle(xy=(0, 0), radius=42, alpha=0)
                )

        if self.schmell_boolean != [False] * len(self.ids):
            self.X, self.Y = np.mgrid[
                self.x_0 : self.x_1 : complex(0, self.schmell_N),
                self.y_0 : self.y_1 : complex(0, self.schmell_N),
            ]
            self.testpos = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
            self.schmellmagnitude_shape = np.zeros((self.schmell_N, self.schmell_N))
            for i in range(partN):
                if self.schmell_boolean[i]:
                    pos = self.positions[0, i, :].magnitude
                    self.schmell_magnitude, _ = calc_chemical_potential(
                        pos, self.testpos
                    )
                    self.schmellmagnitude_shape += self.schmell_magnitude.reshape(
                        (self.schmell_N, self.schmell_N)
                    )
                    self.schmell_maximum, _ = calc_chemical_potential(
                        np.array([500, 500]),
                        np.array([500, 500 + self.radius_col[i] / 5]),
                    )  # the 5 is for aesthetics
            self.schmell[0] = self.ax.pcolormesh(
                self.X,
                self.Y,
                self.schmellmagnitude_shape,
                vmin=np.min(self.schmellmagnitude_shape),
                vmax=self.schmell_maximum,
                cmap=self.schmellcolor,
                shading="nearest",
                zorder=0,
            )
        else:
            (self.schmell[0],) = self.ax.plot([], [], zorder=0, alpha=0)
        t = round(self.times[1], 0)
        self.time_annotate[0] = self.ax.annotate(
            f"time in ${t:g~L}$",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            zorder=partN * 3 + partN * self.n_cones,
        )
        self.written_info[0] = self.ax.annotate(
            "", xy=(0.70, 0.60), xycoords="axes fraction"
        )

    def animation_maze_setup(self, folder, filename):
        maze_file = open(folder + filename, "rb")
        self.maze_points = pickle.load(maze_file)
        self.wall_thickness = pickle.load(maze_file)
        self.maze_walls = pickle.load(maze_file)
        self.maze = [0] * len(self.maze_walls)

        # update the limits of the plot

        xmin = 1000
        xmax = 0
        ymin = 1000
        ymax = 0
        for wall in self.maze_walls:
            if wall[0] < xmin:
                xmin = wall[0]
            if wall[0] > xmax:
                xmax = wall[0]
            if wall[1] < ymin:
                ymin = wall[1]
            if wall[1] > ymax:
                ymax = wall[1]
            if wall[2] < xmin:
                xmin = wall[2]
            if wall[2] > xmax:
                xmax = wall[2]
            if wall[3] < ymin:
                ymin = wall[3]
            if wall[3] > ymax:
                ymax = wall[3]
        Delta_max_x = max(np.max(self.positions[:, :, 0].magnitude), xmax) - min(
            np.min(self.positions[:, :, 0].magnitude), xmin
        )
        mean_x = (
            min(np.min(self.positions[:, :, 0].magnitude), xmin)
            + max(np.max(self.positions[:, :, 0].magnitude), xmax)
        ) / 2
        Delta_max_y = max(np.max(self.positions[:, :, 1].magnitude), ymax) - min(
            np.min(self.positions[:, :, 1].magnitude), ymin
        )
        mean_y = (
            min(np.min(self.positions[:, :, 1].magnitude), ymin)
            + max(np.max(self.positions[:, :, 1].magnitude), ymax)
        ) / 2
        max_region = max(Delta_max_x, Delta_max_y) * 1.2
        self.x_0 = mean_x - max_region / 2
        self.x_1 = mean_x + max_region / 2
        self.y_0 = mean_y - max_region / 2
        self.y_1 = mean_y + max_region / 2
        # self.ax.set_xlim(self.x_0, self.x_1)
        # self.ax.set_ylim(self.y_0, self.y_1)
        self.ax.set_xlim(495, 515)
        self.ax.set_ylim(570, 595)

        for i, wall in enumerate(self.maze_walls):
            vec_along_wall = [wall[2] - wall[0], wall[3] - wall[1]]
            wall_length = np.linalg.norm(vec_along_wall)
            shift = [
                vec_along_wall[1] * self.wall_thickness / (2 * wall_length),
                -vec_along_wall[0] * self.wall_thickness / (2 * wall_length),
            ]  # turning around the corner yields misplaced walls,
            # -> small cosmetic shift walls, cosmetic if wall is thin
            self.maze[i] = self.ax.add_patch(
                patches.Rectangle(
                    (wall[0] + shift[0], wall[1] + shift[1]),
                    wall_length,
                    self.wall_thickness,
                    angle=180 / np.pi * angle_from_vector(vec_along_wall),
                    color="black",
                    alpha=0.5,
                )
            )

    def animation_plt_update(self, frame):
        t = round(self.times[frame], 0)
        self.time_annotate[0].set(text=f"time in ${t:g~L}$")

        if self.written_info_data is not None:
            self.written_info[0].set(text=self.written_info_data[frame])

        if self.schmell_boolean != [False] * len(self.ids):
            self.schmellmagnitude_shape = np.zeros((self.schmell_N, self.schmell_N))

        for i in range(len(self.positions[0, :, 0])):
            directors_angle = np.arctan2(
                self.directors[frame, i, 1], self.directors[frame, i, 0]
            )
            xdata = self.positions[: frame + 1, i, 0].magnitude
            ydata = self.positions[: frame + 1, i, 1].magnitude
            self.part_body[i].set(center=(xdata[frame], ydata[frame]))
            if self.trace_boolean[i] and not self.trace_fade_boolean[i]:
                self.trace[i].set_data(xdata, ydata)
            elif self.trace_boolean[i] and self.trace_fade_boolean[i]:
                points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                self.trace[i].set_segments(segments)
                x = np.linspace(0, frame, frame + 1)  # switch to self.time
                alphas = 1 / 5 + 4 / 5 * np.exp((-x[-1] + x) / 100)
                self.trace[i].set_array(alphas)
            else:
                pass
            if self.vision_cone_boolean[i]:
                for j in range(self.n_cones):
                    self.cone[i * self.n_cones + j].set(
                        center=(xdata[frame], ydata[frame])
                    )
                    anglerange = 2 * self.cone_half_angle[i]
                    theta1 = -self.cone_half_angle[i] + anglerange / self.n_cones * j
                    theta2 = -self.cone_half_angle[i] + anglerange / self.n_cones * (
                        j + 1
                    )
                    self.cone[i * self.n_cones + j].set(
                        theta1=(directors_angle + theta1) * 180 / np.pi,
                        theta2=(directors_angle + theta2) * 180 / np.pi,
                    )
                    if self.vision_cone_data is not None:
                        self.cone[i * self.n_cones + j].set(
                            alpha=self.vision_cone_data_frame[frame][i][j]
                        )
            if self.eyes_boolean[i]:
                lefteye_x = (
                    self.radius_col[i] * 0.9 * np.cos(directors_angle + np.pi / 4)
                )
                lefteye_y = (
                    self.radius_col[i] * 0.9 * np.sin(directors_angle + np.pi / 4)
                )
                righteye_x = (
                    self.radius_col[i] * 0.9 * np.cos(directors_angle - np.pi / 4)
                )
                righteye_y = (
                    self.radius_col[i] * 0.9 * np.sin(directors_angle - np.pi / 4)
                )
                self.part_lefteye[i].set(
                    center=(xdata[frame] + lefteye_x, ydata[frame] + lefteye_y)
                )
                self.part_righteye[i].set(
                    center=(xdata[frame] + righteye_x, ydata[frame] + righteye_y)
                )
            if self.schmell_boolean[i]:
                pos = self.positions[frame, i, :].magnitude
                self.schmell_magnitude, _ = calc_chemical_potential(pos, self.testpos)
                self.schmellmagnitude_shape += self.schmell_magnitude.reshape(
                    (self.schmell_N, self.schmell_N)
                )
        if self.schmell_boolean != [False] * len(self.ids):
            self.schmell[0].set_array(self.schmellmagnitude_shape)
        if self.maze_boolean:
            return tuple(
                self.trace
                + self.cone
                + self.part_body
                + self.part_lefteye
                + self.part_righteye
                + self.time_annotate
                + self.schmell
                + self.written_info
                + self.maze
            )
        else:
            return tuple(
                self.trace
                + self.cone
                + self.part_body
                + self.part_lefteye
                + self.part_righteye
                + self.time_annotate
                + self.schmell
                + self.written_info
            )


def load_traj(foldername, ureg):
    positions = None
    directors = None
    times = None

    with h5py.File(foldername + "/trajectory.h5py") as traj_file:
        positions = np.array(traj_file["colloids/Unwrapped_Positions"][:, :, :2])
        directors = np.array(traj_file["colloids/Directors"][:, :, :2])
        times = np.array(traj_file["colloids/Times"][:, 0, 0])
        ids = np.array(traj_file["colloids/Ids"])[0, :, 0]
        types = np.array(traj_file["colloids/Types"])[0, :, 0]
    positions = positions * ureg.micrometer
    times = times * ureg.second

    if positions is None or directors is None or times is None:
        raise Exception(
            "Reading the positions, directors or times of the simulation failed"
        )

    return positions, directors, times, ids, types


def load_extra_data_to_gif(ani_instance, parameters):
    if "title_data" in parameters.keys() and parameters["title_data"] is not None:
        ani_instance.written_info_data = parameters["title_data"]
    else:
        ani_instance.written_info_data = None
    if (
        "vision_cone_data" in parameters.keys()
        and parameters["vision_cone_data"] is not None
    ):
        ani_instance.vision_cone_data = parameters["vision_cone_data"]
    else:
        ani_instance.vision_cone_data = None


def gifvisualization(
    foldername, positions, directors, times, ids, types, parameters, gif_file_names
):
    ureg = parameters["ureg"]
    files = os.listdir(foldername)

    if "trajectory.h5py" in files:
        positions, directors, times, ids, types = load_traj(foldername, ureg)

    fig, ax = plt.subplots(figsize=(7, 7))
    # setup the units for automatic ax_labeling

    positions.ito(ureg.micrometer)
    times.ito(ureg.second)
    ani_instance = Animations(
        fig,
        ax,
        positions,
        directors,
        times,
        ids,
        types,
        vision_cone_boolean=[False, False, False],
        cone_radius=parameters["detectionRadiusPosition"].to(ureg.micrometer).magnitude,
        n_cones=parameters["nCones"],
        cone_half_angle=parameters["visionHalfAngle"].magnitude,
        trace_boolean=[True, True, True],
        trace_fade_boolean=[True, True, True],
        eyes_boolean=[True, False, False],
        radius_col=[
            parameters["radiusColloid"].to(ureg.micrometer).magnitude,
            parameters["rodThickness"].to(ureg.micrometer).magnitude / 2,
            0,
        ],
        schmell_boolean=False,
        schmell_ids=parameters["rodBorderpartsId"],
        maze_boolean=False,
    )

    load_extra_data_to_gif(ani_instance, parameters)

    ani_instance.animation_plt_init()

    ani_instance.animation_maze_setup(
        parameters["mazeFolder"], parameters["mazeFileName"]
    )
    # set start and end of visualization and set the interval of between frames
    ani = FuncAnimation(
        fig,
        ani_instance.animation_plt_update,
        frames=range(2350, len(times[:])),
        blit=True,
        interval=100,
    )

    plt.show()
    # outcomment this to build a gif file (might take a while)

    ani.save(gif_file_names, fps=60)


def slidervisualization(
    foldername, positions, directors, times, ids, types, parameters, gif_file_names
):
    ureg = parameters["ureg"]
    files = os.listdir(foldername)

    if "trajectory.h5py" in files:
        positions, directors, times, ids, types = load_traj(foldername, ureg)

    fig, ax = plt.subplots(figsize=(7, 7.5))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # setup the units for automatic ax_labeling
    ureg = parameters["ureg"]
    positions.ito(ureg.micrometer)
    times.ito(ureg.second)

    ani_instance = Animations(
        fig,
        ax,
        positions,
        directors,
        times,
        ids,
        types,
        vision_cone_boolean=[False, False, False],
        cone_radius=parameters["detectionRadiusPosition"].to(ureg.micrometer).magnitude,
        n_cones=parameters["nCones"],
        cone_half_angle=parameters["visionHalfAngle"].magnitude,
        trace_boolean=[True, True, True],
        trace_fade_boolean=[True, True, True],
        eyes_boolean=[True, False, False],
        radius_col=[
            parameters["radiusColloid"].to(ureg.micrometer).magnitude,
            parameters["rodThickness"].to(ureg.micrometer).magnitude / 2,
            0,
        ],
        schmell_boolean=False,
        schmell_ids=parameters["rodBorderpartsId"],
        maze_boolean=False,
    )

    load_extra_data_to_gif(ani_instance, parameters)

    ani_instance.animation_plt_init()

    ani_instance.animation_maze_setup(
        parameters["mazeFolder"], parameters["mazeFileName"]
    )

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="gray")
    time_interval = parameters["writeInterval"].to(ureg.second).magnitude

    t = times.units
    slider_frame = Slider(
        ax_slider,
        f"time in ${t:~L}$",
        0,
        (len(times[:]) - 1) * time_interval,
        valinit=time_interval,
        valstep=time_interval,
    )

    def plt_slider_update(frame):
        frame = int(slider_frame.val / slider_frame.valstep)
        ani_instance.animation_plt_update(frame)
        fig.canvas.draw_idle()
        return

    slider_frame.on_changed(plt_slider_update)
    plt.show()
