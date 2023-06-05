import inspect
import os
import pickle
import random

import bottleneck as bn
import h5py
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from scipy import optimize

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


def schmell_grad_from_schmell_mag(schmell_mag):
    #  parameters with arbitrary units currently set to aim amplitude \approx 1
    diffusion_coefficient = 670  # glucose in water in mum^2/s
    density_decay_per_second = 0.4
    amount_per_second_from_source = 1800  # chemics
    rescale_fac = np.sqrt(density_decay_per_second / diffusion_coefficient)
    const = 4182.925639571625
    amplitude = amount_per_second_from_source / const

    if schmell_mag != 0:
        distance = optimize.newton(A_kv_zero, 0.005, args=(schmell_mag,), tol=0.001)
    else:
        distance = 300  # if no schmell source is in reach then newton doesn't converge

    norm_schmell_gradient = amplitude * rescale_fac * kvp(0, distance)

    return norm_schmell_gradient


def A_kv_zero(x, schmell_mag):
    return 0.5 * kv(0, x) - schmell_mag


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
        ax_set_xlim,
        ax_set_ylim,
        vision_cone_boolean,
        cone_radius,
        cone_vision_of_types,
        n_cones,
        cone_half_angle,
        trace_boolean,
        trace_fade_boolean,
        eyes_boolean,
        arrow_boolean,
        body_color,
        body_color_based_on_action_for_type,
        body_color_alpha,
        radius_col,
        background_boolean,
        background_sites,
        background_colors,
        background_data_generator,
        rod_rotation_chess_board_boolean,
        rod_length,
        rod_center_points,
        maze_boolean,
    ):
        self.color_index = None
        self.x_1 = None
        self.x_0 = None
        self.y_1 = None
        self.y_0 = None
        self.ax_set_xlim = ax_set_xlim
        self.ax_set_ylim = ax_set_ylim
        self.fig = fig
        self.ax = ax

        self.positions = positions
        self.directors = directors
        self.times = times
        self.ids = ids
        self.types = types
        _, count_types = np.unique(self.types, return_counts=True)
        self.n_types = len(count_types)

        # extra data
        self.written_info_data = None
        self.vision_cone_data = None
        self.vision_cone_data_frame = None
        self.action_data = None

        self.vision_cone_boolean = [False] * len(self.ids)
        self.cone_radius = [0] * len(self.ids)
        self.cone_vision_of_types = cone_vision_of_types
        self.n_cones = 1
        self.cone_half_angle = [0] * len(self.ids)
        self.trace_boolean = [False] * len(self.ids)
        self.trace_fade_boolean = [False] * len(self.ids)
        self.eyes_boolean = [False] * len(self.ids)
        self.arrow_boolean = [False] * len(self.ids)
        self.radius_col = [0] * len(self.ids)

        self.background_boolean = background_boolean
        self.background_sites = background_sites
        self.background_colors = background_colors
        self.background_data_generator = background_data_generator
        self.background_N = 200

        """
                for i in range(len(self.ids)):
            if int(self.ids[i]) in schmell_ids:
                # those ids correspond to chemical emitting colloids
                self.schmell_boolean[i] = schmell_boolean
        self.schmell_boolean[len(self.ids)] = schmell_boolean

        """

        self.rod_rotation_chess_board_boolean = [False] * len(self.ids)
        self.rod_length = rod_length
        self.rod_center_points = rod_center_points

        self.maze_boolean = maze_boolean  # type=2 corresponds to wall particles
        self.maze_points = {}
        self.wall_thickness = 42
        self.maze_walls = []

        # all the visual objects
        self.rod_rotation_chess_board_field = [0] * len(self.ids)
        self.trace = [0] * len(self.ids)
        self.part_body = [0] * len(self.ids)
        self.body_color = body_color
        self.body_color_alpha = body_color_alpha
        self.body_color_based_on_action_for_type = body_color_based_on_action_for_type
        self.part_lefteye = [0] * len(self.ids)
        self.part_righteye = [0] * len(self.ids)
        self.part_arrow = [0] * len(self.ids)
        self.time_annotate = [0]
        self.written_info = [0]
        self.maze = []

        self.set_gif_options_for_different_ids(
            vision_cone_boolean,
            cone_radius,
            n_cones,
            cone_half_angle,
            trace_boolean,
            trace_fade_boolean,
            eyes_boolean,
            arrow_boolean,
            radius_col,
            rod_rotation_chess_board_boolean,
            rod_center_points,
        )

    def set_gif_options_for_different_ids(
        self,
        vision_cone_boolean,
        cone_radius,
        n_cones,
        cone_half_angle,
        trace_boolean,
        trace_fade_boolean,
        eyes_boolean,
        arrow_boolean,
        radius_col,
        rod_rotation_chess_board_boolean,
        rod_center_points,
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
                self.arrow_boolean[i] = arrow_boolean[self.types[i]]
                self.radius_col[i] = radius_col[self.types[i]]
            elif self.types[i] == possible_types[1]:
                # type=1 correspond to rod colloids
                # print(possible_types[1])
                self.vision_cone_boolean[i] = vision_cone_boolean[self.types[i]]
                self.trace_boolean[i] = trace_boolean[self.types[i]]
                self.trace_fade_boolean[i] = trace_fade_boolean[self.types[i]]
                self.eyes_boolean[i] = eyes_boolean[self.types[i]]
                self.arrow_boolean[i] = arrow_boolean[self.types[i]]
                self.radius_col[i] = radius_col[self.types[i]]
            elif self.types[i] == possible_types[2]:
                # type=3 correspond to whatever colloids
                # print(possible_types[2])
                self.vision_cone_boolean[i] = vision_cone_boolean[self.types[i]]
                self.trace_boolean[i] = trace_boolean[self.types[i]]
                self.trace_fade_boolean[i] = trace_fade_boolean[self.types[i]]
                self.eyes_boolean[i] = eyes_boolean[self.types[i]]
                self.arrow_boolean[i] = arrow_boolean[self.types[i]]
            else:
                raise Exception("unknown colloid type in visualization")

        for i in range(len(self.ids)):
            if int(self.ids[i]) in rod_center_points:
                # those ids correspond to base particle of the chess board background

                self.rod_rotation_chess_board_boolean[i] = (
                    rod_rotation_chess_board_boolean
                )

    # General initialization runs once and calls extra preparation functions to help
    def animation_plt_init(self):
        # calc figure limits
        delta_max_x = np.max(self.positions[:, :, 0].magnitude) - np.min(
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
        max_region = max(delta_max_x, delta_max_y) * 1.2
        self.x_0 = mean_x - max_region / 2
        self.x_1 = mean_x + max_region / 2
        self.y_0 = mean_y - max_region / 2
        self.y_1 = mean_y + max_region / 2
        if self.ax_set_xlim is not None:
            self.x_0 = self.ax_set_xlim[0]
            self.x_1 = self.ax_set_xlim[1]
        if self.ax_set_ylim is not None:
            self.y_0 = self.ax_set_ylim[0]
            self.y_1 = self.ax_set_ylim[1]
        self.ax.set_xlim(self.x_0, self.x_1)
        self.ax.set_ylim(self.y_0, self.y_1)
        l_units = self.positions.units
        self.ax.set_xlabel(f"x-position in ${l_units:~L}$")
        self.ax.set_ylabel(f"y-position in ${l_units:~L}$")

        # some general tests
        if len(self.positions[0, :, 0]) != len(self.ids):
            raise Exception(
                "position[0,:,0] is "
                + str(len(self.written_info_data))
                + " long and self.ids is "
                + str(len(self.ids))
                + "they must be equal."
                " maybe check your load_traj function call,"
                " if you use swarmrl and it's trajectory.hdf5 file call"
            )

        if self.written_info_data is not None:
            if len(self.written_info_data) != len(self.times):
                raise Exception(
                    "In Visualization written_info_data is "
                    + str(len(self.written_info_data))
                    + " long and self.times is "
                    + str(len(self.times))
                )

        # prepare trace fade (colors)
        self.color_names = []
        for color_name, color in colors.TABLEAU_COLORS.items():
            self.color_names.append(color_name)
        norm = plt.Normalize(0, 1)
        # different particles should have different colors
        # so make a small random list here
        self.color_index = [random.randint(1, 9) for _ in range(len(self.ids))]
        self.mycolor = [0] * len(self.ids)
        for i in range(len(self.ids)):
            cfade = colors.to_rgb(self.color_names[self.color_index[i]]) + (0.0,)
            self.mycolor[i] = colors.LinearSegmentedColormap.from_list(
                "my", [cfade, self.color_names[self.color_index[i]]]
            )

        try:
            if self.action_data is None and self.body_color_information != []:
                print(
                    "there is no action data to color the particle bodies with this"
                    " input. Falling back to type"
                )
                self.body_color_based_on_action_for_type = []
        except ValueError:
            pass

        # prepare rod rotation chess board
        if self.rod_rotation_chess_board_boolean:
            self.prepare_rod_rotation_chess_board_data()

        # prepare vision cones
        if self.n_cones < 1:  # default =1
            print(
                "You have chosen zero vision cones (self.n_cones<1), I made"
                " self.vision_cone_boolean[i]=False because no vision is what you"
                " want.  "
            )
            self.vision_cone_boolean = [False] * len(self.ids)
            self.n_cones = 1
        self.cone = [0] * (self.n_cones * len(self.ids) * self.n_types)

        vision_cone_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]

        if self.vision_cone_boolean != [False] * len(self.ids):
            self.prepare_vision_cone_data()

        # Init visual objects / patches

        # z_order:
        # schmell & rod rotation chess board    =0
        # trace                                 <n_parts
        # vision cone                           <n_parts + n_parts*n_cones*
        # part_body                             <2*n_parts + n_parts*n_cones
        # eyes or  arrow                        <4*n_parts + n_parts*n_cones
        # writtenInfo and time_info             =4*n_parts + n_parts*n_cones +1

        self.init_rod_rotation_chess_board()

        # init background coloring
        if self.background_boolean:
            self.init_background()
        else:
            (self.background[0],) = self.ax.plot([], [], zorder=0, alpha=0)

        n_parts = len(self.ids)

        self.time_annotate[0] = self.ax.annotate(
            "time:",
            xy=(0.02, 0.93),
            xycoords="axes fraction",
            zorder=n_parts * 4 + n_parts * self.n_cones + 1,
        )
        self.written_info[0] = self.ax.annotate(
            "",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            zorder=n_parts * 4 + n_parts * self.n_cones + 1,
        )

        # loop over the particles and init its individual visual effects

        for i in range(n_parts):
            self.part_body[i] = self.ax.add_patch(
                patches.Circle(
                    xy=(-27000, -27000),
                    radius=self.radius_col[i],
                    alpha=self.body_color_alpha,
                    color=self.body_color["type"][
                        self.types[i]
                    ],  # init the colors with types
                    zorder=n_parts + n_parts * self.n_cones + i,
                )
            )
            if self.eyes_boolean[i]:
                self.part_lefteye[i] = self.ax.add_patch(
                    patches.Circle(
                        xy=(-27000, -27000),
                        radius=self.radius_col[i] / 5,
                        alpha=0.7,
                        color="k",
                        zorder=n_parts * 2 + n_parts * self.n_cones + 2 * i,
                    )
                )
                self.part_righteye[i] = self.ax.add_patch(
                    patches.Circle(
                        xy=(-27000, -27000),
                        radius=self.radius_col[i] / 5,
                        alpha=0.7,
                        color="k",
                        zorder=n_parts * 2 + n_parts * self.n_cones + 1 + 2 * i,
                    )
                )
            else:
                self.part_lefteye[i] = self.ax.add_patch(
                    patches.Circle(xy=(0, 0), radius=42, visible=False)
                )
                self.part_righteye[i] = self.ax.add_patch(
                    patches.Circle(xy=(0, 0), radius=42, visible=False)
                )

            if self.arrow_boolean[i]:
                self.part_arrow[i] = self.ax.add_patch(
                    patches.FancyArrow(
                        x=0,
                        y=0,
                        dx=0,
                        dy=0,
                        overhang=0.1,
                        head_width=self.radius_col[i] * 0.95,
                        head_length=2 * self.radius_col[i] * 0.95,
                        length_includes_head=True,
                        color="k",
                        zorder=n_parts * 2 + n_parts * self.n_cones + 2 * i,
                    )
                )
            else:
                self.part_arrow[i] = self.ax.add_patch(
                    patches.FancyArrow(x=0, y=0, dx=0, dy=0, visible=False)
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

            if self.n_types > len(vision_cone_colors):
                raise Exception(
                    "A foul was not creative enough to set sufficient colors"
                    + "You need "
                    + str(self.n_types)
                    + " colors, because you have that much different types to detect"
                    + " independently."
                )

            if self.vision_cone_boolean[i]:
                for j in range(self.n_cones):
                    for k in range(self.n_types):
                        self.cone[
                            i * self.n_cones * self.n_types + j * self.n_types + k
                        ] = self.ax.add_patch(
                            patches.Wedge(
                                center=(0, 0),
                                r=self.cone_radius[i],
                                theta1=0,
                                theta2=0,
                                alpha=0.2,
                                color=vision_cone_colors[k],
                                zorder=n_parts
                                + i * self.n_cones * self.n_types
                                + j * self.n_types
                                + k,
                            )
                        )
            else:
                for j in range(self.n_cones):
                    for k in range(self.n_types):
                        self.cone[
                            i * self.n_cones * self.n_types + j * self.n_types + k
                        ] = self.ax.add_patch(
                            patches.Wedge(
                                center=(0, 0), r=42, theta1=0, theta2=0, visible=False
                            )
                        )

    # Extra preparation that runs once  __ Extra preparation that runs once

    def prepare_rod_rotation_chess_board_data(self):
        angles = np.ones(
            (len(self.directors[:, 0, 0]) + 1, len(self.directors[0, :, 0]))
        )
        # one step more than the self.time is long because
        # after np.diff it will get one shorter
        # there is some error then in the beginning

        for step in range(len(angles[:, 0]) - 1):
            for par in range(len(angles[0, :])):
                angles[step + 1, par] = np.arctan2(
                    self.directors[step, par, 1], self.directors[step, par, 0]
                )

        smooth_rotation_noise = 50
        if smooth_rotation_noise >= len(self.directors[:, 0, 0]):
            smooth_rotation_noise = len(self.directors[:, 0, 0] - 1)

        # get rid of overflows in the angle Unit rad
        diff_angles = np.diff(angles, axis=0)

        diff_angles = np.where(diff_angles > 1, diff_angles - 2 * np.pi, diff_angles)
        diff_angles = np.where(diff_angles < -1, diff_angles + 2 * np.pi, diff_angles)

        # the edge case handling in the bn.move_mean function
        # is better and it is faster than np.convolve

        diff_angles_smooth = bn.move_mean(
            diff_angles, window=smooth_rotation_noise, min_count=1, axis=0
        )

        self.diff_angle_signs = np.where(diff_angles_smooth > 0, 0.1, 0.4)

    def init_rod_rotation_chess_board(self):
        n_parts = len(self.ids)
        if self.rod_rotation_chess_board_boolean != [False] * n_parts:
            for i in range(n_parts):
                if self.rod_rotation_chess_board_boolean[i]:
                    self.rod_rotation_chess_board_field[i] = self.ax.add_patch(
                        patches.Circle(
                            xy=(-27000, -27000),
                            radius=self.rod_length / 2,
                            alpha=0.7,
                            color="k",
                            zorder=0,
                        )
                    )
                else:
                    self.rod_rotation_chess_board_field[i] = self.ax.add_patch(
                        patches.Circle(xy=(0, 0), radius=42, visible=False)
                    )

    def prepare_vision_cone_data(self):
        if self.vision_cone_data is None:
            print(
                "You haven't set self.vision_cone_boolean[i] !=0 for some i, i.e. you"
                " want vision cones but no self.vision_cone_data in the options are"
                " provided. Then default is created"
            )
            each_type = [0.05 for _ in range(self.n_types)]
            each_cone = np.array(
                [each_type] * self.n_cones
            )  # len(type) is probably 6 instead of supposed 2
            self.vision_cone_data = [
                [[idid, each_cone] for idid in range(len(self.ids))]
                for _ in range(len(self.times))
            ]

        if self.vision_cone_data is not None:
            if len(self.vision_cone_data) != len(self.times):
                raise Exception(
                    "vision_cone_data is "
                    + str(len(self.vision_cone_data))
                    + " long and self.times is "
                    + str(len(self.times))
                    + "they should be equal."
                )
            if len(self.vision_cone_data[0]) > len(self.ids):
                raise Exception(
                    "vision_cone_data[0] is "
                    + str(len(self.vision_cone_data[0]))
                    + " long and the number of colloids is "
                    + str(len(self.ids))
                    + "the first should be smaller or equal,"
                    "because maybe there are rodparticles "
                    "that do not have a vision cone."
                    "it should be checked for every  possible i "
                    "as a replacment for 0"
                )
            if len(self.vision_cone_data[0][0]) != 2:
                raise Exception(
                    "vision_cone_data[0][0] is "
                    + str(len(self.vision_cone_data[0][0]))
                    + " long and  it should be 2 containing the  observer colloid.id"
                    " and the vision data per frame, per colloid in a np.array of "
                    " all types visible."
                )
            if len(self.vision_cone_data[0][0][1][:, 0]) != self.n_cones:
                raise Exception(
                    "vision_cone_data[0][0][1][:, 0] is "
                    + str(len(self.vision_cone_data[0][0][1][:, 0]))
                    + " long and the number of cones is "
                    + str(self.n_cones)
                    + " they should be equal"
                )

        # expand vision cone data
        if self.vision_cone_data is not None:
            self.vision_cone_data_frame = np.zeros(
                (len(self.times), len(self.ids), self.n_cones, self.n_types)
            )
            for frame in range(
                len(self.times)
            ):  # not correct for self.ids with holes,...
                for c_id in range(len(self.ids)):
                    for given_c_id in range(len(self.vision_cone_data[frame])):
                        if c_id == self.vision_cone_data[frame][given_c_id][0]:
                            self.vision_cone_data_frame[frame, c_id] = (
                                self.vision_cone_data[frame][given_c_id][1]
                            )

            # color adjustment for each color separately
            for detected_type in range(self.n_types):
                if (
                    np.max(self.vision_cone_data_frame[:, :, :, detected_type]) != 0
                    and detected_type in self.cone_vision_of_types
                ):
                    norm_vals = self.vision_cone_data_frame[
                        :, :, :, detected_type
                    ] / np.mean(self.vision_cone_data_frame[:, :, :, detected_type])
                    self.vision_cone_data_frame[:, :, :, detected_type] = (
                        np.arctan(norm_vals / 1) * 1 / np.pi
                    )  # divide norm_vals by ca. 100 to get shaded colors
                    self.vision_cone_data_frame[:, :, :, detected_type] += 0.05

    def init_background(self):
        self.background = [0] * len(self.background_sites)
        self.X, self.Y = np.mgrid[
            self.x_0 : self.x_1 : complex(0, self.background_N),
            self.y_0 : self.y_1 : complex(0, self.background_N),
        ]
        self.testpos = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
        self.background_data_shape = np.zeros(
            (len(self.background_sites), self.background_N, self.background_N)
        )

        # copy the dictionary
        self.background_color_map = self.background_colors
        self.max_val_pcolormesh = {}

        # this holds a ditionary with the functions in the class
        self.background_data_sources = dict(
            inspect.getmembers(
                self.background_data_generator, predicate=inspect.ismethod
            )
        )

        for key_idx, key in enumerate(self.background_sites):
            cfade = colors.to_rgb(self.background_colors[key]) + (0.0,)
            self.background_color_map[key] = colors.LinearSegmentedColormap.from_list(
                "my", [cfade, self.background_colors[key]]
            )
            if "fixed_pos" in key:
                source_pos = self.background_sites[key]
                self.background_data_magnitude, self.max_val_pcolormesh[key] = (
                    self.background_data_sources["get_" + key + "_data"](
                        source_pos, self.testpos
                    )
                )
                self.background_data_shape[
                    key_idx, :, :
                ] += self.background_data_magnitude.reshape(
                    self.background_N, self.background_N
                )
                self.background_data_shape[key_idx, :, :] /= np.amax(
                    self.background_data_shape[key_idx, :, :]
                )
            elif "id_pos" in key:
                source_ids = self.background_sites[key]
                pos = self.positions[0, source_ids, :2].magnitude
                self.background_data_magnitude, self.max_val_pcolormesh[key] = (
                    self.background_data_sources["get_" + key + "_data"](
                        pos, self.testpos
                    )
                )
                self.background_data_shape[
                    key_idx, :, :
                ] += self.background_data_magnitude.reshape(
                    self.background_N, self.background_N
                )
                self.background_data_shape[key_idx, :, :] /= np.amax(
                    self.background_data_shape[key_idx, :, :]
                )
            else:
                print(
                    "Unknown key in Visualization/Animation dictionary background_sites"
                    " and probably then also in background_colors."
                )

        for key_idx, key in enumerate(self.background_sites):
            self.background[key_idx] = self.ax.pcolormesh(
                self.X,
                self.Y,
                np.arctan(self.background_data_shape[key_idx, :, :] * 2.5),
                vmin=np.arctan(np.min(self.background_data_shape[key_idx, :, :] * 2.5)),
                vmax=self.max_val_pcolormesh[key],
                cmap=self.background_color_map[key],
                shading="nearest",
                zorder=0,
            )

    def animation_maze_setup(self, folder, filename):
        maze_file = open(folder + filename, "rb")
        self.maze_dic = pickle.load(maze_file)
        self.wall_thickness = self.maze_dic["wall_thickness"]
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
        delta_max_x = max(np.max(self.positions[:, :, 0].magnitude), xmax) - min(
            np.min(self.positions[:, :, 0].magnitude), xmin
        )
        mean_x = (
            min(np.min(self.positions[:, :, 0].magnitude), xmin)
            + max(np.max(self.positions[:, :, 0].magnitude), xmax)
        ) / 2
        delta_max_y = max(np.max(self.positions[:, :, 1].magnitude), ymax) - min(
            np.min(self.positions[:, :, 1].magnitude), ymin
        )
        mean_y = (
            min(np.min(self.positions[:, :, 1].magnitude), ymin)
            + max(np.max(self.positions[:, :, 1].magnitude), ymax)
        ) / 2
        max_region = max(delta_max_x, delta_max_y) * 1.2
        self.x_0 = mean_x - max_region / 2
        self.x_1 = mean_x + max_region / 2
        self.y_0 = mean_y - max_region / 2
        self.y_1 = mean_y + max_region / 2

        if self.ax_set_xlim is not None:
            self.x_0 = self.ax_set_xlim[0]
            self.x_1 = self.ax_set_xlim[1]
        if self.ax_set_ylim is not None:
            self.y_0 = self.ax_set_ylim[0]
            self.y_1 = self.ax_set_ylim[1]

        self.ax.set_xlim(self.x_0, self.x_1)
        self.ax.set_ylim(self.y_0, self.y_1)

        # the ax limits have changed now we need to init it again
        self.background[0].set_alpha(0)
        if self.background_boolean:
            self.init_background()
        else:
            (self.background[0],) = self.ax.plot([], [], zorder=0, alpha=0)

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
                    color="k",
                    alpha=0.5,
                )
            )

    # Updates all the elements in the visualization each frame
    def animation_plt_update(self, frame):
        if len(self.times) <= frame:
            raise Exception(
                " There are "
                + len(self.times[frame])
                + "frame available.You try to access the frame: "
                + str(frame)
                + " which is to high"
            )

        # Update the time counter
        t = round(self.times[frame], 0)
        self.time_annotate[0].set(text=f"time: ${t:g~L}$")

        # Updating the written Info
        if self.written_info_data is not None:
            if "|" in self.written_info_data[frame]:
                first_text, sep, second_text = self.written_info_data[frame].partition(
                    "|"
                )
                self.written_info[0].set(text=first_text + "\n" + second_text)
            else:
                self.written_info[0].set(text=self.written_info_data[frame])

        # Updating the background field
        if self.background_boolean:
            for key_idx, key in enumerate(self.background_sites):
                if "id_pos" in key:
                    self.background_data_shape[key_idx, :, :] *= 0
                    source_ids = self.background_sites[key]
                    pos = self.positions[frame, source_ids, :2].magnitude
                    self.background_data_magnitude, _ = self.background_data_sources[
                        "get_" + key + "_data"
                    ](pos, self.testpos)
                    self.background_data_shape[
                        key_idx, :, :
                    ] += self.background_data_magnitude.reshape(
                        self.background_N, self.background_N
                    )
                    self.background_data_shape[key_idx, :, :] /= np.amax(
                        self.background_data_shape[key_idx, :, :]
                    )
                else:
                    print(
                        "Unknown key in Visualization/Animation dictionary"
                        " background_sites and probably then also in background_colors."
                    )

            for key_idx, key in enumerate(self.background_sites):
                self.background[key_idx].set_array(
                    np.arctan(self.background_data_shape[key_idx, :, :] * 2.5)
                )

        # Update objects for different ids
        for i in range(len(self.ids)):
            directors_angle = np.arctan2(
                self.directors[frame, i, 1], self.directors[frame, i, 0]
            )
            xdata = self.positions[: frame + 1, i, 0].magnitude
            ydata = self.positions[: frame + 1, i, 1].magnitude

            # Update position
            self.part_body[i].set(center=(xdata[frame], ydata[frame]))

            # Update color
            if self.body_color_based_on_action_for_type != []:
                if self.types[i] not in self.body_color_based_on_action_for_type:
                    self.part_body[i].set(color=self.body_color["type"][self.types[i]])
                    # print(self.body_color[0][self.types[i]])
                elif self.types[i] in self.body_color_based_on_action_for_type:
                    self.part_body[i].set(
                        color=self.body_color["action"][int(self.action_data[frame, i])]
                    )
                    # print(self.body_color[1][int(self.action_data[frame,i])])

            # Update chess board pattern of rods
            if self.rod_rotation_chess_board_boolean != [False] * len(self.ids):
                if self.rod_rotation_chess_board_boolean[i]:
                    self.rod_rotation_chess_board_field[i].set(
                        center=(xdata[frame], ydata[frame]),
                        alpha=self.diff_angle_signs[frame, i],
                    )

            # Update traces behind ids
            if self.trace_boolean[i] and not self.trace_fade_boolean[i]:
                self.trace[i].set_data(xdata, ydata)
            elif self.trace_boolean[i] and self.trace_fade_boolean[i]:
                points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                self.trace[i].set_segments(segments)
                x = np.linspace(0, frame, frame + 1)  # might switch to self.time
                alphas = 1 / 5 + 4 / 5 * np.exp((-x[-1] + x) / 100)
                self.trace[i].set_array(alphas)
            else:
                pass

            # Update vision cones
            if self.vision_cone_boolean[i]:
                for j in range(self.n_cones):
                    for k in range(self.n_types):
                        self.cone[
                            i * self.n_cones * self.n_types + j * self.n_types + k
                        ].set(center=(xdata[frame], ydata[frame]))

                        angle_range = 2 * self.cone_half_angle[i]
                        theta1 = (
                            -self.cone_half_angle[i] + angle_range / self.n_cones * j
                        )
                        theta2 = -self.cone_half_angle[
                            i
                        ] + angle_range / self.n_cones * (j + 1)
                        self.cone[
                            i * self.n_cones * self.n_types + j * self.n_types + k
                        ].set(
                            theta1=(directors_angle + theta1) * 180 / np.pi,
                            theta2=(directors_angle + theta2) * 180 / np.pi,
                        )
                        if self.vision_cone_data is not None:
                            self.cone[
                                i * self.n_cones * self.n_types + j * self.n_types + k
                            ].set(alpha=self.vision_cone_data_frame[frame, i, j, k])

            # Update eyes
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

            # Update arrows

            if self.arrow_boolean[i]:
                end_x = self.radius_col[i] * 0.4 * np.cos(directors_angle + np.pi)
                end_y = self.radius_col[i] * 0.4 * np.sin(directors_angle + np.pi)
                start_x = self.radius_col[i] * np.cos(directors_angle)
                start_y = self.radius_col[i] * np.sin(directors_angle)
                # set_data is available here at least for matplotlib 3.6.4
                self.part_arrow[i].set_data(
                    x=xdata[frame] + end_x,
                    y=ydata[frame] + end_y,
                    dx=start_x - end_x,
                    dy=start_y - end_y,
                )

        if self.maze_boolean:
            return tuple(
                self.trace
                + self.cone
                + self.part_body
                + self.part_lefteye
                + self.part_righteye
                + self.part_arrow
                + self.time_annotate
                + self.background
                + self.written_info
                + self.rod_rotation_chess_board_field
                + self.maze
            )
        else:
            return tuple(
                self.trace
                + self.cone
                + self.part_body
                + self.part_lefteye
                + self.part_righteye
                + self.part_arrow
                + self.time_annotate
                + self.background
                + self.written_info
                + self.rod_rotation_chess_board_field
            )


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


def load_traj_vis(folder_name, ureg):
    positions = None
    directors = None
    times = None

    with h5py.File(folder_name + "/trajectory.hdf5") as traj_file:
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

    return positions, directors, times, ids, types, ureg


def visualization(
    vis_mode="",
    folder_name=None,
    positions=[],
    directors=[],
    times=[],
    ids=[],
    types=[],
    parameters={},
    gif_file_path=None,
):
    ureg = parameters["ureg"]
    if folder_name is not None:
        files = os.listdir(folder_name)
        if "trajectory.hdf5" in files:
            positions, directors, times, ids, types, ureg = load_traj_vis(
                folder_name, ureg
            )

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
        vision_cone_boolean=[True, False, False],
        cone_radius=parameters["detection_radius_position"]
        .to(ureg.micrometer)
        .magnitude,
        n_cones=parameters["n_cones"],
        cone_half_angle=parameters["vision_half_angle"].magnitude,
        trace_boolean=[True, True, True],
        trace_fade_boolean=[True, True, True],
        eyes_boolean=[False, False, False],
        arrow_boolean=[True, False, False],
        radius_col=[
            parameters["radius_colloid"].to(ureg.micrometer).magnitude,
            parameters["rod_thickness"].to(ureg.micrometer).magnitude / 2,
            0,
        ],
        schmell_boolean=False,
        schmell_ids=parameters["rod_border_parts_id"],
        maze_boolean=False,
    )

    load_extra_data_to_gif(ani_instance, parameters)

    ani_instance.animation_plt_init()

    if parameters["maze_file_name"] != "None":
        ani_instance.animation_maze_setup(
            parameters["maze_folder"], parameters["maze_file_name"]
        )
    ani_instance.ax.grid(True)

    if vis_mode == "GIF":
        # set start and end of visualization and set the interval of between frames
        begin_frame = 1
        end_frame = len(times[:])
        ani = FuncAnimation(
            fig,
            ani_instance.animation_plt_update,
            frames=range(begin_frame, end_frame),
            blit=True,
            interval=100,
        )
    elif vis_mode == "SLIDER":
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="gray")
        time_interval = parameters["write_interval"].to(ureg.second).magnitude
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
    else:
        raise Exception('you need to choose vis_mode = "GIF" or "SLIDER".')

    plt.show()

    # to build a gif file (might take a while)
    if gif_file_path is not None and vis_mode == "GIF":
        ani.save(gif_file_path, fps=60)
