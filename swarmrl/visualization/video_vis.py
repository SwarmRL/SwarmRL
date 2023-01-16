import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

from scipy.special import kv # modivied bessel function of second kind of real order v
from scipy.special import kvp # derivative of modified bessel Function of second kind of real order



import random
import h5py
import numpy as np
import pint


def calc_schmell(schmellpos,colpos,Dcoltrans = 670): #glucose in water in mum^2/s
    Deltadistance = np.linalg.norm(np.array(schmellpos) - np.array(colpos),axis=-1)
    Deltadistance = np.where(Deltadistance == 0, 5, Deltadistance)
    # prevent zero division
    direction = (schmellpos - colpos) / np.stack([Deltadistance, Deltadistance], axis=-1)
    O=0.4 #  in per second
    J=1800 # in chemics per second whatever units the chemics are in. These values are chosen acording to the Amplitude \approx 1
    #const = -2 * np.pi * Dcoltrans * rodthickness / 2 * np.sqrt(O / Dcoltrans) * kvp(0, np.sqrt(
    #   O / Dcoltrans) * rodthickness / 2) #reuse already calculated value
    const=4182.925639571625
    # J=const*A the const sums over the chemics that flow throw an imaginary boundary at radius rodthickness/2
    # A is the Amplitude of the potential i.e. the chemical density (google "first ficks law" for theoretical info)
    A = J / const

    l=np.sqrt(O/Dcoltrans)*Deltadistance
    schmellmagnitude = A*kv(0,l)
    schmellgradient = - np.stack([A* np.sqrt(O/Dcoltrans)*kvp(0,l),A*np.sqrt(O/Dcoltrans)*kvp(0,l)],axis=-1)*direction
    return schmellmagnitude, schmellgradient

class Animations:
    def __init__(self, fig, ax, positions, directors, times, ids, types):
        self.color_index = None
        self.x_1 = None
        self.x_0 = None
        self.fig = fig
        self.ax = ax
        self.written_info_data = None
        self.positions = positions
        self.directors = directors
        self.times = times
        self.ids = ids
        self.types = types
        self.vision_cone_data = None

        self.vision_cone_boolean = [False] * len(self.ids)
        self.cone_radius = [0] * len(self.ids)
        self.nCones = 1
        self.cone_angle = [0] * len(self.ids)
        self.trace_boolean = [False] * len(self.ids)
        self.trace_fade_boolean = [False] * len(self.ids)
        self.eyes_boolean = [False] * len(self.ids)
        self.radius_col = [0] * len(self.ids)
        self.schmell_boolean = [False] * len(self.ids)

        self.trace = [0] * len(self.ids)

        self.part_body = [0] * len(self.ids)
        self.part_lefteye = [0] * len(self.ids)
        self.part_righteye = [0] * len(self.ids)
        self.time_annotate = [0]
        self.schmell = [0]
        self.written_info= [0]

    def animation_plt_init(self):

        # calc figure limits
        Delta_max_x = np.max(self.positions[:, :, 0]) - np.min(self.positions[:, :, 0])
        mean_x = (np.min(self.positions[:, :, 0] + np.max(self.positions[:, :, 0]))) / 2
        delta_max_y = np.max(self.positions[:, :, 1]) - np.min(self.positions[:, :, 1])
        mean_y = (np.min(self.positions[:, :, 1] + np.max(self.positions[:, :, 1]))) / 2
        max_region = max(Delta_max_x, delta_max_y) * 1.2
        self.x_0 = mean_x.magnitude - max_region.magnitude / 2
        self.x_1 = mean_x.magnitude + max_region.magnitude / 2
        self.y_0 = mean_y.magnitude - max_region.magnitude / 2
        self.y_1 = mean_y.magnitude + max_region.magnitude / 2
        self.ax.set_xlim(self.x_0, self.x_1)
        self.ax.set_ylim(self.y_0, self.y_1)
        l = self.positions.units
        self.ax.set_xlabel(f'x-position in ${l:~L}$')
        self.ax.set_ylabel(f'y-position in ${l:~L}$')
        self.ax.grid(True)

        if self.written_info_data != None:
            if len(self.written_info_data) != len(self.times):
                raise Exception("In Visualization written_info_data is " + str(len(self.written_info_data)) + " long and self.times is " + str(
                        len(self.times)))

        # prepare colors for trace fade and schmell
        self.color_names = []
        for color_name, color in colors.TABLEAU_COLORS.items():
            self.color_names.append(color_name)
        norm = plt.Normalize(0, 1)
        self.color_index = [random.randint(1, 9) for _ in range(len(self.ids))]
        self.mycolor = [0] * len(self.ids)

        for i in range(len(self.ids)):
            cfade = colors.to_rgb(self.color_names[self.color_index[i]]) + (0.0,)
            self.mycolor[i] = colors.LinearSegmentedColormap.from_list('my', [cfade, self.color_names[self.color_index[i]]])

        # prepare  schmell
        self.schmell_N = 50
        cfade = colors.to_rgb(self.color_names[4]) + (0.0,)
        self.schmellcolor = colors.LinearSegmentedColormap.from_list('my', [cfade, self.color_names[4]])

        #prepare vision cones
        if self.nCones < 1: #default =1
            print(
                "You have choosen zero vision cones (self.nCones<1), I made self.vision_cone_boolean[i]=False because no vision is what you want.  ")
            self.vision_cone_boolean = [False] * len(self.ids)
            self.nCones = 1
        self.cone = [0] * (self.nCones * len(self.ids))

        if self.vision_cone_boolean!=[False] * len(self.ids) and self.vision_cone_data == None:
            print("You set haven't self.vision_cone_boolean[i] !=0 for some i, i.e. you want vision cones but no self.vision_cone_data in the options are provided. Then default is created")
            self.vision_cone_data = [[[ 0.2  for _ in range(self.nCones)] for _ in range(len(self.ids))] for _ in range(len(self.times)) ]

        if self.vision_cone_boolean!=[False] * len(self.ids) and self.vision_cone_data != None :
            if len(self.vision_cone_data) != len(self.times):
                raise Exception("vision_cone_data is " + str(len(self.vision_cone_data)) + " long and self.times is " + str(
                    len(self.times)))

        partN = len(self.ids)
        for i in range(partN):
            self.part_body[i] = self.ax.add_patch(
                patches.Circle(xy=(0, 0),
                               radius=self.radius_col[i],
                               alpha=0.7,
                               color='g',
                               zorder=partN + partN * self.nCones + i))
            if self.trace_boolean[i] and not self.trace_fade_boolean[i]:
                self.trace[i], = self.ax.plot([], [], zorder=i + 1)
            elif self.trace_boolean[i] and self.trace_fade_boolean[i]:
                lc = LineCollection([[[0, 0], [0, 0]]], norm=norm, cmap=self.mycolor[i], zorder=i + 1)
                self.trace[i] = self.ax.add_collection(lc)
            elif not self.trace_boolean[i]:
                self.trace[i], = self.ax.plot([], [], zorder=i + 1, alpha=0)
            else:
                raise Exception("self.trace_boolean is neither True nor False")

            if self.vision_cone_boolean[i]:
                for j in range(self.nCones):
                    self.cone[i*self.nCones+j] = self.ax.add_patch(patches.Wedge(center=(0, 0),
                                                                r=self.cone_radius[i],
                                                                theta1=0,
                                                                theta2=0,
                                                                alpha=0.2,
                                                                color='r',
                                                                zorder=partN + i*self.nCones+j))
            else:
                for j in range(self.nCones):
                    self.cone[i*self.nCones+j] = self.ax.add_patch(patches.Wedge(center=(0, 0), r=42, theta1=0, theta2=0, alpha=0))
            if self.eyes_boolean[i]:
                self.part_lefteye[i] = self.ax.add_patch(patches.Circle(xy=(0, 0),
                                                                        radius=self.radius_col[i] / 5,
                                                                        alpha=0.7,
                                                                        color='k', zorder=partN*2 + partN * self.nCones + 2 * i))
                self.part_righteye[i] = self.ax.add_patch(patches.Circle(xy=(0, 0),
                                                                         radius=self.radius_col[i] / 5,
                                                                         alpha=0.7,
                                                                         color='k', zorder=partN*2 + partN * self.nCones + 1 + 2 * i))
            else:
                self.part_lefteye[i] = self.ax.add_patch(patches.Circle(xy=(0, 0), radius=42, alpha=0))
                self.part_righteye[i] = self.ax.add_patch(patches.Circle(xy=(0, 0), radius=42, alpha=0))

        if self.schmell_boolean != [False] * len(self.ids):
            self.X, self.Y = np.mgrid[self.x_0:self.x_1:complex(0, self.schmell_N),
                             self.y_0:self.y_1:complex(0, self.schmell_N)]
            self.testpos = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
            self.schmellmagnitude_shape = np.zeros((self.schmell_N, self.schmell_N))
            for i in range(partN):
                if self.schmell_boolean[i]:
                    pos = self.positions[0, i, :].magnitude
                    self.schmell_magnitude, _ = calc_schmell(pos, self.testpos)
                    self.schmellmagnitude_shape += self.schmell_magnitude.reshape((self.schmell_N, self.schmell_N))
                    self.schmell_maximum, _ = calc_schmell(np.array([500, 500]), np.array(
                        [500, 500 + self.radius_col[i] / 5]))  # the 5 is for aesthetics
            self.schmell[0] = self.ax.pcolormesh(self.X, self.Y, self.schmellmagnitude_shape,
                                                 vmin=np.min(self.schmellmagnitude_shape), vmax=self.schmell_maximum,
                                                 cmap=self.schmellcolor,
                                                 shading='nearest',
                                                 zorder=0)
        else:
            self.schmell[0], = self.ax.plot([], [], zorder=0, alpha=0)
        t = round(self.times[1], 0)
        self.time_annotate[0] = self.ax.annotate(f"time in ${t:g~L}$",
                                                 xy=(.02, .95),
                                                 xycoords='axes fraction',
                                                 zorder=partN * 3 + partN * self.nCones)
        self.written_info[0] = self.ax.annotate(f"",xy=(.70, .60),xycoords='axes fraction')

    def animation_plt_update(self, frame):
        t = round(self.times[frame], 0)
        self.time_annotate[0].set(text=f"time in ${t:g~L}$")

        if self.written_info_data != None:
            self.written_info[0].set(text=self.written_info_data[frame])

        if self.schmell_boolean != [False] * len(self.ids):
            self.schmellmagnitude_shape = np.zeros((self.schmell_N, self.schmell_N))

        for i in range(len(self.positions[0, :, 0])):
            directors_angle = np.arctan2(self.directors[frame, i, 1], self.directors[frame, i, 0])
            xdata = self.positions[:frame + 1, i, 0].magnitude
            ydata = self.positions[:frame + 1, i, 1].magnitude
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
                for j in range(self.nCones):
                    self.cone[i*self.nCones+j].set(center=(xdata[frame], ydata[frame]))
                    anglerange = 2*self.cone_angle[i]
                    theta1 = - self.cone_angle[i]+anglerange/self.nCones*j
                    theta2 = - self.cone_angle[i]+anglerange/self.nCones*(j+1)
                    self.cone[i*self.nCones+j].set(theta1=(directors_angle + theta1) * 180 / np.pi,
                                                    theta2=(directors_angle + theta2) * 180 / np.pi)
                    if self.vision_cone_data != None:
                        self.cone[i*self.nCones+j].set(alpha=self.vision_cone_data[frame][i][j])
            if self.eyes_boolean[i]:
                lefteye_x = self.radius_col[i] * 0.9 * np.cos(directors_angle + np.pi / 4)
                lefteye_y = self.radius_col[i] * 0.9 * np.sin(directors_angle + np.pi / 4)
                righteye_x = self.radius_col[i] * 0.9 * np.cos(directors_angle - np.pi / 4)
                righteye_y = self.radius_col[i] * 0.9 * np.sin(directors_angle - np.pi / 4)
                self.part_lefteye[i].set(center=(xdata[frame] + lefteye_x, ydata[frame] + lefteye_y))
                self.part_righteye[i].set(center=(xdata[frame] + righteye_x, ydata[frame] + righteye_y))
            if self.schmell_boolean[i]:
                pos = self.positions[frame, i, :].magnitude
                self.schmell_magnitude, _ = calc_schmell(pos, self.testpos)
                self.schmellmagnitude_shape += self.schmell_magnitude.reshape((self.schmell_N, self.schmell_N))
        if self.schmell_boolean != [False] * len(self.ids):
            self.schmell[0].set_array(self.schmellmagnitude_shape)

        return tuple(
            self.trace + self.cone + self.part_body + self.part_lefteye + self.part_righteye + self.time_annotate + self.schmell + self.written_info )

def set_gif_options(ani_instance, parameters, ureg):
    # Adjust in for loop what you want (default is False) and place parameters that you have
    possible_types = list(dict.fromkeys(ani_instance.types))
    for i in range(len(ani_instance.ids)):
        if ani_instance.types[i] == possible_types[0]:  # type=0 correspond to normal colloids
            # print(possible_types[0])
            ani_instance.vision_cone_boolean[i] = True
            ani_instance.cone_radius[i] = parameters['detectionRadiusPosition'].to(ureg.micrometer).magnitude
            ani_instance.nCones = parameters['nCones'] #  different Cone numbers for different particles is not yet supported
            ani_instance.cone_angle[i] = parameters['visionHalfAngle'].magnitude
            ani_instance.trace_boolean[i] = True
            ani_instance.trace_fade_boolean[i] = True
            ani_instance.eyes_boolean[i] = True
            ani_instance.radius_col[i] = parameters["radiusColloid"].to(ureg.micrometer).magnitude
        elif ani_instance.types[i] == possible_types[1]:  # type=1 correspond to rod colloids
            # print(possible_types[1])
            ani_instance.vision_cone_boolean[i] = False
            ani_instance.trace_boolean[i] = False
            ani_instance.trace_fade_boolean[i] = False
            ani_instance.eyes_boolean[i] = False
            ani_instance.radius_col[i] = parameters["rodThickness"].to(ureg.micrometer).magnitude
            if int(ani_instance.ids[i]) in parameters[
                'rodBorderpartsId']:  # those ids correspond to chemical emitting colloids
                ani_instance.schmell_boolean[i] = False
        elif ani_instance.types[i] == possible_types[2]:  # type=2 correspond to ??? colloids
            # print(possible_types[2])
            ani_instance.vision_cone_boolean[i] = False
            ani_instance.trace_boolean[i] = False
            ani_instance.trace_fade_boolean[i] = False
            ani_instance.eyes_boolean[i] = False
            ani_instance.radius_col[i] = 0
        else:
            raise Exception("unknown colloid type in visualisation")

def load_extra_data_to_gif(ani_instance, parameters):
    if "title_data" in parameters.keys() and parameters["title_data"]!= None :
        ani_instance.written_info_data=parameters["title_data"]
    else:
        ani_instance.written_info_data=None
    if "vision_cone_data" in parameters.keys() and parameters["vision_cone_data"]!= None:
        ani_instance.vision_cone_data=parameters["vision_cone_data"]
    else:
        ani_instance.vision_cone_data=None
def gifvisualization(positions, directors, times, ids, types, parameters, gif_file_names):
    fig, ax = plt.subplots(figsize=(7, 7))
    # setup the units for automatic ax_labeling
    ureg = parameters['ureg']
    positions.ito(ureg.micrometer)
    times.ito(ureg.second)
    ani_instance = Animations(fig, ax, positions, directors, times, ids, types)

    load_extra_data_to_gif(ani_instance, parameters)

    set_gif_options(ani_instance, parameters, ureg)

    ani_instance.animation_plt_init()
    # set start and end of visualization
    ani = FuncAnimation(fig, ani_instance.animation_plt_update, frames=len(times[:]), blit=True, interval=1)

    plt.show()
    # outcomment this to build a gif file (might take a while )
    # ani.save(f'{fold_struct_names["analysis_outfolder"]}/animation.gif', fps=60)



def slidervisualization(positions, directors, times, ids, types, parameters):
    # make a slider controlled plot, to pause at specific positions in the simulation and have a closer look at it
    fig, ax = plt.subplots(figsize=(7, 7.5))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # setup the units for automatic ax_labeling
    ureg = parameters['ureg']
    positions.ito(ureg.micrometer)
    times.ito(ureg.second)

    ani_instance = Animations(fig, ax, positions, directors, times, ids, types)

    load_extra_data_to_gif(ani_instance, parameters)

    set_gif_options(ani_instance, parameters, ureg)

    ani_instance.animation_plt_init()
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='gray')
    time_interval = parameters['writeInterval'].to(ureg.second).magnitude

    t = times.units
    slider_frame = Slider(ax_slider,
                          f'time in ${t:~L}$',
                          0,
                          (len(times[:]) - 1) * time_interval,
                          valinit=time_interval,
                          valstep=time_interval
                          )

    def plt_slider_update(frame):
        frame = int(slider_frame.val / slider_frame.valstep)
        ani_instance.animation_plt_update(frame)
        fig.canvas.draw_idle()
        return

    slider_frame.on_changed(plt_slider_update)
    plt.show()
