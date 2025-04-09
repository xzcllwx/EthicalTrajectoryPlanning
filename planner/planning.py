from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction, Occupancy

import numpy as np
import os
import sys

import math

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(module_path)

from commonroad_helper_functions.exceptions import (
    GoalReachedNotification,
    NoGlobalPathFoundError,
    ScenarioCompatibilityError,
)
from planner.utils.goalcheck import GoalReachedChecker
# from planner.GlobalPath.lanelet_based_planner import LaneletPathPlanner
from planner.utils.timers import ExecTimer
from planner.Frenet.utils.helper_functions import get_max_curvature
from planner.Frenet.utils.calc_trajectory_cost import distance
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D

class Planner(object):
    """Main Planner Class"""

    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        ego_id: int,
        vehicle_params,
        exec_timer=None,
    ):

        # commonroad scenario
        self.__scenario = scenario

        # current lanelet the planning vehicle is moving on
        self.__current_lanelet_id = None

        # planning problem for the planner
        self.__planning_problem = planning_problem
        self.__goal_checker = GoalReachedChecker(planning_problem)

        # initial time step
        self.__time_step = 0

        # ID of the planning vehicle
        self.__ego_id = ego_id

        # initial state of the planning vehicle
        self.__ego_state = planning_problem.initial_state

        # minimum trajectory length
        self.__min_trajectory_length = 50

        # prediction
        self.__prediction = None

        # Timer for timing execution times.
        self.__exec_timer = (
            ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
        )

        # Variables that contain the global path
        # TODO Remove everything except the reference spline.....
        # NOTE They are all referenced within the Frenet Planner -> tricky to remove....
        self.global_path_length = None
        self.global_path = None
        self.global_path_to_goal = None
        self.global_path_after_goal = None
        self.__reference_spline = None
        # self.plan_global_path(scenario, planning_problem, vehicle_params)

        # trajectory
        dt = 0.1
        self._trajectory = {
            "s_loc_m": np.zeros(self.min_trajectory_length),
            "d_loc_m": np.zeros(self.min_trajectory_length),
            "d_d_loc_mps": np.zeros(self.min_trajectory_length),
            "d_dd_loc_mps2": np.zeros(self.min_trajectory_length),
            "x_m": np.zeros(self.min_trajectory_length),
            "y_m": np.zeros(self.min_trajectory_length),
            "psi_rad": np.zeros(self.min_trajectory_length),
            "kappa_radpm": np.zeros(self.min_trajectory_length),
            "v_mps": np.zeros(self.min_trajectory_length),
            "ax_mps2": np.zeros(self.min_trajectory_length),
            "time_s": np.arange(0, dt * self.min_trajectory_length, dt),
        }
        
        # belief
        self.__belief = None
        
    def init_global_path(self, ref_path):
        if self.__reference_spline is None: # 只初始化一次
            # transform the local path to the global coordinate system
            # ref_path = np.matmul(ref_path, rot_mat.T)
            # ref_path = ref_path + translation
            self.global_path_to_goal = ref_path
            self.__plan_global_path(ref_path)

    def step(
        self,
        scenario: Scenario,
        current_lanelet_id: int,
        time_step: int,
        ego_state: State,
        predictions=None,
        cov = None,
        belief=None,
        v_max=50,
    ):
        """Main Step Function of the Planner

        This method generates a new trajectory for the current scenario und prediction.
        It is a wrapper for the planner-type depending actual step method "_step_planner()" that updates the trajectory.
        "_step_planner()" must be overloaded by inheriting classes that implement planning methods.

        :param scenario: commonroad scenario
        :param current_lanelet_id: current lanelet id of the planning vehicle
        :param time_step: current time step
        :param ego_state: current state of the planning vehicle
        :param prediction: prediction
        :param v_max: maximum allowed velocity on the current lanelet in m/s
        """

        self.__current_lanelet_id = current_lanelet_id
        self.__time_step = time_step
        self.__ego_state = ego_state
        self.__belief = belief
           
        self._update_scenario(ego_state, predictions[:, 0, :, :])
        self.__prediction = {}
        dt = 0.1
        if predictions is not None:
            for i in range(predictions.shape[0]):
                object = self.__scenario.obstacle_by_id(i+1)
                self.__prediction[i] = {}
                self.__prediction[i]['shape'] = {}
                self.__prediction[i]['shape']['length'] = object.obstacle_shape.length
                self.__prediction[i]['shape']['width'] = object.obstacle_shape.width
                for j in range(predictions.shape[1]):
                    prediction_obj = predictions[i][j]
                    cov_obj = cov[i][j]
                    
                    self.__prediction[i][j] = {}
                    # self.__prediction[i][0]['pos_list'] = np.array([[p[0], p[1]] for p in prediction_obj]).reshape(-1, 2)
                    self.__prediction[i][j]['pos_list'] = np.transpose(prediction_obj[:2, :], (1, 0)) 
                    self.__prediction[i][j]['orientation_list'] = prediction_obj[2, :].flatten()
                    self.__prediction[i][j]['v_list'] = (np.sqrt(
                        np.diff(prediction_obj[0, :])**2 + np.diff(prediction_obj[1, :])**2
                    ) / dt).tolist()
                    self.__prediction[i][j]['v_list'].append(self.__prediction[i][j]['v_list'][-1])
                    self.__prediction[i][j]['cov_list'] = cov_obj



        # TODO: Include maximum allowed speed
        self.__v_max = v_max

        # Check if the goal is alreay reached
        # self.__check_goal_reached()

        # call the planner-type depending step function to generate a new trajectory
        return self._step_planner()

    def _update_scenario(self, ego_state, prediction):
    
        state_args = dict()
        state_args['position'] = ego_state.position
        state_args['orientation'] = ego_state.orientation
        state_args['time_step'] = self.__time_step
        current_state = State(**state_args)
        ego = self.__scenario.obstacle_by_id(self.__ego_id)
        if ego is not None:
            if ego.prediction is None:
                traj = Trajectory(self.__time_step, [current_state])
                ego.prediction = TrajectoryPrediction(traj, ego.obstacle_shape)
                ego.initial_state = current_state
            else:
                ego.prediction.trajectory.state_list.append(current_state)
                length = ego.obstacle_shape.length
                width = ego.obstacle_shape.width
                occupied_region = Rectangle(length=length, width=width, center=current_state.position, orientation=current_state.orientation)
                ego.prediction.occupancy_set.append(Occupancy(current_state.time_step, occupied_region))
            
        N_agent = prediction.shape[0]
        # transform the prediction to the global coordinate system
        for i in range(N_agent):
            prediction_obj = prediction[i]
            # prediction_obj[:, :2] = np.matmul(prediction_obj[:, :2], rot_mat.T)
            # prediction_obj[:, :2] = prediction_obj[:, :2] + translation

            state_args = dict()
            state_args['position'] = np.array([prediction_obj[0, 0], prediction_obj[1, 0]])
            state_args['orientation'] = prediction_obj[2, 0]
            state_args['time_step'] = self.__time_step
            current_state = State(**state_args)
            object = self.__scenario.obstacle_by_id(i+1)
            if object.prediction is None:
                traj = Trajectory(self.__time_step, [current_state])
                object.prediction = TrajectoryPrediction(traj, object.obstacle_shape)
                object.initial_state = current_state
            else:
                object.prediction.trajectory.state_list.append(current_state)
                length = object.obstacle_shape.length
                width = object.obstacle_shape.width
                occupied_region = Rectangle(length=length, width=width, center=current_state.position, orientation=current_state.orientation)
                # occupied_region = object.obstacle_shape.rotate_translate_local(
                #     current_state.position, current_state.orientation)
                object.prediction.occupancy_set.append(Occupancy(current_state.time_step, occupied_region))
                        
    def _step_planner(self):
        """Planner step function

        This method directyl changes the planne trajectory. It must be overloaded by an inheriting planner class.
        There is no basic trajectory planning implemented.
        """
        raise NotImplementedError(
            "No basic trajectory planning implemented. "
            "Overload the method _step_planner() to generate a trajectory."
        )

    def __check_goal_reached(self):
        # Get the ego vehicle
        self.goal_checker.register_current_state(self.ego_state)
        if self.goal_checker.goal_reached_status():
            raise GoalReachedNotification("Goal reached in time!")
        elif self.goal_checker.goal_reached_status(ignore_exceeded_time=True):
            raise GoalReachedNotification("Goal reached but time exceeded!")

    def __plan_global_path(self, ref_path):
        """Plan a global path to the planning's problem target area.

        Args:
            scenario (_type_): _description_
            planning_problem (_type_): _description_
            vehicle_params (_type_): _description_
            initial_state (_type_, optional): _description_. Defaults to None.

        Raises:
            NoGlobalPathFoundError: _description_
        """
        with self.exec_timer.time_with_cm("initialization/plan global path"):
            # calculate the global path for the planning problem
            try:
                # if the velocity is pretty high, increase the max length for a lane change
                self.__reference_spline = CubicSpline2D(
                    x=ref_path[:, 0], y=ref_path[:, 1]
                )
            # raise error if global path planner fails
            # belif planning 这里有问题
            except TypeError:
                raise NoGlobalPathFoundError(
                    "Failed. Could not find a global path for the planning problem."
                ) from None


    @property
    def planning_problem(self):
        """Planning problem to be solved"""
        return self.__planning_problem

    @property
    def goal_checker(self):
        """Return the goal checker."""
        return self.__goal_checker

    @property
    def exec_timer(self):
        """Return the exec_timer object."""
        return self.__exec_timer

    @property
    def reference_spline(self):
        """Return the reference spline object."""
        return self.__reference_spline

    @property
    def scenario(self):
        """Commonroad scenario"""
        return self.__scenario

    @property
    def time_step(self):
        """Current time step"""
        return self.__time_step

    @property
    def ego_id(self):
        """ID of the planning vehicle"""
        return self.__ego_id

    @property
    def ego_state(self):
        """Current state of the planning vehicle"""
        return self.__ego_state

    @property
    def min_trajectory_length(self):
        """Minimum length of the planned trajectory"""
        return self.__min_trajectory_length

    @property
    def trajectory(self):
        """Planned trajectory"""
        return self._trajectory

    @property
    def prediction(self):
        """Prediction"""
        return self.__prediction

    @property
    def v_max(self):
        """maximum velocity"""
        return self.__v_max

    @property
    def current_lanelet_id(self):
        """Current lanelet"""
        return self.__current_lanelet_id
    
    @property
    def belief(self):
        """Belief"""
        return self.__belief


# TODO move to separate file
def check_curvature_of_global_path(
    global_path: np.ndarray, planning_problem, vehicle_params, ego_state
):
    """
    Check the curvature of the global path.

    If the curvature is to high, points of the global path are removed to smooth the global path. In addition, a new point is added which ensures the initial orientation.

    Args:
        global_path (np.ndarray): Coordinates of the global path.

    Returns:
        np.ndarray: Coordinates of the new, smooth global path.

    """
    global_path_curvature_ok = False

    # get start velocity of the planning problem
    start_velocity = planning_problem.initial_state.velocity

    # calc max curvature for the initial velocity
    max_initial_curvature, _ = get_max_curvature(
        vehicle_params=vehicle_params, v=start_velocity
    )

    # get x and y from the global path
    global_path_x = global_path[:, 0].tolist()
    global_path_y = global_path[:, 1].tolist()

    # add a point to the global path to ensure the initial orientation of the planning problem
    # never delete this point or the initial point
    new_x = ego_state.position[0] + np.cos(ego_state.orientation) * 0.1
    new_y = ego_state.position[1] + np.sin(ego_state.orientation) * 0.1
    global_path_x.insert(1, new_x)
    global_path_y.insert(1, new_y)

    # check if the curvature of the global path is ok
    while global_path_curvature_ok is False:
        # calc the already covered arc length for the points of global path
        global_path_s = [0.0]

        for i in range(len(global_path_x) - 1):
            p_start = np.array([global_path_x[i], global_path_y[i]])
            p_end = np.array([global_path_x[i + 1], global_path_y[i + 1]])
            global_path_s.append(distance(p_start, p_end) + global_path_s[-1])

        # calculate the curvature of the global path
        dx = np.gradient(global_path_x, global_path_s)
        dy = np.gradient(global_path_y, global_path_s)

        ddx = np.gradient(dx, global_path_s)
        ddy = np.gradient(dy, global_path_s)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # loop through every curvature of the global path
        global_path_curvature_ok = True
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > max_initial_curvature:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                if global_path_s[index_closest_path_point] <= 10.0:
                    global_path_x.pop(index_closest_path_point)
                    global_path_y.pop(index_closest_path_point)
                    global_path_curvature_ok = False
                    break

        # also check if the curvature is smaller than the turning radius anywhere
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > get_max_curvature(
                vehicle_params=vehicle_params, v=0.0
            )[0]:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                global_path_x.pop(index_closest_path_point)
                global_path_y.pop(index_closest_path_point)
                global_path_curvature_ok = False
                break

    # create the new global path
    new_global_path = np.array([np.array([global_path_x[0], global_path_y[0]])])
    for i in range(1, len(global_path_y)):
        new_global_path = np.concatenate(
            (
                new_global_path,
                np.array([np.array([global_path_x[i], global_path_y[i]])]),
            )
        )

    return new_global_path


# EOF
