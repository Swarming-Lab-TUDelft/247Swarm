#!/usr/bin/env python3

"""
Node for controlling a single Crazyflie using a state machine.
"""

import time
import sys
import numpy as np
import math
import traceback

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from std_msgs.msg import String
from topic_interface.msg import StringList, Location, ControllerCommand, PosVelList

from .helper_classes import RollingList
from .logger import Logger

# bash command: ros2 run cf_swarm CrazyflieNodeV2 --ros-args -p uri:="radio://0/90/2M/247E000003" -p radio_id:=0

# read settings from configuration file
from .config import *


pos_PID_scaling = 0.67
YAW_PID_scaling = 1.0
POS_PID = [2 * pos_PID_scaling, 0.25* pos_PID_scaling, -1*pos_PID_scaling] # P x,y,z; I z; D x,y,z;
YAW_PID = [0.5*YAW_PID_scaling, 1.0, math.pi * YAW_PID_scaling] # P,I,D, I-limit for yaw, in Crazyflie firmware settings are [6, 1.0, 0.35] PID, seems to be degree controller

NO_YAW = 400

# drone states
INITIALISING = "initialising"
CHECKING_PAD = "checking pad"
STARTING = "starting"
CHECK_CHARGING = "check charging"
CHARGING = "charging"
WAITING = "waiting"
PRE_TAKE_OFF = "pre take off"
TAKING_OFF = "taking off"
SWARMING = "swarming"
RETURNING = "returning"
LANDING = "landing"
LANDING_IN_PLACE = "land in place"
SHUTDOWN = "shutdown"
ERROR_HANDLING = "error handling"
ERROR = "error"
DISCONNECTED = "disconnected"


#supervisor bit allocations
CAN_BE_ARMED = 7
IS_ARMED = 6
AUTO_ARM = 5
CAN_FLY = 4
IS_FLYING = 3
TUMBLED = 2



class Drone(Node):
    def __init__(self):
        super().__init__('CrazyflieNode', automatically_declare_parameters_from_overrides=True)

        # get parameters
        self.uri = self.get_parameter('uri').get_parameter_value().string_value
        self.radio_id = self.get_parameter('radio_id').get_parameter_value().integer_value
        self.uri_idx = None

        # variables to store RadioHandler data
        self.radio_uris = []
        self.drone_response = ""
        self.controller_command = ""
        self.controller_announcement = ""

        # current state of parameter logs
        self.logging_state = {
            "pos vel": {int(1/COMMAND_UR*1000): False, int(1/COMMAND_UR_STANDBY*1000): False},
            "system state": {int(1/SYSTEM_PARAM_UR*1000): False},
            }

        # QoS profile for latching topics (allows for subscribers to receive the last published message when they subscribe)
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # create subscribers
        self.radio_uris_sub = self.create_subscription(StringList, f'ID{self.radio_id}/self_uris', self.update_connected_drones, qos_profile=latching_qos)
        self.drone_response_sub = self.create_subscription(StringList, f'ID{self.radio_id}/response', self.update_drone_response, qos_profile=latching_qos)
        self.drone_param_sub = self.create_subscription(StringList, f'ID{self.radio_id}/drone_parameters', self.update_drone_param, 10)
        self.receive_pad_sub = self.create_subscription(Location, 'pad_location', self.pad_location, 10)
        self.controller_command_sub = self.create_subscription(ControllerCommand, 'controller_command', self.update_controller_command, 10)
        self.CA_command_sub = self.create_subscription(PosVelList, 'CA_command', self.CA_command_callback, 10)
        self.GUI_command_sub = self.create_subscription(String, 'GUI_command', self.GUI_command_callback, 10)
        self.controller_announcement_sub = self.create_subscription(String, 'controller_announcement', self.controller_announcement_cb, 10)
        self.receive_velocity_sub = self.create_subscription(PosVelList, 'posvel_target', self.posvel_target_cb, 10)

        # create publishers
        self.command_pub = self.create_publisher(String, 'E' + self.uri.split('/')[-1] + '/command', qos_profile=latching_qos)
        self.req_charge_pub_ = self.create_publisher(String, 'cf_charge_req', 10)
        self.publish_pad_location = self.create_publisher(Location, 'init_pad_location', 10)
        self.state_pub = self.create_publisher(String, 'E' + self.uri.split('/')[-1] + '/state', 10)
        self.msgs_pub = self.create_publisher(String, 'E' + self.uri.split('/')[-1] + '/msgs', 10)
        self.CA_pub = self.create_publisher(String, 'E' + self.uri.split('/')[-1] + '/CA', 10)
        self.error_pub = self.create_publisher(String, 'error', 10)

        self.drone_response_timer = None

        # logger
        self.log = Logger(self.get_logger(), self.msgs_pub, mode="info")

        # states and their functions
        self.state_start = True  # only true the first time a state function is executed
        self.state_timer = time.time()
        self.prev_state = ""
        self.state_func = {
            INITIALISING: self.initialise,
            CHECKING_PAD: self.check_pad,
            STARTING: self.startup,
            CHECK_CHARGING: self.check_charging,
            CHARGING: self.charge,
            WAITING: self.wait,
            PRE_TAKE_OFF: self.pre_takeoff_check,
            TAKING_OFF: self.take_off,
            SWARMING: self.swarming,
            RETURNING: self.returning,
            LANDING: self.land,
            LANDING_IN_PLACE: self.land_in_place,
            SHUTDOWN: self.shutdown,
            ERROR_HANDLING: self.error_handling,
            ERROR: self.error,
            DISCONNECTED: self.disconnected,
        }

        # define led bits (for operator information purposes)
        self.led_bits = {
            "ENABLE": 0b10000000,
            "DISABLE": 0,
            "BLUE_L": 0b10000001,
            "GREEN_L": 0b10000010,
            "RED_L": 0b10000100,
            "GREEN_R": 0b10001000,
            "RED_R": 0b10010000,
            "BLUE_R": 0b10100000
        }

        # lighthouse active variable and timer
        self.time_lh_change = time.time()
        self.lh_prev = 0
        self.lh_state = 0  # 0 = stopped for at least 1.5 second, 1 = active for at least 1.5 seconds

        # land variables
        self.landing_position = [0, 0, 0]
        self.landing_cleared = False
        self.land_again = False
        self.return_after_takeoff = False
        self.land_counter = 0
        self.after_land_in_place_state = ERROR

        self.is_flying = False
        
        # sys logging variables
        self.battery_state = 0
        self.battery_voltage = 0
        self.lh_active = 0
        self.supervisor = [0]*8
        
        # yaw variables
        self.target_yaw = 0

        # target variables
        self.final_velocity = [0.0, 0.0, 0.0]
        self.final_yaw = NO_YAW
        self.desired_velocity = [0.0, 0.0, 0.0]
        self.target_mode = "position"
        self.target = [0.0, 0.0, 0.0]
        self.target_msg_idx = None
        self.target_pos = [0.0, 0.0, 0.0]
        self.CA_msg_idx = None
        self.CA_velocity = [0.0, 0.0, 0.0]
        self.time_last_vel_sent = time.time()
        self.time_last_CA_rec = time.time()
        self.time_last_vel_update = time.time()
        self.time_take_off_start = 0

        # initial position calculation
        self.samples = RollingList(50, init_values=[0.0, 0.0, 0.0])
        self.last_pos = [0, 0, 0]
        self.starting_pos = None
        
        # integral PID-controller variables
        self.desired_yaw = NO_YAW
        self.integral_z = 0
        self.integral_xy = [0, 0]

        # position and velocity variables
        self.position = [0, 0, 0]
        self.velocity = [0, 0, 0]
        self.initial_position = np.array([0,0,0])

        # error message variable
        self.error_msg = ""
        
        # Create a callback for self.loop() which belongs to a personal MutuallyExclusiveCallbackGroup such that other callbacks can be executed in parallel
        loop_group = MutuallyExclusiveCallbackGroup()
        self.state = INITIALISING

        self.area = "SAFE"  # SAFE, HIGH_RISK, OUT_OF_BOUNDS

        # error handling
        self.error_handling_config = [False, lambda: True, -1, ERROR, lambda: True, -1, ERROR, WAITING]  # (reboot, reboot condition, reboot_timeout, reboot_timeout_state, return condition, return_timeout, return_timout_state, return state)
        self.reboot_timout_timer = time.time()
        self.return_timout_timer = time.time()
        # error handling templates, WITHOUT RETURN STATE
        self.error_handling_dict = {
            "reboot": [True, lambda: True, -1, ERROR, lambda: True, -1, ERROR],
            "drone tumbled": [True, lambda: self.supervisor[TUMBLED] == 0 and self.area == "SAFE", -1, ERROR, lambda: self.lh_state == 1, 10, ERROR_HANDLING],
            "LH stopped": [False, lambda: True, -1, ERROR, lambda: self.lh_state == 1, -1, ERROR],
            "out of bounds": [False, lambda: self.area == "SAFE", -1, ERROR, lambda: True, -1, ERROR],
            "empty": [False, lambda: True, -1, ERROR, lambda: True, -1, ERROR],
        }
        self.error_handling_max_timeout = 20

        # create timers
        self.loop_call = self.create_timer(1/MAIN_LOOP_UR, self.callback_loop, callback_group=loop_group)
        self.system_state_timer = self.create_timer(1/SYSTEM_PARAM_UR, self.system_state_timer_cb)
        self.system_state_timer.cancel()
        if CA_MODE != "off":
            self.CA_send_timer = self.create_timer(1/COMMAND_UR, self.CA_send_timer_cb)
            self.CA_enabled = False
            self.disable_CA()


    ############################# Callbacks #############################
    
    def update_connected_drones(self, msg):
        """
        Every time the list of connected drones is updated, get the index of the current drone in the updated list.
        """
        self.radio_uris = msg.sl
        for i, value in enumerate(self.radio_uris):
            if value == self.uri:
                self.uri_idx = i
                break

    def update_drone_response(self, msg):
        """
        Store response from radiohandler.
        """
        if self.uri_idx is None:
            if self.drone_response_timer is None:
                self.drone_response_timer = self.create_timer(0.1, lambda msg_i=msg: self.update_drone_response(msg_i))
        else:
            if self.drone_response_timer is not None:
                self.drone_response_timer.destroy()
                self.drone_response_timer = None
            self.drone_response = str(msg.sl[self.uri_idx])

            if self.drone_response.startswith("logcb"):
                data = self.drone_response.split(':')
                self.logging_state[data[1]][int(data[2].split('.')[0])] = bool(int(data[3]))  # key: param | value: (started, period)

    def update_drone_param(self, msg):
        """
        Store drone parameters.
        """
        if self.uri_idx is not None:
            # drone parameters: x/y/z/vx/vy/vz//battery state/battery voltage/lighthouse active/supervisor
            posvel, system_state = [x.split("/") for x in msg.sl[self.uri_idx].split("//")]
            
            self.position = [float(posvel[0]), float(posvel[1]), float(posvel[2])]
            self.velocity = [float(posvel[3]), float(posvel[4]), float(posvel[5])]

            self.battery_state = int(float(system_state[0]))
            self.battery_voltage = float(system_state[1])
            self.lh_active = float(system_state[2])
            self.supervisor = [int(x) for x in format(int(system_state[3]), '08b')]


    def update_controller_command(self, msg):
        """
        Store controller commands.
        """
        if msg.uri == self.uri or msg.uri == self.uri.split('/')[-1] or msg.uri == "all":
            self.controller_command = msg.data

            if msg.data == "land in place" and self.is_flying:
                self.land_in_place_and_set_state(WAITING)
    
    def controller_announcement_cb(self, msg):
        """
        Store controller announcement.
        """
        self.controller_announcement = msg.data
        
    def posvel_target_cb(self, msg):
        """
        Save the target position or velocity, and yaw.
        """
        if self.target_msg_idx and msg.data[self.target_msg_idx].uri in (self.uri.split('/')[-1], self.uri):
            posvel = msg.data[self.target_msg_idx]
            self.target = posvel.vec
            self.target_mode = posvel.mode
            self.target_yaw = posvel.yaw
        else:
            for i, posvel in enumerate(msg.data):
                if posvel.uri in (self.uri.split('/')[-1], self.uri):
                    self.target = posvel.vec
                    self.target_mode = posvel.mode
                    self.target_yaw = posvel.yaw
                    self.target_msg_idx = i
                    break
        
        self.final_yaw = self.target_yaw if ENABLE_YAW else NO_YAW

    def pad_location(self, msg):
        """
        Called when the PadManger sends a location for a pad to land on.
        """
        if msg.uri == self.uri:
            self.log.info(f'Landing position received, cleared: {msg.clear}')
            self.landing_position = msg.location
            self.landing_cleared = msg.clear

    def CA_command_callback(self, msg):
        """
        Revieve and store collision avoidance commands.
        """
        if self.CA_msg_idx and msg.data[self.CA_msg_idx].uri in (self.uri.split('/')[-1], self.uri):
            self.CA_velocity = msg.data[self.CA_msg_idx].vec
        else:
            for i, posvel in enumerate(msg.data):
                if posvel.uri in (self.uri.split('/')[-1], self.uri):
                    self.CA_velocity = posvel.vec
                    self.CA_msg_idx = i
                    break
        
        self.CA_velocity = np.clip(self.CA_velocity, -CLIP_VEL, CLIP_VEL)

        self.time_last_CA_rec = time.time()
    
    def GUI_command_callback(self, msg):
        """
        Terminate this node when the GUI sends a terminate message.
        """
        if msg.data == "terminate/kill all":
            executor.shutdown(timeout_sec=0)
            sys.exit()
    
    
    ############################# Timers #############################

    def system_state_timer_cb(self):
        """
        Called at a fixed rate to check positioning, battery, and bounds.
        """
        # lighthouse
        if self.lh_active != self.lh_prev:
            self.time_lh_change = time.time()
            self.lh_prev = self.lh_active
        
        if self.lh_active == 0 and time.time() - self.time_lh_change > 1.5:
            self.lh_state = 0
        elif self.lh_active != 0 and time.time() - self.time_lh_change > 1.5:
            self.lh_state = 1
        
        # check if the drone has tumbled
        if self.supervisor[TUMBLED] != 0 and self.state not in (ERROR, ERROR_HANDLING):
            self.log.info("drone tumbled")
            return self.handle_error("drone tumbled", RETURNING)
        
        # check bounds
        if self.lh_state == 1:
            if (LH_HIGH_RISK_BOUNDS[0][0] < self.position[0] < LH_HIGH_RISK_BOUNDS[0][1]
                and LH_HIGH_RISK_BOUNDS[1][0] < self.position[1] < LH_HIGH_RISK_BOUNDS[1][1]
                and self.position[2] < LH_HIGH_RISK_BOUNDS[2][1]):
                self.area = "SAFE"
            elif (ABS_BOUNDS[0][0] < self.position[0] < ABS_BOUNDS[0][1]
                and ABS_BOUNDS[1][0] < self.position[1] < ABS_BOUNDS[1][1]
                and self.position[2] < ABS_BOUNDS[2][1]):
                if ENABLE_LH_HIGH_RISK:
                    self.area = "HIGH_RISK"
                else:
                    self.area = "SAFE"
            else:
                if ENABLE_BOUNDS:
                    self.area = "OUT_OF_BOUNDS"
                else:
                    self.area = "SAFE"

        # battery check
        if self.battery_state == 4:
            self.log.info("battery shutdown")
            self.state = SHUTDOWN
    
    def CA_send_timer_cb(self):
        """
        Publish desired velocity, position and velocity to the collision avoidance node.
        """
        msg = String()
        msg.data = f"{self.desired_velocity[0]}/{self.desired_velocity[1]}/{self.desired_velocity[2]}/{self.position[0]}/{self.position[1]}/{self.position[2]}/{self.velocity[0]}/{self.velocity[1]}/{self.velocity[2]}"
        self.CA_pub.publish(msg)
    
    ############################# Helper functions #############################
    
    def send_command(self, command, response, timeout=1, tries=1):
        """
        Send a command to the drone and wait for it to be processed.
        """
        for i in range(tries):
            self.command_pub.publish(String(data=command))
            command_timer = time.time()
            self.drone_response = ""
            if timeout < 0: return True
            while True:
                if self.drone_response == response:
                    return True
                elif self.drone_response == "disconnected":
                    raise Exception(f"in {self.state}: drone got disconnected")
                elif time.time() - command_timer > timeout:
                    break
        
        self.log.info(f"in {self.state}: command timed out {tries} times for {command}")
        return False

    def send_velocity(self, vel, yaw):
        """
        Send a velocity command to the drone.
        """
        if yaw != NO_YAW:
            yaw = np.clip(yaw, -360, 360)
        self.command_pub.publish(String(data=f"velocity/{vel[0]}/{vel[1]}/{vel[2]}/{yaw}"))
    
    def start_of_state(self):
        """
        Return True the first time the state function is executed.
        """
        return self.state_start
    
    def end_of_state(self):
        """
        Return True the last time the state function is executed. Only works properly if the state is changed inside the state function!
        """
        return self.state != self.prev_state

    def set_led(self, led):
        """
        Enable, disable, or turn on an led.
        """
        self.send_command(f"led/{self.led_bits[led]}", "led set")
        time.sleep(0.1)
    
    def reboot(self):
        """
        Reboot the drone.
        """
        self.logging_state = {
            "pos vel": {int(1/COMMAND_UR*1000): False, int(1/COMMAND_UR_STANDBY*1000): False},
            "system state": {int(1/SYSTEM_PARAM_UR*1000): False},
            }
        if not self.send_command("reboot", "rebooted", timeout=3, tries=3):
            return False
        time.sleep(5)
        self.lh_state = 0
        return True
    
    def publish_state(self):
        """
        Publish the current state of the drone.
        """
        self.state_pub.publish(String(data=self.state))

    def distance_to(self, target, hor=False):
        """
        Return the distance between the target and current position. If hor is True, only the horizontal distance is taken.
        """
        target = list(target).copy()
        if hor: target[2] = self.position[2]
        return np.linalg.norm([self.position[i] - target[i] for i in range(3)])

    def land_and_set_state(self, pos, state):
        """
        Perform landing procedure and set the state to the given state.
        Is meant to be called in a loop.
        """
        if self.distance_to(pos, hor=True) < 0.05 and self.position[2] < LAND_H + 0.15 + pos[2]:
            if self.distance_to(pos, hor=True) < 0.015 and self.position[2] < 0.20 + pos[2]:
                if self.position[2] < 0.07 + pos[2]:
                    if not self.send_command("stop motors", "motors stopped", tries=3):
                        return self.handle_error("reboot", self.after_land_in_place_state)
                    self.state = state
                    return
                else:
                    vel = self.run_pid_controller((pos[0], pos[1], 0.0), xy_integral=True)
                    self.send_velocity((vel[0], vel[1], -0.3), NO_YAW)
            else:
                vel = self.run_pid_controller((pos[0], pos[1], 0.15 + pos[2]), xy_integral=True)
                self.send_velocity(vel, NO_YAW)
        else:
            vel = self.run_pid_controller((pos[0], pos[1], LAND_H))
            self.send_velocity(vel, NO_YAW)
    
    def run_pid_controller(self, targ_pos, pos_pid=POS_PID, xy_integral=False):
        """
        Calculate desired velocity and yawrate.
        TODO: clean up
        """
        current_time = time.time()
        # calculate the z-axis integral term for desired velocity
        self.integral_z += (targ_pos[2] - self.position[2]) * (current_time - self.time_last_vel_update)
        self.integral_z = np.clip(self.integral_z, -0.5, 0.5)
        
        # calculate the xy-axis integral term for desired velocity
        if xy_integral:
            self.integral_xy[0] += (targ_pos[0] - self.position[0]) * (current_time - self.time_last_vel_update)
            self.integral_xy[1] += (targ_pos[1] - self.position[1]) * (current_time - self.time_last_vel_update)
            self.integral_xy = np.clip(self.integral_xy, -0.5, 0.5)
        else:
            self.integral_xy = [0, 0]
        
        self.time_last_vel_update = time.time()
        
        # calculate desired velocity for CA
        desired_vel = [
            pos_pid[0]*(targ_pos[0] - self.position[0]) + 3*pos_pid[1]*self.integral_xy[0] + pos_pid[2]*self.velocity[0],
            pos_pid[0]*(targ_pos[1] - self.position[1]) + 3*pos_pid[1]*self.integral_xy[1] + pos_pid[2]*self.velocity[1],
            pos_pid[0]*(targ_pos[2] - self.position[2]) + pos_pid[1]*self.integral_z
        ]

        # limit velocities
        desired_vel = np.clip(desired_vel, -2.0, 2.0)

        return desired_vel
    
    def fly(self):
        """
        Send velocity command to drone and perform inflight checks.
        """
        if time.time() - self.time_last_vel_sent > 1/COMMAND_UR:
            self.send_velocity(np.clip(self.final_velocity, -CLIP_VEL, CLIP_VEL), self.final_yaw)
            self.time_last_vel_sent = time.time()

        self.inflight_check()
    
    def inflight_check(self):
        """
        Perform inflight checks: positioning, bounds, and velocity limit.
        """
        if self.lh_state == 0:
            self.log.info("landing in place because LH stopped")
            self.land_in_place_and_set_state(ERROR_HANDLING, ("LH stopped", RETURNING))
        else:
            if self.area == "OUT_OF_BOUNDS":
                self.log.info("drone out of bounds")
                self.land_in_place_and_set_state(ERROR_HANDLING, ("out of bounds", RETURNING))
            elif self.area == "HIGH_RISK" and self.lh_active == 0:
                self.log.info("lost LH in high risk area")
                self.land_in_place_and_set_state(ERROR_HANDLING, ("LH stopped", RETURNING))
        
        if np.linalg.norm(self.velocity) > VELOCITY_LIMIT:
            self.log.info("max velocity exceeded")
            self.land_in_place_and_set_state(ERROR_HANDLING, ("drone tumbled", RETURNING))
        
    
    def enable_CA(self):
        """
        Send position, and velocities to collision avoidance node.
        """
        if CA_MODE == "off": return

        self.CA_send_timer.reset()
        self.CA_enabled = True
    
    def disable_CA(self):
        """
        Stop sending position, and velocities to collision avoidance node.
        """
        if CA_MODE == "off": return

        self.CA_send_timer.cancel()
        self.CA_enabled = False
        self.CA_pub.publish(String(data=f"0/0/0/100/100/{-int(self.uri.split('247E')[-1])*5}/0/0/0"))  # send a non-interfering position to CA
    
    def stop_param_log(self, param, period):
        """
        Stop parameter logging for param with the given period.
        """
        if self.logging_state[param][period]:
            i = 0
            t = time.time()
            while self.logging_state[param][period] and i < 5:
                if time.time() - t > 2.0:
                    self.command_pub.publish(String(data=f"stop param log/{param}"))
                    i += 1
                    t = time.time()
            if self.logging_state[param][period]:
                self.log.info(f"failed to stop param log:{param}:{period}")
                return False
        return True
    
    def start_param_log(self, param, period):
        """
        Start parameter logging for param with the given period.
        """
        if not self.logging_state[param][period]:
            i = 0
            t = time.time()
            while not self.logging_state[param][period] and i < 5:
                if time.time() - t > 2.0:
                    self.command_pub.publish(String(data=f"param log/{param}/{period}"))
                    i += 1
                    t = time.time()
            if not self.logging_state[param][period]:
                self.log.info(f"failed to start param log:{param}:{period}")
                return False
        return True

    def enable_param_logging_full(self):
        """
        Enable system state logging and position and velocity logging at the full rate.
        """
        if not self.stop_param_log("pos vel", int(1/COMMAND_UR_STANDBY*1000)) \
            or not self.start_param_log("pos vel", int(1/COMMAND_UR*1000)) \
            or not self.start_param_log("system state", int(1/SYSTEM_PARAM_UR*1000)):
            return False
        return True
    
    def enable_param_logging_standby(self):
        """
        Enable system state logging and position and velocity logging at the standby rate.
        """
        if not self.stop_param_log("pos vel", int(1/COMMAND_UR*1000)) \
            or not self.start_param_log("pos vel", int(1/COMMAND_UR_STANDBY*1000)) \
            or not self.start_param_log("system state", int(1/SYSTEM_PARAM_UR*1000)):
            return False
        return True

    def land_in_place_and_set_state(self, state, error_args=None):
        """
        Land in place and afterwards go to the given state. If the next state is ERROR or ERROR_HANDLING, use error_args.
        """
        self.state = LANDING_IN_PLACE
        self.after_land_in_place_state = state
        if state == ERROR:
            self.error_msg = error_args
        elif state == ERROR_HANDLING:
            self.error_handling_config = [*self.error_handling_dict[error_args[0]], error_args[1]]
    
    def handle_error(self, template, return_state=None):
        """
        Set return state to got to after ERROR_HANDLING and select the error handling template.
        """
        if not return_state:
            return_state = self.state
        
        self.error_handling_config = [*self.error_handling_dict[template], return_state]
        self.state = ERROR_HANDLING

    
    ############################# State machine #############################

    def initialise(self):
        """
        Reboot the drone after all radios are connected and enable full parameter logging.
        """
        self.publish_state()

        if self.drone_response == "connected" and self.controller_announcement == "radios ready":
            self.system_state_timer.reset()
            self.controller_announcement = ""
            if not self.reboot():
                self.handle_error("reboot", CHECKING_PAD)
            if self.enable_param_logging_full():
                self.state = CHECKING_PAD
            else:
                self.handle_error("reboot", CHECKING_PAD)
            
    

    def check_pad(self):
        """
        If the drone is charging, perform simple outlier detection on the initial position and publish it to the PadManager.
        TODO: implement better outlier detection. Ideas: Test for 1 drone. 0.1 seems a bit rough since it allows a 0.6m range for all parameters 99% interval
                                                         Doesnt this just push the location after 1 sample? Is 0, 0, 0 deleted from samples? Print samples
        """
        if np.all(self.initial_position == [0, 0, 0]) and self.position != self.last_pos and (self.battery_state == 1 or self.battery_state == 2):
            self.samples.append(self.position)
            self.last_pos = self.position

            if np.any(self.samples[-1] != [0, 0, 0]):
                mean = np.mean(self.samples.data, axis=0)
                std = np.std(self.samples.data, axis=0)

                # remove outliers that are more than 2 standard deviations away from the mean
                new_samples = np.delete(self.samples.data, np.any(np.abs(self.samples.data - mean) > 2*std, axis=1), axis=0)
                if np.all(np.std(new_samples, axis=0) < 0.005):
                    self.get_logger().debug(f"Pad STD smaller than 5mm for {len(new_samples)} samples")
                    self.initial_position = np.mean(new_samples, axis=0)
                    self.get_logger().debug(f"Concluded pad position: {[round(x, 4) for x in self.initial_position]} +- {[round(x, 4) for x in std]}")
                    msg = Location()
                    msg.uri = self.uri
                    msg.location = list(self.initial_position)
                    self.publish_pad_location.publish(msg)
                    self.state = STARTING
        elif self.battery_state == 0 and time.time() - self.state_timer > 2:
            self.log.info("drone not started on landing pad")
            if self.lh_state == 1:
                self.starting_pos = self.position
            else:
                self.starting_pos = WAIT_POS_RETURN
            self.state = STARTING
        elif time.time() - self.state_timer > 20:
            self.log.info("drone took too long to initialise pad location")
            if self.lh_state == 1:
                self.starting_pos = self.position
            else:
                self.starting_pos = WAIT_POS_RETURN
            self.state = STARTING
        elif self.battery_state == 3 or self.battery_state == 4:
            self.state = SHUTDOWN
            self.log.info("battery too low: shutting down")


    def startup(self):
        """
        Reduce the parameter logging rate.
        """
        if self.start_of_state():
            if not self.enable_param_logging_standby():
                return self.handle_error("reboot")
        if time.time() - self.state_timer > 3:
            if STARTUP_TO_WAITING:
                self.state = WAITING
            else:
                self.state = CHARGING


    def check_charging(self):
        """
        Check if the drone is charging after landing, otherwise try again.
        """
        if self.battery_state == 1 or self.battery_state == 2:
            self.state = CHARGING
            self.land_counter = 0
        elif time.time() - self.state_timer > 3:
            if self.land_counter >= LANDING_MAX_TRIES:
                if self.battery_state == 3:
                    self.state = ERROR
                    self.error_msg = "Can't seem to find landing pad and battery too low"
                else:
                    self.log.info("Can't seem to find landing pad, going to WAITING")
                    self.state = WAITING
                    self.land_counter = 0
            elif self.controller_command not in ("land in place", "emergency land"):
                self.land_again = True
                self.state = TAKING_OFF
        

    def charge(self):
        """
        Wait until the battery is charged enough to take off.
        """
        if self.start_of_state():
            if not self.enable_param_logging_standby():
                return self.handle_error("reboot")

        # go to waiting if the battery percentage is above 80%
        if  self.battery_voltage > 4.1: # self.battery_state == 2 or
            self.state = WAITING
        
        if self.battery_state not in (1, 2):
            self.log.info("removed from charging pad")
            self.state = WAITING


    def wait(self):
        """
        Drone is ready to take off and is waiting for take off command.
        """
        if self.start_of_state():
            if not self.enable_param_logging_standby():
                return self.handle_error("reboot")
            
        if self.controller_command == "take off":
            self.controller_command = ""
            self.state = PRE_TAKE_OFF
        
        if self.battery_state == 3:
            self.state = ERROR
            self.error_msg = "battery too low in waiting"


    def pre_takeoff_check(self):
        """
        Increase the parameter logging rate and go to TAKING_OFF when the drone has a stable position.
        """
        if self.start_of_state():
            if not self.enable_param_logging_full():
                return self.handle_error("reboot")

            self.state_timer = time.time()
        
        if time.time() - self.state_timer > 5:
            return self.handle_error("reboot")

        if self.lh_state == 1:
            self.state = TAKING_OFF


    def take_off(self):
        """
        Take off to 0.3m and go to SWARMING.
        """
        if self.start_of_state():
            self.initial_position = self.position
            # send a position above the drone to the collision avoidance node such that no drones will fly directly above it
            self.CA_pub.publish(String(data=f"0/0/0/{self.initial_position[0]}/{self.initial_position[1]}/0.5/0/0/0"))

            self.state_timer = time.time()
        
        # if the drone hasn't taken off after 5 seconds
        if time.time() - self.state_timer > 5 and self.velocity[2] < 0.1:
            if self.land_again:
                return self.handle_error("reboot", TAKING_OFF)
            return self.handle_error("reboot", WAITING)
        
        # take off procedure
        if time.time() - self.time_last_vel_sent > 1/COMMAND_UR:
            # if the drone is high enough to start collision avoidance
            if CA_MODE != "off" and self.position[2] > CA_RADIUS:
                if not self.CA_enabled:
                    self.enable_CA()
                self.desired_velocity = self.run_pid_controller((self.initial_position[0], self.initial_position[1], 0.5))
                self.send_velocity(self.CA_velocity, NO_YAW)
            else:
                vel = self.run_pid_controller((self.initial_position[0], self.initial_position[1], 0.5))
                self.command_pub.publish(String(data=f"hover/{vel[0]}/{vel[1]}/0/{0.5}"))
            
            self.time_last_vel_sent = time.time()
        
        # go to swarming when the drone reached 0.3m
        if self.position[2] > 0.3:
            self.send_velocity((0, 0, 0), NO_YAW)
            if self.land_again:
                self.state = LANDING
                self.land_again = False
                self.land_counter += 1
            elif self.return_after_takeoff:
                self.state = RETURNING
                self.return_after_takeoff = False
            else:
                self.target = WAIT_POS_TAKEOFF
                self.target_mode = "position"
                self.state = SWARMING
        
        self.inflight_check()
            

    def swarming(self):
        """
        Swarm and go to RETURNING when battery is empty or return command is received.
        """
        if self.start_of_state():
            self.land_counter = 0
            self.integral_z = 0
            self.time_last_CA_rec = time.time()
        
        # if collision avoidance has to be used
        if CA_MODE == "full":
            if self.target_mode == "position":
                self.desired_velocity = self.run_pid_controller(self.target)
            else:
                self.desired_velocity = self.target
            
            self.final_velocity = self.CA_velocity

            if time.time() - self.time_last_CA_rec > 2:
                self.log.info("No CA message received for 2 second")
                self.land_in_place_and_set_state(ERROR, "No CA message received for 2 second")
                return
        else:
            if self.target_mode == "position":
                self.final_velocity = self.run_pid_controller(self.target)
            else:
                self.final_velocity = self.target

        if self.battery_state == 3 or self.controller_command == "return":
            self.state = RETURNING
            self.controller_command = ""
        
        self.fly()

    
    def returning(self):
        """
        Fly to waiting position, request charging pad, wait until chargin pad is cleared, and go to LANDING.
        Land in place when no landing pad is found.
        """
        if self.start_of_state():
            
            if self.return_after_takeoff:
                self.state = TAKING_OFF
                return

            self.landing_position = [0, 0, 0]
            if self.battery_state == 1:
                self.state = CHARGING
                return
            
            if self.starting_pos is None:
                self.req_charge_pub_.publish(String(data=self.uri))
            self.target_pos = self.position[0], self.position[1], 1.0
        
        if CA_MODE != "off":
            self.desired_velocity = self.run_pid_controller(self.target_pos)
            self.final_velocity = self.CA_velocity

            if time.time() - self.time_last_CA_rec > 2:
                self.log.info("No CA message received for 2 second")
                self.land_in_place_and_set_state(ERROR, "No CA message received for 2 second")
                return
        else:
            self.final_velocity = self.run_pid_controller(self.target_pos)
        
        self.fly()

        if time.time() - self.state_timer > 2 and self.supervisor[IS_FLYING] == 0:
            return self.handle_error("reboot")

        if np.all(self.landing_position == [0, 0, 0]) and time.time() - self.state_timer > 5:
            if self.battery_state == 3:
                self.land_in_place_and_set_state(ERROR, "no landing pad and battery too low")
            else:
                self.land_in_place_and_set_state(WAITING, "no landing pad")
        elif np.any(self.landing_position != [0, 0, 0]):
            if self.landing_cleared or self.battery_state == 3:
                self.target_pos = self.landing_position[0], self.landing_position[1], LAND_H
                if self.distance_to(self.target_pos) < 0.10:
                    self.state = LANDING
            else:
                self.target_pos = WAIT_POS_RETURN
        else:
            if self.starting_pos is None:
                self.target_pos = WAIT_POS_RETURN
            else:
                self.target_pos = self.starting_pos[0], self.starting_pos[1], LAND_H
        

    def land(self):
        """
        Land.
        """
        if time.time() - self.time_last_vel_sent > 1/COMMAND_UR:
            self.land_and_set_state(self.landing_position, CHECK_CHARGING)
            self.time_last_vel_sent = time.time()
        
        if time.time() - self.state_timer > 10:
            return self.land_in_place_and_set_state(ERROR_HANDLING, ("LH stopped", CHECK_CHARGING))
        
        self.inflight_check()
    

    def land_in_place(self):
        """
        Land the drone in place.
        """
        if self.start_of_state():
            self.lh_state = 0

        self.send_velocity((0, 0, -0.5), NO_YAW)

        if self.position[2] < 0.12 or time.time() - self.state_timer > 4:
            if not self.send_command("stop motors", "motors stopped", tries=3):
                return self.handle_error("reboot", self.after_land_in_place_state)
            self.state = self.after_land_in_place_state
    

    def error_handling(self):
        """
        Try to handle the error that occured based on the given template. If the error can't be handled, go to ERROR.
        """
        if self.start_of_state:
            self.reboot_timout_timer = time.time()
        
        do_reboot, reboot_condition, reboot_timout, reboot_timout_state, return_condition, return_timout, return_timout_state, return_state = self.error_handling_config

        if reboot_condition():
            reboot_timout = -1
            if do_reboot:
                if not self.reboot():
                    self.error_handling_config[0] = True
                    self.state = ERROR_HANDLING
                    return
                if not self.enable_param_logging_full():
                    self.log.info("could not enable param logging in error handling state")
                    self.error_handling_config = [*self.error_handling_dict['reboot'], return_state]
                    self.state = ERROR_HANDLING
                    return
            else:
                if not self.enable_param_logging_standby():
                    self.log.info("could not disable posvel logging in error handling state")
                    self.error_handling_config = [*self.error_handling_dict['reboot'], return_state]
                    self.state = ERROR_HANDLING
                    return
            self.return_timout_timer = time.time()
            while not return_condition():
                if return_timout > 0 and time.time() - self.return_timout_timer > return_timout:
                    self.log.info("return timed out")
                    self.state = return_timout_state
                    return
                elif time.time() - self.return_timout_timer > self.error_handling_max_timeout:
                    self.log.info("max timeout reached for return condition")
                    self.error_handling_config[0] = True
                    self.state = ERROR_HANDLING
                    return
            if not do_reboot:
                if not self.enable_param_logging_full():
                    self.log.info("could not enable param logging in error handling state")
                    self.error_handling_config = [*self.error_handling_dict['reboot'], return_state]
                    self.state = ERROR_HANDLING
                    return
            self.state = return_state
            return
        
        if reboot_timout > 0 and time.time() - self.reboot_timout_timer > reboot_timout:
            self.log.info("reboot timed out")
            self.state = reboot_timout_state
            return
        elif time.time() - self.reboot_timout_timer > self.error_handling_max_timeout:
            self.log.info("max timeout reached for reboot condition")
            self.error_handling_config[0] = True
            self.state = ERROR_HANDLING
            return

    
    def error(self):
        """
        Error state for when the drone is not able to fix the error.
        """
        #publish the error message at start of state
        if self.start_of_state():
            self.log.info(self.error_msg)

            # send error message to the error topic
            self.error_pub.publish(String(data=self.error_msg))

            if not self.enable_param_logging_standby():
                self.log.info("could not disable posvel logging in error state")


    def disconnected(self):
        """
        The drone is diconnected.
        """
        if self.start_of_state():
            self.logging_state = {
                "pos vel": {int(1/COMMAND_UR*1000): False, int(1/COMMAND_UR_STANDBY*1000): False},
                "system state": {int(1/SYSTEM_PARAM_UR*1000): False},
            }
            self.log.info("Drone disconnected")
        
        if self.drone_response == "connected":
            self.state = INITIALISING
            self.controller_announcement = "radios ready"

    def shutdown(self):
        """
        Shutdown the drone.
        """
        raise SystemExit
    
    ############################# Main loop #############################

    def callback_loop(self):
        """
        Main loop that calls the state functions.
        """
        if self.prev_state != self.state:
            self.log.info(self.state)
            self.publish_state()
            self.prev_state = self.state
            self.state_start = True
            self.state_timer = time.time()

            if CA_MODE != "off":
                if self.state in (SWARMING, RETURNING, LANDING, LANDING_IN_PLACE): # states that require collision avoidance
                    self.enable_CA()
                else:
                    self.disable_CA()
        
        if self.drone_response == "disconnected":
            self.state = DISCONNECTED
        
        if self.state in (TAKING_OFF, SWARMING, RETURNING, LANDING):
            self.is_flying = True
        else:
            self.is_flying = False

        try:
            # call state function
            self.state_func[self.state]()
            self.state_start = False
        except Exception as e:
            self.error_msg = f"error in {self.state}: {e}"
            traceback.print_exc()
            self.state = ERROR


            
############################# Main function #############################
executor = None
def main(args=None):
    global executor
    rclpy.init(args=args)
    node = Drone()

    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        executor.shutdown(timeout_sec=0)

if __name__ == "__main__":
    main()
