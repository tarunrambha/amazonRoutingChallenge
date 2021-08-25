import os
from os import path
import math

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import sys, json, time
import pandas as pd
import multiprocessing as mp  # TODO remove from final version
from datetime import datetime, timedelta
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
from training import *

# TODO: Handle all division exceptions 

CONST_INFTY = 1000000.0  # If the TSP is not solved, the time is set to this value
CONST_TIMEOUT = 60  # Maximum time in seconds for solving TSP with time windows
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))  # Get Directory


def haversine(lat1, lng1, lat2, lng2):
    # distance between latitudes and longitudes in km
    d_lat = (lat2 - lat1) * math.pi / 180.0
    d_lon = (lng2 - lng1) * math.pi / 180.0

    # convert to radians
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(d_lat / 2), 2) +
         pow(math.sin(d_lon / 2), 2) *
         math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


class Route:
    # class constructor
    def __init__(self):
        self.index = 0
        self.key = 'NULL'
        self.station_code = 'NULL'
        self.date = 'NULL'
        self.departure_time = 'NULL'
        self.executor_capacity = 0.0
        self.score = 'NULL'
        self.stop = []  # list of stop objects

        self.stop_key_index_dict = {}  # dict of stops ordered by what was actually traversed
        self.travel_time_dict = {}  # travel time between stops in seconds

        # derived members
        self.num_stops = 0  # number of stops included in the route (includes depot)
        self.num_stops_tw = 0  # number of stops with time windows
        self.num_tw_violations = 0  # number of stops which violate time windows

        self.num_packages = 0  # number of packages to be delivered along the route
        self.num_packages_tw = 0  # number of packages with time windows on the route 
        self.num_scans_delivered = 0  # based on delivered status; not available during apply 
        self.num_scans_not_delivered = 0  # based on status; N/A for apply; compare with estimated values 
        self.num_packages_not_delivered = 0  # estimated num of packages not delivered due to scans 
        self.vol_packages_not_delivered = 0.0  # estimated vol of packages not delivered 

        self.packages_per_stop = 0.0  # number of packages per stop
        self.travel_time = 0.0  # total time taken to travel the actual route (ignores service times)
        self.service_time = 0.0  # sum of service times across all stops
        self.total_journey_time = 0.0  # sum of travel time plus service time
        self.filled_capacity = 0.0  # total volume of all the packages
        self.filled_capacity_percent = 0.0  # percentage of truck volume occupied by the packages
        self.volume_seconds_traveled = 0.0  # total route volume time in cm3 seconds (uses travel time not journey time)
        self.vst_ratio = 0.0  # volume seconds traveled divided by travel time x filled_capacity

        self.total_wait_time = 0.0  # if the truck is early at a stop with time windows that are met (0 is good)
        self.total_end_slack = 0.0  # difference between end time window and actual end time for time windows met
        self.total_max_slack = 0.0  # difference between end time window and start time window + service time
        self.slack_ratio = 0.0  # ratio of total end slack to total max slack

        # package level time window metrics (assumes every package at a stop is delivered first)
        # this is an approx; wait time may not require waiting since they can deliver other packages
        self.weighted_pkg_wait_time = 0.0  # total weighted (by volume) wait time for packages
        self.weighted_pkg_end_slack = 0.0  # total weighted (by volume) end slack for packages
        self.weighted_pkg_max_slack = 0.0  # total weighted (by volume) max slack for packages
        self.weighted_pkg_slack_ratio = 0.0  # total weighted (by volume) slack ratios for packages

        self.max_segment_time = 0.0  # maximum travel time between stops (excludes to and from origin)
        self.var_segment_time = 0.0  # variance of segment time
        self.segment_time_ratio = 0.0  # ratio compared to total journey time
        self.max_segment_dist = 0.0  # maximum distance traveled between stops (excludes to and from origin)
        self.var_segment_dist = 0.0  # variance of segment distance
        self.segment_dist_ratio = 0.0  # ratio compared to total distance
        self.total_dist = 0.0  # total haversine distance of route
        self.average_speed = 0.0  # total haversine distance / total time

        self.is_sequence_feasible = True  # boolean variable which records if the actual route is feasible or not
        self.weekday = 0  # Mon is 0 and Sun is 6
        self.is_weekend = False  # is set to true if weekday is Sat or Sun

        # zonal members
        self.zone_stop_dict = {}  # dictionary of zones as keys and stop list as values
        self.num_zones = 0  # number of stop zones
        self.num_zone_switches = 0  # number of times we switch to a different zone along a route
        self.switch_stop_ratio = 0  # ratio of number of zone switches to number of stops
        self.switch_zone_ratio = 0  # ratio of number of zone switches to number of zones (>= 1 ideal case is num zones)

        # TSP related members
        self.tsp_solver_status = 0  # 0 Not Solved 1 Success 2 No Soln 3 Timeout 4 Infeasible
        self.tsp_route_time = CONST_INFTY  # optimal TSP time as provided by the solver
        self.tsp_optimality_gap = 0.0  # between the TSP time and the actual route time as a percentage
        self.tsp_route_dict = {}  # dictionary of stops along the TSP tour which is used for scoring
        self.is_tsp_feasible = True  # checks if TSP is feasible or not

    class Stop:
        def __init__(self):
            self.key = 0
            self.order = -1
            self.lat = 0.0
            self.lng = 0.0
            self.proj_x = 0.0  # UTM projection of lat long
            self.proj_y = 0.0  # UTM projection of lat long
            self.type = 'NULL'
            self.zone_id = 'NULL'
            self.package_dict = {}  # package data is stored simply as a dictionary since we may not process it further

            # derived members
            self.planned_service_time = 0.0  # total planned service time for all packages at this stop in seconds
            self.is_tw_present = False  # indicator to check if there is a TW constraint or not
            self.is_tw_violated = False  # indicator for violations of time windows
            self.start_time_window = 'NULL'  # tightest start time for all packages at this stop
            self.end_time_window = 'NULL'  # tightest end time this stop (in seconds from departure time)
            self.actual_start_time = 0.0  # actual start time according to travel time data in seconds from start
            self.actual_end_time = 0.0  # actual end time using the planned service time

            self.wait_time = 0.0  # positive part of difference between start time window and actual start time
            self.slack_time = 0.0  # difference between end time window and actual end time 
            self.max_slack = 0.0  # difference between end time window and start time window + planned_service_time

            # package level time window metrics (assumes every package at the stop is delivered first)
            # this is an approx; wait time may not require waiting since they can deliver other packages
            self.weighted_pkg_wait_time = 0.0  # weighted (by volume) wait time for packages
            self.weighted_pkg_end_slack = 0.0  # weighted (by volume) end slack for packages
            self.weighted_pkg_max_slack = 0.0  # weighted (by volume) max slack for packages
            self.weighted_pkg_slack_ratio = 0.0  # weighted (by volume) slack ratios for packages

            self.num_packages = 0  # total number of packages
            self.num_packages_tw = 0  # number of packages with time windows at this stop
            self.num_scans_delivered = 0  # total number of packages delivered based on scans
            self.num_scans_not_delivered = 0  # total number of packages not delivered based on scans
            self.num_packages_not_delivered = 0  # number of packages not delivered due to time windows
            self.total_package_vol = 0.0  # total volume of all packages to be delivered at this stop
            self.vol_package_undelivered = 0.0  # volume of packages that could not be delivered

            # derived members related to TSP
            self.tsp_order = -1  # order of stops according to TSP
            self.actual_tsp_start_time = 0.0
            self.actual_tsp_end_time = 0.0
            self.is_tsp_feasible = True

        def __repr__(self):
            return "(" + str(self.key) + "," + str(self.order) + "," + str(self.zone_id) + ")"

    def compute_route_features(self):
        """This function computes several features for each route using the stop features
         that can be used for exploratory analysis"""
        self.weekday = self.date.weekday()
        if self.weekday >= 5:
            self.is_weekend = True
        self.num_stops = len(self.stop)
        self.travel_time = 0.0
        self.stop[0].actual_start_time = 0.0
        self.stop[0].actual_end_time = 0.0
        for i in range(len(self.stop)):  # add total number of packages across stops and their volume
            self.num_packages += self.stop[i].num_packages
            self.num_packages_tw += self.stop[i].num_packages_tw
            self.num_scans_delivered += self.stop[i].num_scans_delivered
            self.num_scans_not_delivered += self.stop[i].num_scans_not_delivered
            self.num_packages_not_delivered += self.stop[i].num_packages_not_delivered
            self.vol_packages_not_delivered += self.stop[i].vol_package_undelivered
            self.filled_capacity += self.stop[i].total_package_vol
            self.service_time += self.stop[i].planned_service_time
            self.total_wait_time += self.stop[i].wait_time
            self.total_end_slack += self.stop[i].slack_time
            self.total_max_slack += self.stop[i].max_slack

            self.weighted_pkg_wait_time += self.stop[i].weighted_pkg_wait_time
            self.weighted_pkg_end_slack += self.stop[i].weighted_pkg_end_slack
            self.weighted_pkg_max_slack += self.stop[i].weighted_pkg_max_slack
            self.weighted_pkg_slack_ratio += self.stop[i].weighted_pkg_slack_ratio

        self.packages_per_stop = self.num_packages / self.num_stops
        self.filled_capacity_percent = self.filled_capacity / self.executor_capacity
        current_volume = self.filled_capacity
        for i in range(len(self.stop) - 1):  # find arrival and departure times at all stops
            self.travel_time += self.travel_time_dict[self.stop[i].key][self.stop[i + 1].key]
            self.volume_seconds_traveled += current_volume * self.travel_time_dict[self.stop[i].key][
                self.stop[i + 1].key]
            current_volume -= self.stop[i + 1].total_package_vol

        self.travel_time += self.travel_time_dict[self.stop[self.num_stops - 1].key][
            self.stop[0].key]  # add travel time from last stop to depot

        segment_travel_times = [self.travel_time_dict[self.stop[i].key][self.stop[i + 1].key] for i in
                                range(1, len(self.stop) - 1)]
        self.max_segment_time = max(segment_travel_times)
        self.var_segment_time = np.var(segment_travel_times)
        self.segment_time_ratio = self.max_segment_time / self.travel_time

        segment_distances = [haversine(self.stop[i].lat, self.stop[i].lng, self.stop[i + 1].lat, self.stop[i + 1].lng)
                             for i in range(1, len(self.stop) - 1)]
        self.max_segment_dist = max(segment_distances)
        self.var_segment_dist = np.var(segment_distances)
        self.segment_dist_ratio = self.max_segment_dist / sum(segment_distances)
        self.total_dist = sum(segment_distances)
        self.total_dist += haversine(self.stop[0].lat, self.stop[0].lng, self.stop[1].lat, self.stop[1].lng)
        self.total_dist += haversine(self.stop[self.num_stops - 1].lat, self.stop[self.num_stops - 1].lng,
                                     self.stop[0].lat, self.stop[0].lng)
        self.average_speed = self.total_dist / self.travel_time

        self.vst_ratio = self.volume_seconds_traveled / (self.filled_capacity * self.travel_time)
        self.slack_ratio = self.total_end_slack / self.total_max_slack
        self.total_journey_time = self.travel_time + self.service_time

        unique_zone_list = [self.stop[i].zone_id for i in range(len(self.stop))]
        unique_zone_list = list(set(unique_zone_list))
        self.num_zones = len(unique_zone_list)
        for zone in unique_zone_list:
            self.zone_stop_dict[zone] = []
        for i in range(len(self.stop)):
            self.zone_stop_dict[self.stop[i].zone_id].append(self.stop[i].order)

        for i in range(len(self.stop) - 1):
            if self.stop[i].zone_id != self.stop[i + 1].zone_id:
                self.num_zone_switches += 1

        self.num_zone_switches += 1  # to account for last switch to depot
        self.switch_stop_ratio = self.num_zone_switches / self.num_stops
        self.switch_zone_ratio = self.num_zone_switches / self.num_zones

    def compute_stop_package_features(self):
        """This function computes several features for each stop that can be used for exploratory analysis"""
        # Replace NaN zone IDs with zone ID of nearest stop by travel time
        # if nearest also has NaN, then find the next nearest stop; Exclude depot station
        for i in range(len(self.stop)):
            if self.stop[i].zone_id != self.stop[i].zone_id and self.stop[i].type != 'Station':
                min_dist = 100000000
                nearest_zone_id = "Null"
                for j in range(len(self.stop)):
                    if j != i and self.stop[j].type != 'Station':
                        if self.travel_time_dict[self.stop[i].key][self.stop[j].key] < min_dist and self.stop[
                            j].zone_id == self.stop[j].zone_id:
                            min_dist = self.travel_time_dict[self.stop[i].key][self.stop[j].key]
                            nearest_zone_id = self.stop[j].zone_id
                self.stop[i].zone_id = nearest_zone_id
            elif self.stop[i].type == 'Station':
                self.stop[i].zone_id = 'Depot'

        self.stop[0].actual_start_time = 0.0
        self.stop[0].actual_end_time = 0.0
        for i in range(len(self.stop)):
            self.stop[i].order = i
            self.stop[i].num_packages = len(self.stop[i].package_dict.keys())
            self.stop[i].start_time_window = datetime.combine(self.date, self.departure_time)
            self.stop[i].end_time_window = (self.stop[i].start_time_window + timedelta(hours=24)).strftime(
                '%Y-%m-%d %H:%M:%S')
            self.stop[i].end_time_window = datetime.strptime(self.stop[i].end_time_window, '%Y-%m-%d %H:%M:%S')
            self.stop[i].is_tw_present = False

            for package_value in self.stop[i].package_dict.values():
                package_start_time = 0.0
                package_end_time = 86400.0
                package_value['is_tw_present'] = False
                if package_value['scan_status'] == "DELIVERED":
                    self.stop[i].num_scans_delivered += 1

                dimension_dict = package_value['dimensions']
                temp_prod = 1.0
                for value in dimension_dict.values():
                    temp_prod = temp_prod * value
                package_value['volume'] = temp_prod

                self.stop[i].total_package_vol += temp_prod
                self.stop[i].planned_service_time += package_value['planned_service_time_seconds']

                #  set the tightest start and end times at each stop
                if str(package_value['time_window']['start_time_utc']) != 'nan':
                    self.stop[i].is_tw_present = True
                    self.stop[i].num_packages_tw += 1
                    package_value['is_tw_present'] = True

                    package_start_time = datetime.strptime(str(package_value['time_window']['start_time_utc']),
                                                           '%Y-%m-%d %H:%M:%S')
                    if package_start_time > self.stop[i].start_time_window:
                        self.stop[i].start_time_window = package_start_time

                    package_start_time = datetime.strptime(str(package_value['time_window']['start_time_utc']),
                                                           '%Y-%m-%d %H:%M:%S')
                    package_start_time -= datetime.combine(self.date, self.departure_time)
                    package_start_time = package_start_time.total_seconds()

                if str(package_value['time_window']['end_time_utc']) != 'nan':
                    package_end_time = datetime.strptime(str(package_value['time_window']['end_time_utc']),
                                                         '%Y-%m-%d %H:%M:%S')
                    if package_end_time < self.stop[i].end_time_window:
                        self.stop[i].end_time_window = package_end_time

                    package_end_time = datetime.strptime(str(package_value['time_window']['end_time_utc']),
                                                         '%Y-%m-%d %H:%M:%S')
                    package_end_time -= datetime.combine(self.date, self.departure_time)
                    package_end_time = package_end_time.total_seconds()

                package_value['start_time'] = package_start_time
                package_value['end_time'] = package_end_time

            # calculate actual arrival and departure times
            if i > 0:
                self.stop[i].actual_start_time = self.stop[i - 1].actual_end_time + \
                                                 self.travel_time_dict[self.stop[i - 1].key][self.stop[i].key]
                self.stop[i].actual_end_time = self.stop[i].actual_start_time + self.stop[i].planned_service_time

            self.stop[i].num_scans_not_delivered = self.stop[i].num_packages - self.stop[i].num_scans_delivered

            # convert start and end time windows in seconds from departure time
            self.stop[i].start_time_window -= datetime.combine(self.date, self.departure_time)
            self.stop[i].end_time_window -= datetime.combine(self.date, self.departure_time)
            self.stop[i].start_time_window = self.stop[i].start_time_window.total_seconds()
            self.stop[i].end_time_window = self.stop[i].end_time_window.total_seconds()
            self.stop[0].end_time_window = 0.0  # no need to wait at the depot

            if i > 0:
                if self.stop[i].is_tw_present:
                    self.stop[i].wait_time = max(self.stop[i].start_time_window - self.stop[i].actual_start_time, 0)
                    self.stop[i].slack_time = max(self.stop[i].end_time_window - self.stop[i].actual_end_time, 0)
                    self.stop[i].max_slack = self.stop[i].end_time_window - (
                            self.stop[i].start_time_window + self.stop[i].planned_service_time)

            # calculate time window metrics at a package level
            for package_value in self.stop[i].package_dict.values():
                if package_value['is_tw_present']:
                    package_value['wait_time'] = max(package_value['start_time'] - self.stop[i].actual_start_time, 0)
                    package_value['end_slack'] = max(package_value['end_time'] - self.stop[i].actual_end_time, 0)
                    package_value['max_slack'] = package_value['end_time'] - (
                            package_value['start_time'] + package_value['planned_service_time_seconds'])
                    if package_value['max_slack'] > 0:  # avoid div by zero
                        package_value['slack_ratio'] = package_value['end_slack'] / package_value['max_slack']
                    else:
                        package_value['slack_ratio'] = 0

                    # aggregate these at a stop level
                    self.stop[i].weighted_pkg_wait_time += package_value['volume'] * package_value['wait_time']
                    self.stop[i].weighted_pkg_end_slack += package_value['volume'] * package_value['end_slack']
                    self.stop[i].weighted_pkg_max_slack += package_value['volume'] * package_value['max_slack']
                    self.stop[i].weighted_pkg_slack_ratio += package_value['volume'] * package_value['slack_ratio']

            # check for time window violations
            if self.stop[i].is_tw_present:
                self.num_stops_tw += 1
                if self.stop[i].actual_end_time > self.stop[i].end_time_window:  # waiting is allowed
                    self.stop[i].is_tw_violated = True
                    self.num_tw_violations += 1
                    self.is_sequence_feasible = False

            # New method for checking package infeasibility
            for package_value in self.stop[i].package_dict.values():
                if package_value['is_tw_present']:
                    if self.stop[i].actual_start_time + package_value['planned_service_time_seconds'] > package_value[
                        'end_time']:
                        self.stop[i].num_packages_not_delivered += 1
                        self.stop[i].vol_package_undelivered += package_value['volume']

    def display_route_data(self):
        """Function that prints minimal route data to check code progress. Full details are written to a CSV file."""
        print(self.index, self.key, self.score, self.num_stops_tw, self.num_packages_tw,
              self.num_packages_not_delivered,
              '%.2f' % self.vol_packages_not_delivered, '%.2f' % self.total_journey_time, '%.2f' % self.tsp_route_time,
              self.is_tsp_feasible, )
        # print(self.travel_time_dict)
        # if self.index == 162:
        #     for i in range(len(self.stop)):
        #         print(self.stop[i].key, self.stop[i].order, self.stop[i].type,
        #               self.stop[i].zone_id, self.stop[i].num_packages, self.stop[i].num_packages_tw,
        #               self.stop[i].num_scans_not_delivered, self.stop[i].num_packages_not_delivered,
        #               self.stop[i].total_package_vol, self.stop[i].vol_package_undelivered, self.stop[i].start_time_window,
        #               self.stop[i].end_time_window, self.stop[i].actual_start_time, self.stop[i].actual_end_time)
        #         print(json.dumps(self.stop[i].package_dict, indent=4))  # prints the entire package dictionary


def create_tsp_data(rt):
    """Stores the distance matrix for the stops on the route"""
    data = {}
    data['time_matrix'] = []
    row = []
    for i in range(len(rt.stop)):
        for j in range(len(rt.stop)):
            if rt.stop[i].key != rt.stop[j].key:
                row.append(int(math.floor(
                    rt.travel_time_dict[rt.stop[i].key][rt.stop[j].key] * 10 + rt.stop[i].planned_service_time * 10)))
            else:
                row.append(0)  # diagonal elements of the matrix are set to zeros
        data['time_matrix'].append(row)
        row = []

    data['service_time'] = []
    data['time_windows'] = []
    for i in range(len(rt.stop)):
        left_end_point = math.floor(rt.stop[i].start_time_window * 10)
        right_end_point = math.floor(rt.stop[i].end_time_window * 10 - rt.stop[i].planned_service_time * 10)
        data['time_windows'].append((int(left_end_point), int(right_end_point)))
        data['service_time'].append(rt.stop[i].planned_service_time)

    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_tsp_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} seconds'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'OR tools TSP route time: {} seconds\n'.format(route_distance)
    print(plan_output)


def set_tsp_dict(rt, manager, routing, solution):
    actual_route_dict = {}
    for i in range(len(rt.stop)):
        actual_route_dict[rt.stop[i].order] = rt.stop[i].key

    count = 0
    index = routing.Start(0)
    rt.tsp_route_dict = {}
    rt.tsp_route_dict[actual_route_dict[index]] = count
    rt.stop[index].tsp_order = count
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        if manager.IndexToNode(index) != 0:  # the revisit to depot can be ignored in the dictionary
            count += 1
            rt.tsp_route_dict[actual_route_dict[index]] = count
            rt.stop[index].tsp_order = count
    rt.tsp_route_dict = dict(sorted(rt.tsp_route_dict.items(), key=lambda item: item[1]))


def compute_tsp_tour(rt, data):
    """OR tools function that computes the TSP. Settings can be changed for time limits and solution method"""
    try:
        # print('Solving TSP now')
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic. Parameters are set to defaults. Check website for more options.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        rt.tsp_solver_status = routing.status()
        if rt.tsp_solver_status == 1:
            rt.tsp_route_time = solution.ObjectiveValue() / 10.0
            rt.tsp_optimality_gap = (rt.total_journey_time - rt.tsp_route_time) / rt.tsp_route_time
            # Save TSP tour in the form a dictionary for scoring
            set_tsp_dict(rt, manager, routing, solution)
        else:
            rt.tsp_route_time = CONST_INFTY
            rt.tsp_optimality_gap = (rt.total_journey_time - rt.tsp_route_time) / rt.tsp_route_time

        # # Print solution on console
        # if solution:
        #     print_tsp_solution(manager, routing, solution)
    except:
        print('Exception found while analysing TSP for route', rt.index)
        rt.tsp_route_time = CONST_INFTY
        rt.tsp_optimality_gap = (rt.total_journey_time - rt.tsp_route_time) / rt.tsp_route_time


def check_tsp_feasibility(rt):
    """Function that checks the time window feasibility of a route sequence TODO ordered dictionary required?"""
    for stop_key, order in rt.tsp_route_dict.items():
        # print(stop_key, order)
        # print(rt.stp_dict[stop_key])
        index = rt.stop_key_index_dict[stop_key]
        if order == 0:  # origin depot
            rt.stop[index].actual_tsp_start_time = 0.0
            rt.stop[index].actual_tsp_end_time = 0.0
            prev_stop_key = stop_key
            prev_index = rt.stop_key_index_dict[prev_stop_key]
        else:
            rt.stop[index].actual_tsp_start_time = rt.stop[prev_index].actual_tsp_end_time + \
                                                   rt.travel_time_dict[prev_stop_key][stop_key]
            rt.stop[index].actual_tsp_end_time = rt.stop[index].actual_tsp_start_time + rt.stop[
                index].planned_service_time
            prev_stop_key = stop_key
            prev_index = rt.stop_key_index_dict[prev_stop_key]
            if rt.stop[index].actual_tsp_end_time > rt.stop[index].end_time_window:
                rt.stop[index].is_tsp_feasible = False
                rt.is_tsp_feasible = False


def read_route_data():
    """Reads the JSON files and populates class variables"""
    route_list = []
    temp_route = Route()

    training_routes_path = path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
    with open(training_routes_path) as f:
        route_data = json.load(f)

    training_sequence_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
    with open(training_sequence_path) as f:
        actual_sequences = json.load(f)

    training_package_path = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
    with open(training_package_path) as f:
        package_data = json.load(f)

    training_traveltime_path = path.join(BASE_DIR, 'data/model_build_inputs/travel_times.json')
    with open(training_traveltime_path) as f:
        travel_time = json.load(f)

    count = 0
    key_index_dict = {}
    for key in route_data.keys():
        temp_route.index = count
        temp_route.key = key
        key_index_dict[key] = count
        temp_route.station_code = route_data[key]['station_code']
        temp_route.date = datetime.strptime(route_data[key]['date_YYYY_MM_DD'], '%Y-%m-%d').date()
        temp_route.departure_time = datetime.strptime(route_data[key]['departure_time_utc'], '%H:%M:%S').time()
        temp_route.executor_capacity = route_data[key]['executor_capacity_cm3']
        temp_route.score = route_data[key]['route_score']

        # sort stops based on the actual order in which they have were traversed and store them in the nested class
        stop_dict = route_data[key]['stops']
        temp_stop = temp_route.Stop()
        sorted_stop_dict = dict(sorted(actual_sequences[key]['actual'].items(), key=lambda item: item[1]))
        temp_route.stop_key_index_dict = dict(sorted_stop_dict)
        for stop_key in sorted_stop_dict.keys():
            temp_stop.key = stop_key
            temp_stop.lat = stop_dict[stop_key]['lat']
            temp_stop.lng = stop_dict[stop_key]['lng']
            temp_stop.type = stop_dict[stop_key]['type']
            temp_stop.zone_id = stop_dict[stop_key]['zone_id']
            temp_stop.package_dict = package_data[key][stop_key]
            temp_route.stop.append(temp_stop)
            temp_stop = temp_route.Stop()
        temp_route.travel_time_dict = travel_time[key]

        route_list.append(temp_route)
        count += 1
        temp_route = Route()

    return route_list, key_index_dict, travel_time


def read_training_route_data():
    """Reads the JSON files and populates class variables"""
    route_list = []
    temp_route = Route()

    training_routes_path = path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
    with open(training_routes_path) as f:
        route_data = json.load(f)

    training_sequence_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
    with open(training_sequence_path) as f:
        actual_sequences = json.load(f)

    training_package_path = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
    with open(training_package_path) as f:
        package_data = json.load(f)

    training_traveltime_path = path.join(BASE_DIR, 'data/model_build_inputs/travel_times.json')
    with open(training_traveltime_path) as f:
        travel_time = json.load(f)

    training_invalid_score = path.join(BASE_DIR, 'data/model_build_inputs/invalid_sequence_scores.json')
    with open(training_invalid_score) as f:
        invalid_score = json.load(f)

    count = 0
    random.seed(17)
    new_route_data = {}
    new_package_data = {}
    new_travel_time = {}

    new_actual_sequence = {}
    new_invalid_sequence_score = {}

    key_index_dict = {}

    testing_keys = []  # this includes data minus a small fraction of high routes
    for key in route_data.keys():
        temp_route.index = count
        temp_route.key = key
        key_index_dict[key] = count
        temp_route.station_code = route_data[key]['station_code']
        temp_route.date = datetime.strptime(route_data[key]['date_YYYY_MM_DD'], '%Y-%m-%d').date()
        temp_route.departure_time = datetime.strptime(route_data[key]['departure_time_utc'], '%H:%M:%S').time()
        temp_route.executor_capacity = route_data[key]['executor_capacity_cm3']
        temp_route.score = route_data[key]['route_score']

        # sort stops based on the actual order in which they have were traversed and store them in the nested class
        stop_dict = route_data[key]['stops']
        temp_stop = temp_route.Stop()
        sorted_stop_dict = dict(sorted(actual_sequences[key]['actual'].items(), key=lambda item: item[1]))
        temp_route.stop_key_index_dict = dict(sorted_stop_dict)
        for stop_key in sorted_stop_dict.keys():
            temp_stop.key = stop_key
            temp_stop.lat = stop_dict[stop_key]['lat']
            temp_stop.lng = stop_dict[stop_key]['lng']
            temp_stop.type = stop_dict[stop_key]['type']
            temp_stop.zone_id = stop_dict[stop_key]['zone_id']
            temp_stop.package_dict = package_data[key][stop_key]
            temp_route.stop.append(temp_stop)
            temp_stop = temp_route.Stop()
        temp_route.travel_time_dict = travel_time[key]

        route_list.append(temp_route)
        count += 1
        temp_route = Route()

        # TODO: Check if we are using this high subset of routes in the final version
        if route_data[key]['route_score'] == 'High' and random.uniform(0, 1) < 0.15:  # testing data
            testing_keys.append(key)
            new_route_data[key] = route_data[key]
            new_package_data[key] = package_data[key]
            new_travel_time[key] = travel_time[key]
            new_actual_sequence[key] = actual_sequences[key]
            new_invalid_sequence_score[key] = invalid_score[key]

    # TODO: Remove this in the final version. We pull the apply data from their new routes.
    with open(path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json'), "w") as outfile:
        json.dump(new_route_data, outfile)
    with open(path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json'), "w") as outfile:
        json.dump(new_package_data, outfile)
    with open(path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json'), "w") as outfile:
        json.dump(new_travel_time, outfile)
    with open(path.join(BASE_DIR, 'data/model_score_inputs/new_actual_sequences.json'), "w") as outfile:
        json.dump(new_actual_sequence, outfile)
    with open(path.join(BASE_DIR, 'data/model_score_inputs/new_invalid_sequence_scores.json'), "w") as outfile:
        json.dump(new_invalid_sequence_score, outfile)

    # print(testing_keys)
    return route_list, key_index_dict, travel_time, testing_keys


def output_route_df(route, testing_keys):
    """Outputs processed data to a CSV file"""
    row_list_training = []
    row_list_testing = []
    row_list_full = []
    for rt in route:
        temp_dict = {'Index': rt.index,
                     'Key': rt.key,
                     'Station_Code': rt.station_code,
                     'Date': rt.date,
                     'Departure_Time': rt.departure_time,
                     'Executor_Capacity': rt.executor_capacity,
                     'Score': rt.score,
                     'Num_Stops': rt.num_stops,
                     'Num_Stops_TW': rt.num_stops_tw,
                     'Num_Stops_TW_Violations': rt.num_tw_violations,
                     'Num_Packages': rt.num_packages,
                     'Num_Packages_TW': rt.num_packages_tw,
                     'Num_Scans_Delivered': rt.num_scans_delivered,
                     'Num_Scans_Not_Delivered': rt.num_scans_not_delivered,
                     'Num_Packages_Not_Delivered': rt.num_packages_not_delivered,
                     'Vol_Packages_Not_Delivered': rt.vol_packages_not_delivered,
                     'Packages_Per_Stop': rt.packages_per_stop,
                     'Total_Travel_Time': rt.travel_time,
                     'Total_Service_Time': rt.service_time,
                     'Total_Journey_Time': rt.total_journey_time,
                     'Filled_Capacity': rt.filled_capacity,
                     'Filled_Capacity_Percent': rt.filled_capacity_percent,
                     'Volume_Seconds_Traveled': rt.volume_seconds_traveled,
                     'VST_Ratio': rt.vst_ratio,
                     'Total_Wait_Time': rt.total_wait_time,
                     'Total_End_Slack': rt.total_end_slack,
                     'Total_Max_Slack': rt.total_max_slack,
                     'Slack_Ratio': rt.slack_ratio,
                     'Weighted_Pkg_Wait_Time': rt.weighted_pkg_wait_time,
                     'Weighted_Pkg_End_Slack': rt.weighted_pkg_end_slack,
                     'Weighted_Pkg_Max_Slack': rt.weighted_pkg_max_slack,
                     'Weighted_Pkg_Slack_Ratio': rt.weighted_pkg_slack_ratio,
                     'Num_Zones': rt.num_zones,
                     'Num_Zone_Switches': rt.num_zone_switches,
                     'Switch_Stop_Ratio': rt.switch_stop_ratio,
                     'Switch_Zone_Ratio': rt.switch_zone_ratio,
                     'Max_Segment_Time': rt.max_segment_time,
                     'Variance_Segment_Time': rt.var_segment_time,
                     'Segment_Time_Ratio': rt.segment_time_ratio,
                     'Max_Segment_Dist': rt.max_segment_dist,
                     'Variance_Segment_Dist': rt.var_segment_dist,
                     'Segment_Dist_Ratio': rt.segment_dist_ratio,
                     'Total_Dist': rt.total_dist,
                     'Average_Speed': rt.average_speed,
                     'Is_Weekend': int(rt.is_weekend == True),
                     'Is_Sequence_Feasible': int(rt.is_sequence_feasible == True),
                     'TSP_Solver_Status': rt.tsp_solver_status,
                     'TSP_Route_Time': rt.tsp_route_time,
                     'TSP_Optimality_Gap': rt.tsp_optimality_gap,
                     'Is_TSP_Feasible': int(rt.is_tsp_feasible == True)
                     }
        if rt.key in testing_keys:
            row_list_testing.append(temp_dict)
        else:
            row_list_training.append(temp_dict)
        row_list_full.append(temp_dict)

    df = pd.DataFrame(row_list_testing, columns=temp_dict.keys())
    output_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_testing_jun18.csv')
    df.to_csv(output_path, index=False)

    df = pd.DataFrame(row_list_training, columns=temp_dict.keys())
    output_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_training_jun18.csv')
    df.to_csv(output_path, index=False)

    df = pd.DataFrame(row_list_full, columns=temp_dict.keys())
    output_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_full_jun18.csv')
    df.to_csv(output_path, index=False)


def output_stop_df(route, testing_keys):
    """Function to create a CSV file with lat long locations and order of nodes visited for visualization"""
    row_list_training = []
    row_list_testing = []
    for rt in route:
        for i in range(len(rt.stop)):
            temp_dict = {'Route_Index': rt.index,
                         'Route_Key': rt.key,
                         'Stop_Key': rt.stop[i].key,
                         'Stop_Order': rt.stop[i].order,
                         'Latitude': rt.stop[i].lat,
                         'Longitude': rt.stop[i].lng,
                         'X_Coordinate': '%.4f' % rt.stop[i].proj_x,
                         'Y_Coordinate': '%.4f' % rt.stop[i].proj_y,
                         'Zone_ID': rt.stop[i].zone_id,
                         'Planned_Service_Time': rt.stop[i].planned_service_time,
                         'TW_Constraint': rt.stop[i].is_tw_present,
                         'TW_Violated': rt.stop[i].is_tw_violated,
                         'Start_Time_Window': rt.stop[i].start_time_window,
                         'End_Time_Window': rt.stop[i].end_time_window,
                         'Actual_Start_Time': rt.stop[i].actual_start_time,
                         'Actual_End_Time': rt.stop[i].actual_end_time,
                         'Wait_Time': rt.stop[i].wait_time,
                         'Slack_Time': rt.stop[i].slack_time,
                         'Max_Slack': rt.stop[i].max_slack,
                         'Num_Packages': rt.stop[i].num_packages,
                         'Num_Packages_TW': rt.stop[i].num_packages_tw,
                         'Num_Scans_Delivered': rt.stop[i].num_scans_delivered,
                         'Num_Scans_Not_Delivered': rt.stop[i].num_scans_not_delivered,
                         'Num_Packages_Not_Delivered': rt.stop[i].num_packages_not_delivered,
                         'Total_Package_Volume': rt.stop[i].total_package_vol,
                         'Volume_Undelivered': rt.stop[i].vol_package_undelivered,
                         'TSP_Order': rt.stop[i].tsp_order,
                         'Actual_TSP_Start_Time': rt.stop[i].actual_tsp_start_time,
                         'Actual_TSP_End_Time': rt.stop[i].actual_tsp_end_time,
                         'Is_TSP_Feasible': rt.stop[i].is_tsp_feasible}
            if rt.key in testing_keys:
                row_list_testing.append(temp_dict)
            else:
                row_list_training.append(temp_dict)

    df = pd.DataFrame(row_list_testing, columns=temp_dict.keys())
    output_path = path.join(BASE_DIR, 'data/model_build_outputs/stop_summary_testing_jun18.csv')
    df.to_csv(output_path, index=False)

    df = pd.DataFrame(row_list_training, columns=temp_dict.keys())
    output_path = path.join(BASE_DIR, 'data/model_build_outputs/stop_summary_training_jun18.csv')
    df.to_csv(output_path, index=False)


def core_block(rt):
    rt.compute_stop_package_features()
    rt.compute_route_features()
    tsp_data = create_tsp_data(rt)
    compute_tsp_tour(rt, tsp_data)
    check_tsp_feasibility(rt)
    rt.display_route_data()

    return rt


if __name__ == '__main__':
    # begin = time.time()
    # num_cpu = int(mp.cpu_count() * 0.75)  # use 75% of CPUs on the machine

    # route, route_key_index_dict, travel_time_json, testing_keys = read_training_route_data()
    # # route, route_key_index_dict, travel_time_json = read_route_data()
    # print("Data reading complete... in", (time.time() - begin), "seconds")

    # try:
    #     """parallel code"""
    #     begin = time.time()
    #     print('Beginning parallel block...')
    #     with mp.Pool(num_cpu) as pool:
    #         results = pool.map(core_block, [rt for rt in route])
    #     pool.close()
    #     pool.join()

    #     print('Parallel block complete...')
    #     print('Time for this block is %.2f minutes' % ((time.time() - begin) / 60))
    # except:
    #     """serial code"""
    #     begin = time.time()
    #     print('Beginning serial block...')
    #     results = [core_block(rt) for rt in route]

    #     print('Serial block complete...')
    #     print('Time for this block is %.2f minutes' % ((time.time() - begin) / 60))

    # output_route_df(results, testing_keys)
    # output_stop_df(results, testing_keys)

    print('Starting model training...')
    train_xgboost_classifier()

    print('Model build complete...')
