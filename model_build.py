import os
from os import path
import json
import time
import math
import joblib  # loading saved sklearn models

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from datetime import datetime, timedelta
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.pywrapcp import SolutionCollector

import matplotlib.pyplot as plt

CONST_INFTY = 1000000.0  # If the TSP is not solved, the time is set to this value
CONST_TIMEOUT = 60  # Maximum time in seconds for solving TSP with time windows
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

PRINT_VERBOSITY = -1  # 1 high print, 0 min print per route, -1 no print (1 is suggested only for debug mode)
PRINT_CSV_FILES = False

scores_list = [] # list of all route scores for easy plotting
prob_high_list = [] # list of all prob_high values for easy plotting

model_filename_w_path = path.join(BASE_DIR, 'data/model_build_outputs/xgboost_compressed.joblib')
loaded_model = joblib.load(model_filename_w_path)

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
        self.stop = {}  # dictionary of stops (package information is included in it)
        self.travel_time_dict = {}  # travel time between stops in seconds

        # proposed route members
        self.stop_order_dict = {}  # dict of stops with values equal to order of stops visited
        self.stop_sequence_dict = {}  # dict of stops where key is the order and value is the stop keys of the data
        self.prob_high = 0.0  # probability that the route sequence is rated 'High'
        self.best_prob_high = -1.0  # stores highest prob of being rated high
        self.best_stop_order_score = 10000.0
        self.best_stop_order_dict = {}  # stores the associated stop_order_dict of order with highest prob
        self.unique_stop_order_string = "NA"  # temporary string that uniquely identifies different stop-order tours

        # features that need to be recomputed
        self.num_tw_violations = 0  # number of stops which violate time windows
        self.num_packages_not_delivered = 0.0  # number of packages not delivered
        self.vol_packages_not_delivered = 0.0  # volume of packages not delivered
        self.is_sequence_feasible = True  # boolean variable which records if the proposed route is feasible or not

        self.travel_time = 0.0  # total time taken to travel the actual route (ignores service times)
        self.total_journey_time = 0.0  # sum of travel times plus service time
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
        self.weighted_pkg_max_slack = 0.0 # total weighted (by volume) max slack for packages
        self.weighted_pkg_slack_ratio = 0.0 # total weighted (by volume) slack ratios for packages

        self.max_segment_dist = 0.0  # maximum distance traveled between stops (excludes to and from origin)
        self.var_segment_dist = 0.0  # variance of segment distance
        self.segment_dist_ratio = 0.0  # ratio compared to total distance
        self.total_dist = 0.0  # total haversine distance of route
        self.average_speed = 0.0  # total haversine distance / total time

        self.max_segment_time = 0.0  # maximum travel time between stops (excludes to and from origin)
        self.var_segment_time = 0.0  # variance of segment time
        self.segment_time_ratio = 0.0  # ratio compared to total journey time

        # zonal members (will always be constant if we do zone-based TSPs)
        self.num_zone_switches = 0  # number of times we switch to a different zone along a route
        self.switch_stop_ratio = 0  # ratio of num of zone switches to num of stops
        self.switch_zone_ratio = 0  # ratio of num of zone switches to num of zones (>= 1 ideal case is num zones)

        # comparison with unconstrained TSP
        self.tsp_route_time = CONST_INFTY  # optimal TSP time as provided by the solver
        self.tsp_optimality_gap = 0.0  # between the TSP time and the actual route time as a percentage
        self.is_tsp_feasible = True  # checks if TSP is feasible or not

        # static route features (they don't depend on the sequence)
        self.num_stops = 0  # number of stops included in the route (includes depot)
        self.num_stops_tw = 0  # number of stops with time windows
        self.num_packages = 0  # number of packages to be delivered along the route
        self.num_packages_tw = 0  # number of packages with time windows
        self.packages_per_stop = 0.0  # number of packages per stop
        self.executor_capacity = 0.0
        self.filled_capacity = 0.0  # total volume of all the packages
        self.filled_capacity_percent = 0.0  # percentage of truck volume occupied by the packages
        self.service_time = 0.0  # sum of service times across all stops
        self.weekday = 0  # Mon is 0 and Sun is 6
        self.is_weekend = False  # is set to true if weekday is Sat or Sun

        # auxiliary members
        self.origin_stop_key = 'NULL'  # key of the origin stop #
        self.zone_stop_dict = {}  # dictionary of zones as keys and stop key list as values
        self.is_stop_order_feasible = True
        self.stop_order_score = 100000.0;
        self.num_zones = 0  # number of stop zones
        self.tsp_solver_status = 0  # 0 Not Solved 1 Success 2 No Soln 3 Timeout 4 Infeasible
        self.zone_id_to_key = {}  # dictionary of unique ID of zone to its key; useful for zone-based TSP
        self.zone_tsp_tour = []  # list of zone IDs for zone-based TSP
        self.zone_id_to_ordered_stop_dict = {}  # dictionary of unique zone ID as key to ordered stop IDs list as values after within zone TSP

    def display_route_data(self):
        """Function that prints minimal route data to check features."""
        # print('Route features: (key, num_stops_tw, num_tw_violations, num_packages_not_delivered, travel_time, total_journey_time, best_prob_high)')
        print(self.index, self.key, self.num_stops_tw, self.num_tw_violations, self.num_packages_not_delivered,
              '%.2f' % self.travel_time,
              '%.2f' % self.total_journey_time, self.best_prob_high)

    def compute_static_route_features(self):
        """Calculates aggregate metrics from static stop features"""
        self.weekday = self.date.weekday()
        if self.weekday >= 5:
            self.is_weekend = True
        for key in self.stop.keys():
            self.num_stops = len(self.stop.keys())
            self.num_packages += self.stop[key]['num_packages']
            self.num_packages_tw += self.stop[key]['num_packages_tw']
            self.filled_capacity += self.stop[key]['total_package_vol']
            if self.stop[key]['is_tw_present']:
                self.num_stops_tw += 1
            self.service_time += self.stop[key]['planned_service_time']

        # Replace NaN zone IDs with zone ID of nearest stop by travel time
        # if nearest also has NaN, then find the next nearest stop; Exclude depot station
        for key in self.stop.keys():
            if self.stop[key]['zone_id'] != self.stop[key]['zone_id'] and self.stop[key]['type'] != 'Station':
                min_dist = 100000000
                nearest_zone_id = "Null"
                for s in self.stop.keys():
                    if s != key and self.stop[s]['type'] != 'Station':
                        if self.travel_time_dict[key][s] < min_dist and self.stop[s]['zone_id'] == self.stop[s][
                            'zone_id']:
                            min_dist = self.travel_time_dict[key][s]
                            nearest_zone_id = self.stop[s]['zone_id']
                self.stop[key]['zone_id'] = nearest_zone_id
            elif self.stop[key]['type'] == 'Station':
                self.stop[key]['zone_id'] = 'Depot'

        self.packages_per_stop = self.num_packages / self.num_stops
        self.filled_capacity_percent = self.filled_capacity / self.executor_capacity

        unique_zone_list = [self.stop[key]['zone_id'] for key in self.stop.keys()]
        unique_zone_list = list(set(unique_zone_list))
        self.num_zones = len(unique_zone_list)
        for zone in unique_zone_list:
            self.zone_stop_dict[zone] = []
        for key in self.stop.keys():
            self.zone_stop_dict[self.stop[key]['zone_id']].append(key)

    def compute_static_stop_features(self):
        """Calculates metrics such as package volume, number of packages, etc."""
        for stop_key, stop_value in self.stop.items():  # key is stop key and value is a dictionary of stop features
            stop_value['num_packages'] = len(stop_value['packages'].keys())
            stop_value['num_packages_tw'] = 0
            stop_value['total_package_vol'] = 0.0
            stop_value['planned_service_time'] = 0.0
            stop_value['is_tw_present'] = False
            stop_value['start_time_window'] = 0.0
            stop_value['end_time_window'] = 0.0 if stop_key == self.origin_stop_key else 86400.0  # +24 hour time window
            stop_value['wait_time'] = 0.0
            stop_value['slack_time'] = 0.0
            stop_value['max_slack'] = 0.0

            for package_value in stop_value['packages'].values():
                package_start_time = 0.0
                package_end_time = 86400.0
                package_value['is_tw_present'] = False
                dimension_dict = package_value['dimensions']
                temp_prod = 1.0
                for value in dimension_dict.values():
                    temp_prod = temp_prod * value
                package_value['volume'] = temp_prod

                stop_value['total_package_vol'] += temp_prod
                stop_value['planned_service_time'] += package_value['planned_service_time_seconds']

                if str(package_value['time_window']['start_time_utc']) != 'nan':
                    stop_value['is_tw_present'] = True
                    stop_value['num_packages_tw'] += 1
                    package_value['is_tw_present'] = True
                    package_start_time = datetime.strptime(str(package_value['time_window']['start_time_utc']),
                                                           '%Y-%m-%d %H:%M:%S')
                    package_start_time -= datetime.combine(self.date, self.departure_time)
                    package_start_time = package_start_time.total_seconds()
                    if package_start_time > stop_value['start_time_window']:
                        stop_value['start_time_window'] = package_start_time

                if str(package_value['time_window']['end_time_utc']) != 'nan':
                    package_end_time = datetime.strptime(str(package_value['time_window']['end_time_utc']),
                                                         '%Y-%m-%d %H:%M:%S')
                    package_end_time -= datetime.combine(self.date, self.departure_time)
                    package_end_time = package_end_time.total_seconds()
                    if package_end_time < stop_value['end_time_window']:
                        stop_value['end_time_window'] = package_end_time

                package_value['start_time'] = package_start_time
                package_value['end_time'] = package_end_time

    def reset_non_static_features(self):  # TODO: Check if all variables are reset
        self.num_tw_violations = 0  # number of stops which violate time windows
        self.num_packages_not_delivered = 0.0  # number of packages not delivered
        self.vol_packages_not_delivered = 0.0  # volume of packages not delivered
        self.is_sequence_feasible = True  # boolean variable which records if the proposed route is feasible or not

        self.travel_time = 0.0  # total time taken to travel the actual route (ignores service times)
        self.total_journey_time = 0.0  # sum of travel times plus service time
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
        self.weighted_pkg_max_slack = 0.0 # total weighted (by volume) max slack for packages
        self.weighted_pkg_slack_ratio = 0.0 # total weighted (by volume) slack ratios for packages

        self.max_segment_dist = 0.0  # maximum distance traveled between stops (excludes to and from origin)
        self.var_segment_dist = 0.0  # variance of segment distance
        self.segment_dist_ratio = 0.0  # ratio compared to total distance
        self.total_dist = 0.0  # total haversine distance of route
        self.average_speed = 0.0  # total haversine distance / total time

        self.max_segment_time = 0.0  # maximum travel time between stops (excludes to and from origin)
        self.var_segment_time = 0.0  # variance of segment time
        self.segment_time_ratio = 0.0  # ratio compared to total journey time

        # zonal members (will always be constant if we do zone-based TSPs)
        self.num_zone_switches = 0  # number of times we switch to a different zone along a route
        self.switch_stop_ratio = 0  # ratio of num of zone switches to num of stops
        self.switch_zone_ratio = 0  # ratio of num of zone switches to num of zones (>= 1 ideal case is num zones)

        self.tsp_route_time = CONST_INFTY  # optimal TSP time as provided by the solver
        self.tsp_optimality_gap = 0.0  # between the TSP time and the actual route time as a percentage

    def compute_route_features(self):
        for key, value in self.stop_order_dict.items():
            if self.stop[key]['is_tw_violated']:
                self.num_tw_violations += 1
                self.is_sequence_feasible = False

        current_volume = self.filled_capacity
        for key, value in self.stop_order_dict.items():
            if value < self.num_stops - 1:
                next_key = self.stop_sequence_dict[value + 1]
                self.travel_time += self.travel_time_dict[key][next_key]
                self.volume_seconds_traveled += current_volume * self.travel_time_dict[key][next_key]
                current_volume -= self.stop[next_key]['total_package_vol']

        self.travel_time += self.travel_time_dict[self.stop_sequence_dict[self.num_stops - 1]][
            self.origin_stop_key]  # add travel time from last stop to depot

        self.total_journey_time = self.travel_time + self.service_time
        self.tsp_optimality_gap = (self.total_journey_time - self.tsp_route_time) / self.tsp_route_time
        self.vst_ratio = self.volume_seconds_traveled / (self.filled_capacity * self.travel_time)

        # add tw-specific features
        for stop_value in self.stop.values():
            self.total_wait_time += stop_value['wait_time']
            self.total_end_slack += stop_value['slack_time']
            self.total_max_slack += stop_value['max_slack']

            self.weighted_pkg_wait_time += stop_value['weighted_pkg_wait_time']
            self.weighted_pkg_end_slack += stop_value['weighted_pkg_end_slack']
            self.weighted_pkg_max_slack += stop_value['weighted_pkg_max_slack']
            self.weighted_pkg_slack_ratio += stop_value['weighted_pkg_slack_ratio']

        self.slack_ratio = self.total_end_slack / self.total_max_slack

        # set distance-specific features 
        segment_distances = []
        for key, value in self.stop_order_dict.items():
            if 0 < value < self.num_stops - 1:
                next_key = self.stop_sequence_dict[value + 1]
                segment_distances.append(
                    haversine(self.stop[key]['lat'], self.stop[key]['lng'], self.stop[next_key]['lat'],
                              self.stop[next_key]['lng']))

        self.max_segment_dist = max(segment_distances)
        self.var_segment_dist = np.var(segment_distances)
        self.segment_dist_ratio = self.max_segment_dist / sum(segment_distances)
        self.total_dist = sum(segment_distances)
        self.total_dist += haversine(self.stop[self.origin_stop_key]['lat'],
                                     self.stop[self.origin_stop_key]['lng'],
                                     self.stop[self.stop_sequence_dict[1]]['lat'],
                                     self.stop[self.stop_sequence_dict[1]]['lng'])
        self.total_dist += haversine(self.stop[self.stop_sequence_dict[self.num_stops - 1]]['lat'],
                                     self.stop[self.stop_sequence_dict[self.num_stops - 1]]['lng'],
                                     self.stop[self.origin_stop_key]['lat'],
                                     self.stop[self.origin_stop_key]['lng'])
        self.average_speed = self.total_dist / self.travel_time

        # set segment travel time specific features
        segment_travel_times = []
        for key, value in self.stop_order_dict.items():
            if 0 < value < self.num_stops - 1:
                next_key = self.stop_sequence_dict[value + 1]
                segment_travel_times.append(self.travel_time_dict[key][next_key])
        self.max_segment_time = max(segment_travel_times)
        self.var_segment_time = np.var(segment_travel_times)
        self.segment_time_ratio = self.max_segment_time / self.travel_time

        # New method for checking package infeasibility
        for stop_value in self.stop.values():
            for package_value in stop_value['packages'].values():
                if package_value['is_tw_present']:
                    if stop_value['actual_start_time'] + package_value['planned_service_time_seconds'] > package_value[
                        'end_time']:
                        self.num_packages_not_delivered += 1
                        self.vol_packages_not_delivered += package_value['volume']

        # zone-based features
        for key, value in self.stop_order_dict.items():
            if value < self.num_stops - 1:
                next_key = self.stop_sequence_dict[value + 1]
                if self.stop[key]['zone_id'] != self.stop[next_key]['zone_id']:
                    self.num_zone_switches += 1

        self.num_zone_switches += 1  # to account for last switch to depot
        self.switch_stop_ratio = self.num_zone_switches / self.num_stops
        self.switch_zone_ratio = self.num_zone_switches / self.num_zones

    def compute_stop_features(self):
        self.reset_non_static_features()
        # update the actual start and departure times
        self.stop[self.origin_stop_key]['actual_start_time'] = 0.0
        self.stop[self.origin_stop_key]['actual_end_time'] = 0.0
        for key, value in self.stop_order_dict.items():
            self.stop[key]['actual_start_time'] = 0
            self.stop[key]['actual_end_time'] = 0
        for key, value in self.stop_order_dict.items():
            if value > 0:
                previous_key = self.stop_sequence_dict[value - 1]
                self.stop[key]['actual_start_time'] = self.stop[previous_key]['actual_end_time'] + \
                                                      self.travel_time_dict[previous_key][key]
                self.stop[key]['actual_end_time'] = self.stop[key]['actual_start_time'] + self.stop[key][
                    'planned_service_time']

        # check for time window violations
        for key in self.stop_order_dict.keys():
            self.stop[key]['is_tw_violated'] = False
            if self.stop[key]['actual_end_time'] > self.stop[key]['end_time_window']:
                self.stop[key]['is_tw_violated'] = True
                self.is_stop_order_feasible = False

        # calculate time window metrics at a package level
        for stop_value in self.stop.values():
            stop_value['weighted_pkg_wait_time'] = 0.0
            stop_value['weighted_pkg_end_slack'] = 0.0
            stop_value['weighted_pkg_max_slack'] = 0.0
            stop_value['weighted_pkg_slack_ratio'] = 0.0

            for package_value in stop_value['packages'].values():
                if package_value['is_tw_present']:
                    package_value['wait_time'] = max(package_value['start_time'] - stop_value['actual_start_time'], 0)
                    package_value['end_slack'] = max(package_value['end_time'] - stop_value['actual_end_time'], 0)
                    package_value['max_slack'] = package_value['end_time'] - (package_value['start_time'] + package_value['planned_service_time_seconds'])
                    if package_value['max_slack'] > 0:  # avoid div by zero for depot
                        package_value['slack_ratio'] = package_value['end_slack'] / package_value['max_slack']
                    else:
                        package_value['slack_ratio'] = 0
            
                    # aggregate these at a stop level
                    stop_value['weighted_pkg_wait_time'] += package_value['volume'] * package_value['wait_time']
                    stop_value['weighted_pkg_end_slack'] += package_value['volume'] * package_value['end_slack']
                    stop_value['weighted_pkg_max_slack'] += package_value['volume'] * package_value['max_slack']
                    stop_value['weighted_pkg_slack_ratio'] += package_value['volume'] * package_value['slack_ratio']

        # populate time violation features
        for stop_value in self.stop.values():
            if stop_value['is_tw_present']:
                stop_value['wait_time'] = max(stop_value['start_time_window'] - stop_value['actual_start_time'], 0)
                stop_value['slack_time'] = max(stop_value['end_time_window'] - stop_value['actual_end_time'], 0)
                stop_value['max_slack'] = stop_value['end_time_window'] - (
                        stop_value['start_time_window'] + stop_value['planned_service_time'])

    def predict_probability(self):
        self.prob_high = 0.0

        vector_of_features = np.array([
                    self.average_speed,
                    self.executor_capacity,
                    self.filled_capacity,
                    self.filled_capacity_percent,
                    self.is_sequence_feasible,
                    self.is_tsp_feasible,
                    self.is_weekend,
                    self.max_segment_dist,
                    self.max_segment_time,
                    self.num_packages,
                    self.num_packages_not_delivered,
                    self.num_packages_tw,
                    self.num_stops,
                    self.num_stops_tw, 
                    self.num_tw_violations,
                    self.num_zone_switches,
                    self.num_zones,
                    self.packages_per_stop,
                    self.segment_dist_ratio,
                    self.segment_time_ratio,
                    self.slack_ratio,
                    self.switch_stop_ratio, 
                    self.switch_zone_ratio,
                    self.total_dist,
                    self.total_end_slack,
                    self.total_journey_time,
                    self.total_max_slack,
                    self.service_time,
                    self.travel_time,
                    self.total_wait_time,
                    self.tsp_optimality_gap,
                    self.tsp_route_time,
                    self.var_segment_dist,
                    self.var_segment_time,
                    self.vol_packages_not_delivered,
                    self.volume_seconds_traveled,
                    self.vst_ratio,
                    self.weighted_pkg_end_slack,
                    self.weighted_pkg_max_slack,
                    self.weighted_pkg_slack_ratio,
                    self.weighted_pkg_wait_time,
                ])
        
        vector_of_features = vector_of_features.reshape(1, -1)
        # get probability of being HIGH
        prob = loaded_model.predict_proba(vector_of_features)[0, 2]
        if PRINT_VERBOSITY >= 1:
            print('Probability of high', prob)
        self.prob_high = prob
        if prob >= self.best_prob_high:
            # when tied, lean in favor of updating
            self.best_prob_high = self.prob_high
            self.best_stop_order_dict = self.stop_order_dict

    # def score_given_stop_order(self):
    #     """ This code uses the score function of score.py to calculate
    #     route score; It should not be included in FINAL draft
    #     """
    #     # print(actual_routes_json[rt.key])
    #     actual = route2list(actual_routes_json[self.key])

    #     sub_dict = {}
    #     sub_dict['proposed'] = self.stop_order_dict
    #     sub = route2list(sub_dict)
    #     if isinvalid(actual, sub):
    #         rt.stop_order_score = invalid_scores_json[self.key]
    #         feas = False
    #     else:
    #         rt.stop_order_score = score(actual, sub, cost_matrices_json[rt.key])
    #         feas = True
    #     if rt.stop_order_score < rt.best_stop_order_score:
    #         rt.best_stop_order_score = rt.stop_order_score
    #         rt.best_stop_order_dict = rt.stop_order_dict

    # def write_stop_order_features_to_file(self):
    #     """
    #     This function writes non-static stop-order features in a file for easier analysis
    #     """
    #     if PRINT_CSV_FILES:
    #         file_path = path.join(BASE_DIR, 'data/model_apply_outputs/high_route_stop_order_summaries.csv')
    #         f = open(file_path, "a+")  # append mode, or create file if it doesn't exit
    #         f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
    #             self.key,
    #             self.index,
    #             self.unique_stop_order_string,
    #             self.prob_high,
    #             self.stop_order_score,
    #             self.num_tw_violations,
    #             self.is_sequence_feasible,
    #             self.num_packages_not_delivered,
    #             self.vol_packages_not_delivered,
    #             self.travel_time,
    #             self.total_journey_time,
    #             self.volume_seconds_traveled,
    #             self.vst_ratio,
    #             self.total_wait_time,
    #             self.total_end_slack,
    #             self.total_max_slack,
    #             self.slack_ratio,
    #             self.max_segment_dist,
    #             self.var_segment_dist,
    #             self.segment_dist_ratio,
    #             self.total_dist,
    #             self.average_speed,
    #             self.max_segment_time,
    #             self.max_segment_dist,
    #             self.segment_time_ratio))
    #         f.close()
        
    #     scores_list.append(self.best_stop_order_score)
    #     prob_high_list.append(self.prob_high)


def initialize_sequence(rt):
    # initialize stops in random order
    order = 0
    for key in rt.stop.keys():
        rt.stop_order_dict[key] = order
        if order == 0:
            other = key
        if str(rt.stop[key]['type']) == 'Station':
            station = key
            rt.origin_stop_key = key
        order += 1

    # swap the order of depot and the zero th stop
    rt.stop_order_dict[station], rt.stop_order_dict[other] = rt.stop_order_dict[other], rt.stop_order_dict[station]
    rt.stop_order_dict = dict(sorted(rt.stop_order_dict.items(), key=lambda item: item[1]))
    # print(json.dumps(rt.stop_order_dict, indent=4))

    # reverse keys and values
    rt.stop_sequence_dict = {value: key for key, value in rt.stop_order_dict.items()}
    # print(json.dumps(rt.stop_sequence_dict, indent=4))


def create_tsp_data(rt):
    """Stores the distance matrix for the stops on the route for Google OR tools"""
    data = {}
    data['time_matrix'] = []
    row = []
    for i in rt.stop_order_dict.keys():
        for j in rt.stop_order_dict.keys():
            row.append(int(math.floor(rt.travel_time_dict[i][j] * 10 + rt.stop[i]['planned_service_time'] * 10)))
        data['time_matrix'].append(row)
        row = []

    data['service_time'] = []
    data['time_windows'] = []
    for i in rt.stop_order_dict.keys():
        left_end_point = math.floor(rt.stop[i]['start_time_window'] * 10)
        right_end_point = math.floor(rt.stop[i]['end_time_window'] * 10 - rt.stop[i]['planned_service_time'] * 10)
        data['time_windows'].append((int(left_end_point), int(right_end_point)))
        data['service_time'].append(rt.stop[i]['planned_service_time'])

    data['num_vehicles'] = 1
    data['depot'] = 0

    initial_solution = [value for value in rt.stop_order_dict.values() if value > 0]
    data['initial_routes'] = [initial_solution]

    return data


def set_tsp_dict(rt, manager, routing, solution):
    count = 0
    index = routing.Start(0)
    rt.stop_order_dict[rt.origin_stop_key] = count
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        if manager.IndexToNode(index) != 0:  # the revisit to depot can be ignored in the dictionary
            count += 1
            rt.stop_order_dict[rt.stop_sequence_dict[index]] = count
    rt.stop_order_dict = dict(sorted(rt.stop_order_dict.items(), key=lambda item: item[1]))
    # print(json.dumps(rt.stop_order_dict, indent=4))


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


def compute_tsp_tour(rt, data):
    """OR tools function that computes the TSP. Settings can be changed for time limits and solution method"""
    try:
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
        rt.tsp_route_time = CONST_INFTY
        if rt.tsp_solver_status == 1:
            rt.tsp_route_time = solution.ObjectiveValue() / 10.0
            rt.total_journey_time = solution.ObjectiveValue() / 10.0
            # Save TSP tour in the form a dictionary for scoring
            set_tsp_dict(rt, manager, routing, solution)
            # print(rt.total_journey_time)
            rt.compute_stop_features()
            rt.compute_route_features()
            rt.is_tsp_feasible = rt.is_sequence_feasible

        # Print solution on console
        # if solution:
        #     print_tsp_solution(manager, routing, solution[1])
    except:
        print('Exception found while analysing TSP for route', rt.index)
        rt.tsp_route_time = CONST_INFTY
        rt.total_journey_time = CONST_INFTY


def create_reverse_routes(rt):
    """Function to reverse TSP routes"""
    if rt.tsp_solver_status == 1:
        for key, value in rt.stop_order_dict.items():
            if value != 0:
                rt.stop_order_dict[key] = rt.num_stops - value


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        # plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    # print('Maximum of the route distances: {}m'.format(max_route_distance))


def generate_tsp_pool(rt, data):
    """From https://stackoverflow.com/questions/57424868/how-to-collect-more-than-one-solution-google-or-tools-tsp"""
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = data['time_matrix']

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 3
    search_parameters.solution_limit = 15

    # without an initial assignment the CostVar is not available
    assignment = routing.SolveWithParameters(search_parameters)
    collector = initialize_collector(data, manager, routing)

    routing.SolveFromAssignmentWithParameters(assignment, search_parameters)
    rt.tsp_solver_status = routing.status()
    for i in range(collector.SolutionCount()):
        if PRINT_VERBOSITY >= 0:
            print(f'================ solution: {i} ================')
        if PRINT_VERBOSITY >= 1:
            print_solution(data, manager, routing, collector.Solution(i))
        set_tsp_dict(rt, manager, routing, collector.Solution(i))
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'TSP-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()

        if PRINT_VERBOSITY >= 1:
            print('Creating reverse rotue...')
        create_reverse_routes(rt)
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'PST-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()


def initialize_collector(data, manager, routing):
    collector: SolutionCollector = routing.solver().AllSolutionCollector()
    collector.AddObjective(routing.CostVar())
    routing.AddSearchMonitor(collector)

    for node in range(len(data['time_matrix'])):
        collector.Add(routing.NextVar(manager.NodeToIndex(node)))

    for v in range(data['num_vehicles']):
        collector.Add(routing.NextVar(routing.Start(v)))

    return collector


def variable_neighborhood_search(rt):
    tsp_data = create_tsp_data(rt)
    compute_tsp_tour(rt, tsp_data)

    generate_tsp_pool(rt, tsp_data)
    rt.compute_stop_features()
    rt.compute_route_features()



# ==================Begin--All functions for Zone-based TSP calculations=======#
def variable_neighborhood_search_zone_based(rt):
    """
    Pre variable-neighborhood search function that implements zone-based TSP
    and computes the probability of high for the given trained model

    """
    zone_tsp_data = create_inter_zone_tsp_data(rt)
    compute_tsp_tour_zoneBased(rt, zone_tsp_data)
    generate_zone_tsp_pool(rt, zone_tsp_data)

def create_inter_zone_tsp_data(rt):
    """Stores the average distance matrix from each zone to each other"""
    data = {}
    data['time_matrix'] = []
    row = []

    # Label each zone with a unique number. Depot = 0, and others 1,2,3,...
    for z in rt.zone_stop_dict:
        if z == 'Depot':
            rt.zone_id_to_key[0] = z
            break
    count = 1
    for z in rt.zone_stop_dict:
        if z != 'Depot':
            rt.zone_id_to_key[count] = z
            count += 1

    # compute average distance
    for i in range(len(rt.zone_id_to_key)):
        z1 = rt.zone_id_to_key[i]
        for j in range(len(rt.zone_id_to_key)):
            z2 = rt.zone_id_to_key[j]
            if z1 == z2:
                row.append(0)  # same zone
            else:
                sum_tt = 0
                count_stop_pairs = 0
                for s1 in rt.zone_stop_dict[z1]:
                    for s2 in rt.zone_stop_dict[z2]:
                        tt = int(math.floor(
                            rt.travel_time_dict[s1][s2] * 10
                            + rt.stop[s1]['planned_service_time'] * 10))
                        sum_tt += tt
                        count_stop_pairs += 1

                avg_tt = int(math.floor(sum_tt / count_stop_pairs))  # assuming floor operation
                row.append(avg_tt)
        data['time_matrix'].append(row)
        row = []

    data['num_vehicles'] = 1
    data['depot'] = 0
    if PRINT_VERBOSITY >= 1:
        print("Zone ids to key=", rt.zone_id_to_key)
    return data


def compute_tsp_tour_zoneBased(rt, data):
    """
    Computes the zone-based TSP tour and stores it in rt.zone_tsp_tour.
    """
    try:
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Setting first solution heuristic. Parameters are set to defaults.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)  # , solutions=3)
        # populate zone_tsp_tour if solution is found
        if routing.status() == 1:
            index = routing.Start(0)
            rt.zone_tsp_tour.append(manager.IndexToNode(index))
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                if manager.IndexToNode(index) != 0:  # the revisit to depot can be ignored in the dictionary
                    rt.zone_tsp_tour.append(manager.IndexToNode(index))
            plan_output = ""
            for z_id in rt.zone_tsp_tour:
                plan_output += ' {} ->'.format(rt.zone_id_to_key[z_id])
            # print('Zone-based tour is {}'.format(plan_output))
        else:
            print("TSP solution was not found")
    except:
        print('Exception found while analysing TSP for route', rt.index)


def run_within_zone_TSP(rt, tour_of_zones):
    """
    This function runs the TSP within zones given the tour of zones z0 -> z1 -> z2 -> ...
    z0 must be the depot
    Connect z0 (depot) to nearest node n1 in z1. Solve TSP over all stops in
    zone z1 with n1 as depot. Let TSP end at stop n_e1.
    Now connect n_e1 to nearest node node n2 in z2. Solve TSP in z2
    with n2 as depot. And the process continues until the last zone
    TSP is solved. In the end, stitch together the stops and store is as part
    of the stop_order_dictionary for probability evaluation

    Parameters
    ----------
    tour_of_zones : Ordered list of integer zone IDs (see zone_id_to_key dictionary)
    """
    if rt.zone_id_to_key[tour_of_zones[0]] != 'Depot' and len(tour_of_zones) != len(rt.zone_stop_dict):
        print("Given tour of zones does not start at depot zone/stop, or has fewer zones that the route. ERROR")
        print("Tour of zones=", tour_of_zones)

    # end stop of previous zone TSP route
    esopzt_key = rt.zone_stop_dict[rt.zone_id_to_key[0]][0]  # initialize with first stop key at depot zone

    rt.zone_id_to_ordered_stop_dict[0] = [esopzt_key]  # for depot
    for z_id in tour_of_zones:
        z_key = rt.zone_id_to_key[z_id]
        # print("Scanning zone",z_key," with stops", rt.zone_stop_dict[z_key])
        if z_id > 0:
            rt.zone_id_to_ordered_stop_dict[z_id] = []

            # ====1. find key of closest stop to esopzt====
            min_dist = 10000000
            closest_stop_key = "NULL"
            for s_key in rt.zone_stop_dict[z_key]:
                tt = int(math.floor(
                    rt.travel_time_dict[esopzt_key][s_key] * 10
                    + rt.stop[esopzt_key]['planned_service_time'] * 10))
                if tt < min_dist:
                    closest_stop_key = s_key
                    min_dist = tt

            # ====2. create TSP data for all stops within that zone====
            # ---2a. Uniquely number the stops such that closest_stop_key is 0 and others are 1,2,3,...
            stop_tsp_index_to_key = {}
            stop_tsp_index_to_key[0] = closest_stop_key
            count = 1
            for s in rt.zone_stop_dict[z_key]:
                if s != closest_stop_key:
                    stop_tsp_index_to_key[count] = s
                    count += 1
            # ---2b. Store the distance matrix for the stops on the route
            data = {}
            data['time_matrix'] = []
            row = []
            for i in range(len(stop_tsp_index_to_key)):
                s1 = stop_tsp_index_to_key[i]
                for j in range(len(stop_tsp_index_to_key)):
                    s2 = stop_tsp_index_to_key[j]
                    row.append(int(math.floor(
                        rt.travel_time_dict[s1][s2] * 10 + rt.stop[s1]['planned_service_time'] * 10)))
                data['time_matrix'].append(row)
                row = []
            data['num_vehicles'] = 1
            data['depot'] = 0

            # ====3. Solve TSP; if solved, extract TSP tour==========
            try:
                # print('Solving TSP now for zone ',z_key, 'with stops ', rt.zone_stop_dict[z_key])
                manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                                       data['num_vehicles'], data['depot'])
                routing = pywrapcp.RoutingModel(manager)

                def distance_callback(from_index, to_index):
                    """Returns the distance between the two nodes."""
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return data['time_matrix'][from_node][to_node]

                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                # Solve the problem
                solution = routing.SolveWithParameters(search_parameters)  # , solutions=3)
                # If solution is found, store order of stops in rt.zone_id_to_ordered_stop_dict
                if routing.status() == 1:
                    # pull together the stop IDs
                    index = routing.Start(0)
                    associated_stop_id = stop_tsp_index_to_key[manager.IndexToNode(index)]
                    rt.zone_id_to_ordered_stop_dict[z_id].append(associated_stop_id)
                    while not routing.IsEnd(index):
                        index = solution.Value(routing.NextVar(index))
                        if manager.IndexToNode(index) != 0:  # the revisit to depot can be ignored in the dictionary
                            associated_stop_id = stop_tsp_index_to_key[manager.IndexToNode(index)]
                            rt.zone_id_to_ordered_stop_dict[z_id].append(associated_stop_id)

                    # print("---TSP route is", rt.zone_id_to_ordered_stop_dict[z_id])
                else:
                    print("TSP solution was not found")
            except:
                print('Exception found while analysing TSP for route', rt.index, ' for zone', z_key)

            # reset esopzt
            esopzt_key = rt.zone_id_to_ordered_stop_dict[z_id][-1]

    # 3. Now stitch together the stops to create a stop_order_dictionary
    count = 0
    disp = []
    for z_id in tour_of_zones:
        for s_key in rt.zone_id_to_ordered_stop_dict[z_id]:
            rt.stop_order_dict[s_key] = count
            disp.append(s_key)
            count += 1
    # print("Stitched TSP route is = ",disp)
    # print("TSP rt.stop_order_dict=",rt.stop_order_dict)


def set_zone_tsp_tour(rt, manager, routing, solution):
    rt.zone_tsp_tour = []
    index = routing.Start(0)
    rt.zone_tsp_tour.append(manager.IndexToNode(index))
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        if manager.IndexToNode(index) != 0:  # the revisit to depot can be ignored in the dictionary
            rt.zone_tsp_tour.append(manager.IndexToNode(index))
    plan_output = ""
    for z_id in rt.zone_tsp_tour:
        plan_output += ' {} ->'.format(rt.zone_id_to_key[z_id])
    if PRINT_VERBOSITY >= 1:
        print('Zone-based tour is {}'.format(plan_output))


def generate_zone_tsp_pool(rt, data):
    """From https://stackoverflow.com/questions/57424868/how-to-collect-more-than-one-solution-google-or-tools-tsp"""
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = data['time_matrix']

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 3
    search_parameters.solution_limit = 15

    # without an initial assignment the CostVar is not available
    assignment = routing.SolveWithParameters(search_parameters)
    collector = initialize_collector(data, manager, routing)

    routing.SolveFromAssignmentWithParameters(assignment, search_parameters)
    rt.tsp_solver_status = routing.status()
    for i in range(collector.SolutionCount()):
        if PRINT_VERBOSITY >= 0:
            print(f'\n================ solution: {i} ================')
        if PRINT_VERBOSITY >= 1:
            print_solution(data, manager, routing, collector.Solution(i))
        set_zone_tsp_tour(rt, manager, routing, collector.Solution(i))

        if PRINT_VERBOSITY >= 1:
            print("\nZone-TSP route followed as TSP for within-zone TSP (forward)")
        run_within_zone_TSP(rt, rt.zone_tsp_tour)
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'Zone_TSP_forward-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()

        if PRINT_VERBOSITY >= 1:
            print('\nZone-TSP route followed as TSP for within-zone TSP (reverse)')
        create_reverse_routes(rt)
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'Zone_TSP_reverse-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()

        # zone-TSP route passed in reverse
        if PRINT_VERBOSITY >= 1:
            print("\nZone-TSP route followed as PST for within-zone TSP (forward)")
        reversed_tour = ((rt.zone_tsp_tour[::-1])[:-1])  # reverse the list and remove depot
        reversed_tour.insert(0, 0)
        run_within_zone_TSP(rt, reversed_tour)
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'Zone_PST_forward-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()

        if PRINT_VERBOSITY >= 1:
            print('\nZone-TSP route followed as PST for within-zone TSP (reverse)')
        create_reverse_routes(rt)
        rt.compute_stop_features()
        rt.unique_stop_order_string = 'Zone_PST_reverse-' + str(i)
        rt.compute_route_features()
        if PRINT_VERBOSITY >= 1:
            rt.display_route_data()
        rt.predict_probability()


# ==================End--All functions for Zone-based TSP calculations=========#

def read_route_data():
    """Reads the JSON files and populates class variables"""
    route_list = []
    temp_route = Route()

    prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
    with open(prediction_routes_path) as f:
        route_data = json.load(f)

    prediction_package_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json')
    with open(prediction_package_path) as f:
        package_data = json.load(f)

    prediction_traveltime_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
    with open(prediction_traveltime_path) as f:
        travel_time = json.load(f)

    count = 0
    for key in route_data.keys():
        temp_route.index = count  # replace id with index
        temp_route.key = key
        temp_route.station_code = route_data[key]['station_code']
        temp_route.date = datetime.strptime(route_data[key]['date_YYYY_MM_DD'], '%Y-%m-%d').date()
        temp_route.departure_time = datetime.strptime(route_data[key]['departure_time_utc'], '%H:%M:%S').time()
        temp_route.executor_capacity = route_data[key]['executor_capacity_cm3']

        for stop_key in route_data[key]['stops']:
            route_data[key]['stops'][stop_key]['packages'] = package_data[key][stop_key]

        temp_route.stop = route_data[key]['stops']
        temp_route.travel_time_dict = travel_time[key]

        route_list.append(temp_route)
        count += 1
        temp_route = Route()
    if PRINT_VERBOSITY >= 0:
        print('Finished reading %d routes...' % count)
    return route_list


def core_block(rt):
    """Code that computes the best route sequence for each route"""
    initialize_sequence(rt)
    rt.compute_static_stop_features()
    rt.compute_static_route_features()
    # variable_neighborhood_search(rt)
    variable_neighborhood_search_zone_based(rt)
    rt.display_route_data()

    return rt


if __name__ == '__main__':
    begin = time.time()
    route = read_route_data()

    begin = time.time()
    print('Beginning serial block...')
    results = [core_block(rt) for rt in route]
        
    print('Serial block complete...')
    print('Time for this block is %.2f minutes' % ((time.time() - begin) / 60))

    temp_dict = {}
    proposed_route_dict = {}
    for rt in results:
        temp_dict['proposed'] = rt.best_stop_order_dict
        proposed_route_dict[rt.key] = temp_dict
        temp_dict = {}

    with open(path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json'), "w") as outfile:
        json.dump(proposed_route_dict, outfile)

