#### Carla related imports
#from carla.client import make_carla_client, VehicleControl
import carla

from sensors import SensorManager, DisplayManager

### Other scripts
from utils import *
from parameters import *

## Standard imports
import time
from math import cos, sin
import logging
import random
import time
import os
from pprint import pprint
import numpy as np
from numpy.linalg import pinv, inv
import cv2



########### Notes ################
# Lidar image is for visualization, better to use points as model input.
# The parked vehicle bounding boxes are not detected by carla (refer: https://github.com/carla-simulator/carla/issues/2343)
#

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

folders = ['image_rgb', 'labels_bbox', 'velodyne', 'planes', 'radar', 'image_lidar', 'image_radar','image_depth']
### Create the required directory. 
for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


### Create paths to save the data ##########
#GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_DATA_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LIDAR_IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_lidar/{0:06}.png')

LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'labels_bbox/{0:06}.txt')

IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_rgb/{0:06}.png')

RADAR_DATA_PATH = os.path.join(OUTPUT_FOLDER, 'radar/{0:06}.bin')
RADAR_IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_radar/{0:06}.png')

DEPTH_IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_depth/{0:06}.png')

def get_carla_settings(client):
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(8000)
    settings = world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle_list=[]
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle.*')
    ego_bp = world.get_blueprint_library().filter('*audi*')[0]
    spawn_points = world.get_map().get_spawn_points()
    for i in range(0,NUM_VEHICLES):
        point=random.choice(spawn_points)
        vehicle =world.try_spawn_actor(random.choice(vehicle_blueprints), point)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            vehicle_list.append(vehicle)

    ego_vehicle=None
    while ego_vehicle is None:
        ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
    ego_vehicle.set_autopilot(True)
    #vehicle_list.append(ego_vehicle)

    display_manager = DisplayManager(grid_size=[2, 2], window_size=[WINDOW_WIDTH, WINDOW_HEIGHT])
    x_loc=1
    y_loc=0
    depth_sensor=SensorManager(world, display_manager, 'DepthCamera', carla.Transform(carla.Location(x=x_loc,y=y_loc, z=CAMERA_HEIGHT_POS), carla.Rotation(0,0,0)), 
                      ego_vehicle, {}, display_pos=[1, 1])

    rgb_camera=SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=x_loc,y=y_loc, z=CAMERA_HEIGHT_POS), carla.Rotation(0,0,0)), 
                    ego_vehicle, {}, display_pos=[0, 0], vehicles=vehicle_list, depth_sensor=depth_sensor)
    lidar=SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=x_loc,y=y_loc, z=CAMERA_HEIGHT_POS), carla.Rotation(0,0,0)), 
                      ego_vehicle, {'channels' : '40', 'range' : '70',  'points_per_second': '920000', 'rotation_frequency': '20',
                       'upper_fov':'7', 'lower_fov':'-16' }, display_pos=[1, 0])
    radar=SensorManager(world, display_manager, 'Radar', carla.Transform(carla.Location(x=x_loc,y=y_loc, z=CAMERA_HEIGHT_POS), carla.Rotation(0,0,0)), 
                      ego_vehicle, {'range' : '70' }, display_pos=[0,1])
    
    sensors={'depth_sensor':depth_sensor, 'rgb_camera':rgb_camera, 'lidar':lidar, 'radar':radar}

    return world, display_manager, ego_vehicle, original_settings, vehicle_list, sensors


weather_presets=['Default', 'ClearNoon','SoftRainSunset' ] #'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 'CloudyNoon', 'CloudySunset', 
#'Default', 'HardRainNight', 'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 
# 'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 'WetCloudyNight', 'WetCloudyNoon', 
# 'WetCloudySunset', 'WetNight', 'WetNoon', 'WetSunset'];

class CarlaSimulator(object):
    def __init__(self, carla_client):
        self.client = carla_client
        self.captured_frame_no = self.current_captured_frame_num()
        time.sleep(1)
        self.world=None
        self._timer = Timer()
        # To keep track of how far the car has driven since the last capture of data
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0
        # How many frames we have captured since reset
        self._captured_frames_since_restart = 0

    def current_captured_frame_num(self):
        label_path = os.path.join(OUTPUT_FOLDER, 'labels_bbox/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = input("There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(OUTPUT_FOLDER))
        if answer.upper() == "O":
            print("Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        print("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files
    
    def init_sim(self):
        self._captured_frames_since_restart=0
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0

        #####reset stuff ########
        if self.world:
            if self.display_manager:
                self.display_manager.destroy()

            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

            self.world.apply_settings(self.original_settings)

        maps=self.client.get_available_maps()
        cur_map=random.choice(maps)
        print(f'Using map: {cur_map}')
        self.client.load_world(cur_map)
        weather=random.choice(weather_presets)
        print(f'Using waether: {weather}')
        self.world, self.display_manager, self.ego_vehicle, self.original_settings, self.vehicle_list, self.sensors = get_carla_settings(self.client)
        self.world.set_weather(getattr(carla.WeatherParameters,weather, weather))

    def run(self):
        """Launch the PyGame."""
        #Simulation loop
        call_exit = False
        self.init_sim()
        try:
            while True:
                time.sleep(0.01)
                self.world.tick()
                #world.wait_for_tick()
                
                # Render received data
                self.display_manager.render()
                
                reset = self._on_loop()
                if reset:
                    self.init_sim()
                self.save_data()
                if self.captured_frame_no>10000:
                    print('stored more than 10000 images')
                    break

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        call_exit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            call_exit = True
                            break

                if call_exit:
                    break
        finally:
            if self.display_manager:
                self.display_manager.destroy()

            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

            self.world.apply_settings(self.original_settings)

    def read_data(self):
        data={}
        data['image_rgb']=self.sensors['rgb_camera'].data
        data['image_lidar']=self.sensors['lidar'].lidar_img
        data['points_lidar']=self.sensors['lidar'].lidar_data

        data['image_radar']=self.sensors['radar'].radar_img
        data['points_radar']=self.sensors['radar'].radar_data
        data['depth']=self.sensors['depth_sensor'].data
        data['bbox']=self.sensors['rgb_camera'].all_BB
        data['classes']=self.sensors['rgb_camera'].all_classes
        return data

    def _on_loop(self):
        self._timer.tick()
        self.sensor_data = self.read_data()
        
        is_stuck = self._frames_since_last_capture >= NUM_EMPTY_FRAMES_BEFORE_RESET
        is_enough_datapoints = (
            self._captured_frames_since_restart + 1) % NUM_RECORDINGS_BEFORE_RESET == 0

        if (is_stuck or is_enough_datapoints):
            print('Stuck or Alreay generate enough data points')
            self.init_sim()
            return True
        
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            self._timer.lap()
        


    def save_data(self):
        if self.sensor_data['image_rgb'] is not None and self.sensor_data['depth'] is not None:
            distance_driven = self._distance_since_last_recording()
            #print("Distance driven since last recording: {}".format(distance_driven))
            has_driven_long_enough = distance_driven is None or distance_driven > DISTANCE_SINCE_LAST_RECORDING
            if (self._timer.step + 1) % STEPS_BETWEEN_RECORDINGS == 0:
                #image = self.sensor_data['rgb_camera'] #image_converter.to_rgb_array(self.rgb_data)
                # Retrieve and draw datapoints
                #image, datapoints = self._generate_datapoints(self.sensor_data['rgb_camera'])
                if has_driven_long_enough and len(self.sensor_data['classes'])>0:
                    self._agent_location_on_last_capture = vector3d_to_array(self.ego_vehicle.get_transform().location) ## save the last location of agent
                    # Save screen, lidar and kitti training labels together with calibration and groundplane files
                    self._save_training_files()
                    self.captured_frame_no += 1
                    self._captured_frames_since_restart += 1
                    self._frames_since_last_capture = 0
                else:
                    logging.debug("Could save datapoint, but agent has not driven {} meters since last recording (Currently {} meters)".format(
                        DISTANCE_SINCE_LAST_RECORDING, distance_driven))
            else:
                self._frames_since_last_capture += 1
                logging.debug(
                    "Could not save training data - no visible agents of selected classes in scene")

    def _distance_since_last_recording(self):
        if self._agent_location_on_last_capture is None:
            return None
        cur_pos = vector3d_to_array(
            self.ego_vehicle.get_transform().location)
        last_pos = self._agent_location_on_last_capture
        
        return dist_func(cur_pos, last_pos)


    def _save_training_files(self):
        logging.info("Attempting to save at timer step {}, frame no: {}".format(
            self._timer.step, self.captured_frame_no))
        #groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        lidar_data_fname = LIDAR_DATA_PATH.format(self.captured_frame_no)
        lidar_image_fname = LIDAR_IMAGE_PATH.format(self.captured_frame_no)
        
        label_fnnae = LABEL_PATH.format(self.captured_frame_no)
        radar_data_fname = RADAR_DATA_PATH.format(self.captured_frame_no)
        radar_image_fname = RADAR_IMAGE_PATH.format(self.captured_frame_no)
        depth_image_fname = DEPTH_IMAGE_PATH.format(self.captured_frame_no);
        
        ## save images
        color_fmt = cv2.cvtColor(self.sensor_data['image_rgb'].swapaxes(0,1), cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_fname, color_fmt)
        cv2.imwrite(lidar_image_fname, self.sensor_data['image_lidar'])
        cv2.imwrite(radar_image_fname, self.sensor_data['image_radar'].swapaxes(0,1))
        cv2.imwrite(depth_image_fname, self.sensor_data['depth']*20)

        ## save binary files
        self.save_bin_files(lidar_data_fname, self.sensor_data['points_lidar'])
        self.save_bin_files(radar_data_fname, self.sensor_data['points_radar'] )

        ## 
        string=''
        for bb,name in zip(self.sensor_data['bbox'],self.sensor_data['classes']):
            string=string+name+','+''.join([str(b) for b in bb])+'\n'

        with open(label_fnnae, 'w') as f:
             f.write(string)
             


    def save_bin_files(self,name,data):
        data = np.array(data).astype(np.float32)
        data.tofile(name)

        #


def need_this_class(agent):
    return True in [agent.HasField(class_type.lower()) for class_type in CLASSES_TO_DETECT]

if __name__ == '__main__':

    while True:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)
            sim=CarlaSimulator(client)
            sim.run()

        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            break;

