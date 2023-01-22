from utils import CustomTimer
import numpy as np
import carla
import pygame
from parameters import *
import weakref
import cv2
import math


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

#######  Modified the Carla example code #################
class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, vehicles=None, depth_sensor=None):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.vehicles=vehicles
        self.time_processing = 0.0
        self.tics_processing = 0
        self.depth_sensor=depth_sensor
        self.display_man.add_sensor(self)
        self.data=None
        self.all_BB=None
        self.all_classes=None
        self.radar_img=None
        self.radar_data=None

    def init_sensor(self, sensor_type, transform, ego_vehicle, sensor_options):
        self.ego_vehicle=ego_vehicle
        if sensor_type == 'RGBCamera':
            fov=105
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))
            camera_bp.set_attribute('fov', str(fov))

            self.width=disp_size[0]
            self.height=disp_size[1]

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)

            camera.listen(self.save_rgb_image)
            self.K = build_projection_matrix(disp_size[0], disp_size[1], fov)
            self.world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            return camera
        
        if sensor_type == 'DepthCamera':
            fov=105
            camera_dp = self.world.get_blueprint_library().find('sensor.camera.depth')
            disp_size = self.display_man.get_display_size()
            camera_dp.set_attribute('image_size_x', str(disp_size[0]))
            camera_dp.set_attribute('image_size_y', str(disp_size[1]))
            camera_dp.set_attribute('fov', str(fov))

            for key in sensor_options:
                camera_dp.set_attribute(key, sensor_options[key])

            depth_camera = self.world.spawn_actor(camera_dp, transform, attach_to=ego_vehicle)
            depth_camera.listen(self.save_depth_image)
            return depth_camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=ego_vehicle)

            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=ego_vehicle)

            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        elif sensor_type == "Radar":
            self.velocity_range = 7.5

            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')

            disp_size = self.display_man.get_display_size()
            self.width=disp_size[0]
            self.height=disp_size[1]
            self.K = build_projection_matrix(disp_size[0], disp_size[1], 105)

            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=ego_vehicle)
            radar.listen(self.save_radar_image)

            return radar
        
        else:
            return None

    def Get_bb(self,image,depth):
        vehicle=self.ego_vehicle
        self.all_BB=[]
        self.all_classes=[]
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        for npc in self.world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        bb_point_image = get_image_point(bb.location, self.K, self.world_2_camera)
                       
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        bb_cur=np.stack([get_image_point(v, self.K, self.world_2_camera) for v in verts])
                        bb_point_image=bb_cur.mean(axis=0)
                        #print(bb_point_image)

                        #print(verts[0])
                        p1_all=[]
                        p2_all=[]
                        for edge in edges:
                            p1 = bb_cur[edge[0]] #get_image_point(verts[edge[0]], self.K, self.world_2_camera)
                            p2 = bb_cur[edge[1]] #get_image_point(verts[edge[1]],  self.K, self.world_2_camera)
                            if not (p1[0]>0 and p1[0]<self.width and p2[0]>0 and p2[0]<self.width):
                                continue   

                            if not (p1[1]>0 and p1[1]<self.height and p2[1]>0 and p2[1]<self.height):
                                continue
                            p1_all.append(p1)
                            p2_all.append(p2)
                        # if not (bb_point_image[0]>0 and bb_point_image[0]<self.height and bb_point_image[1]>0 and bb_point_image[1]<self.width):
                        #         continue  
                        
                        for p1,p2 in zip(p1_all,p2_all):
                            cv2.line(image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
                        self.all_BB.append(bb_cur)
                        self.all_classes.append('Car')
        return image    

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
       

        #bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.sensor)
        #bounding_boxes=self.Filter_bb(bounding_boxes)
        #array=self.draw_bb(array,bounding_boxes)
        self.world_2_camera = np.array(self.sensor.get_transform().get_inverse_matrix())
        depth=self.depth_sensor.data
        #
        
        
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        

        if self.display_man.render_enabled():
            array_bb=self.Get_bb(array.astype('uint8').copy(),depth)
            self.surface = pygame.surfarray.make_surface(array_bb.swapaxes(0, 1))
        array=array.swapaxes(0, 1)
        self.data=array

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_depth_image(self,image):
        t_start = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        depth= (array[:,:,0]+array[:,:,1]*256+array[:,:,2]*256*256)/(256*256*256-1)
        depth=depth*1000.0
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(depth.swapaxes(0, 1))
        self.data=depth
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1


    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))


        lidar_data = np.array(points[:, :2])
        lidar_data[:,0]=-lidar_data[:,0]

        lidar_data = lidar_data[lidar_data[:,0]<0,:]
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        #lidar_img=lidar_img.swapaxes(0, 1)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img.swapaxes(0, 1))

        self.lidar_img=lidar_img
        self.lidar_data=lidar_data

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
    

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))


        radar_data = np.array(points[:, :2])
        #radar_data=radar_data.swapaxes(0,1)
        #radar_data[:,0]=-radar_data[:,0]
        radar_data = radar_data[radar_data[:,1]<0,:]

        radar_data *= min(disp_size) / lidar_range
        radar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        radar_data = np.fabs(radar_data)  # pylint: disable=E1111
        radar_data = radar_data.astype(np.int32)
        radar_data = np.reshape(radar_data, (-1, 2))
        radar_data=radar_data[:,::-1]
        radar_img_size = (disp_size[0], disp_size[1], 3)
        radar_img = np.zeros((radar_img_size), dtype=np.uint8)
        radar_img[tuple(radar_data.T)] = (255, 255, 255)
        #print(radar_img.shape)
        radar_img=radar_img
        if self.display_man.render_enabled():
           self.surface = pygame.surfarray.make_surface(radar_img)

        self.radar_img=radar_img
        self.radar_data=radar_data

        t_end = self.timer.time()
        #self.data=points
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


### Display manager ##
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None
