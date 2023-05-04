#=========================================
# Name_class: Camera
# Purpose   : This class is used to adjust
# features of the camera from option user at
# "listcam.py"
#=========================================

#-----------------------------------------   
# Warning!!!: Don't modify this class if 
# don't understand all the system because 
# it can make the system work wrong
#-----------------------------------------
# from utility.utils_logger import logger
# from ai_core.utility.ai_logger import aiLogger
from ai_functions.utility.ai_logger import aiLogger

class Camera:
    def __init__(self, raw_camera):
        # print(raw_camera)
        #----------------------------------
        # Id of camera and general Ai flag
        #---------------------------------- 
        self.id                 = raw_camera['stream_id'] 
        self.run_ai             = raw_camera['run_ai']
        self.name               = raw_camera['name']
        # self.run_ai = False

        #----------------------------------
        # In out counting
        #---------------------------------- 
        if 'in_out_lines' not in raw_camera or raw_camera['in_out_lines'] is None:
            raw_camera['in_out_lines'] = [[(0,0.6),(1,0.6)]]
        self.in_out_lines = raw_camera['in_out_lines']
        
        #----------------------------------
        # Zone
        #---------------------------------- 
        if 'zones' not in raw_camera or raw_camera['zones'] is None:
            raw_camera['zones'] = [[(0,0.7),(0.3,0),(1,0.7),(0,1),(0.3,1)]]
        self.zones = raw_camera['zones']


        # self.zones       = [[(20,15),(4,70),(98,73),(98,14)], [(4,71),(0,90),(98,90),(98,75)]]

        #----------------------------------
        # Features of Ai
        #---------------------------------- 
        self.run                = {}
        self.run['statistic']   = {}
        self.config             = {}
        
        
        
        if not self.run_ai:
            self.run['yolo5']               = False
            self.run['retinaface']          = False
            self.run['deeplabv3p']          = False
            self.run['fire_classification'] = False
            self.run['object_counter']      = False
            self.run['crowd_detection']     =  False
            self.run['license_plate_recognition']   =  False
            self.run['statistic']['heat_map']             =  False
            self.run['statistic']['heat_map_direction']   =  False 
            self.run['statistic']['object_counting']      =  False 
            self.run['statistic']['velocity']             =  False
            self.run['without_ai']          = True
        else:
            #--------------------------
            # Yolo
            self.run['yolo5']                   = raw_camera['feature']['statistic']['heat_map']['enabled']             or \
                                                  raw_camera['feature']['statistic']['heat_map_direction']['enabled']   or \
                                                  raw_camera['feature']['statistic']['object_counting']['enabled']      or \
                                                  raw_camera['feature']['statistic']['velocity']['enabled']             or \
                                                  raw_camera['feature']['human']['human_counting']['enabled']           or \
                                                  raw_camera['feature']['human']['crowd_detection']['enabled']          or \
                                                  raw_camera['feature']['human']['virtual_fences']['enabled']           or \
                                                  raw_camera['feature']['vehicle']['vehicle_counting']['enabled']       or \
                                                  raw_camera['feature']['vehicle']['license_plate_recognition']['enabled'] or \
                                                  raw_camera['feature']['vehicle']['parking_violation_detection']['enabled'] or \
                                                  raw_camera['feature']['vehicle']['way_driving_violation_detection']['enabled'] or \
                                                  ('vehicle_speed_violation' in raw_camera['feature']['vehicle'] and raw_camera['feature']['vehicle']['vehicle_speed_violation']['enabled']) or \
                                                  ('traffic_jam_detection' in raw_camera['feature']['vehicle'] and raw_camera['feature']['vehicle']['traffic_jam_detection']['enabled']) or \
                                                  ('human_multiple_camera_tracking' in raw_camera['feature']['human'] and raw_camera['feature']['human']['human_multiple_camera_tracking']['enabled'])


            self.run['yolo_app_center']         = self.run['yolo5']

            self.config['yolo5'] = {}
            min_conf = 0.5
            for f in raw_camera['feature']['human']:
                if (raw_camera['feature']['human'][f]['enabled'] and 'confidence_score' in raw_camera['feature']['human'][f]['configuration']):
                    if min_conf > raw_camera['feature']['human'][f]['configuration']['confidence_score']:
                        min_conf = raw_camera['feature']['human'][f]['configuration']['confidence_score']
            self.config['yolo5']['confidence_score'] = min_conf
            #--------------------------
            # Face
            self.run['retinaface']              = raw_camera['feature']['face']['enabled']
            self.config['retinaface']           = raw_camera['feature']['face']['configuration']
            
            if ('feature_type'  not in self.config['retinaface']):
                self.config['retinaface']['feature_type'] = 'access_control'
            if ('zones'  not in self.config['retinaface'] \
                or  ('zones' in self.config['retinaface'] and len(self.config['retinaface']['zones']) == 0) ):
                # self.config['retinaface']['zones'] =   {
                #                     "id": "unknown",
                #                     "name": "unknown",
                #                     "points": [(0,0),(1,0),(1,1),(0,1)],
                #                 }
                self.config['retinaface']['total_zones'] = [[(0,0),(0,1),(1,1),(1,0)]]
            else:
                self.config['retinaface']['total_zones'] = []
                for z in self.config['retinaface']['zones']:
                    self.config['retinaface']['total_zones'].append(z['points'])

            #--------------------------
            # Deeplab
            self.run['deeplabv3p']              = False

            #--------------------------
            # Fire
            self.run['fire_classification']     = raw_camera['feature']['fire']['enabled']
            self.config['fire_classification']  = raw_camera['feature']['fire']['configuration']
            #-----------------------------------

            if ('human_multiple_camera_tracking' in raw_camera['feature']['human']):
                self.run['human_multiple_camera_tracking']          = raw_camera['feature']['human']['human_multiple_camera_tracking']['enabled']
                self.config['human_multiple_camera_tracking']       = raw_camera['feature']['human']['human_multiple_camera_tracking']['configuration']
            else:
                self.run['human_multiple_camera_tracking'] = False
                self.config['human_multiple_camera_tracking'] = []
            #-------------------------------------
            # counter
            self.config['counting_zones'] = []
            aiLogger.info(raw_camera['feature']['vehicle'])
            aiLogger.info(raw_camera['feature']['vehicle']['vehicle_counting']['enabled'])
            if raw_camera['feature']['vehicle']['vehicle_counting']['enabled']:
                zones = raw_camera['feature']['vehicle']['vehicle_counting']['configuration']['zones']
                aiLogger.info(raw_camera['feature']['vehicle']['vehicle_counting']['configuration']['zones'])
                for zone in zones:
                    zone.update({"type": "vehicle"})
                    self.config['counting_zones'].append(zone)
            if raw_camera['feature']['human']['human_counting']['enabled']:
                zones = raw_camera['feature']['human']['human_counting']['configuration']['zones']
                for zone in zones:
                    zone.update({"type": "person"})
                    self.config['counting_zones'].append(zone)
            if ('human_multiple_camera_tracking' not in raw_camera['feature']['human']):
                raw_camera['feature']['human']['human_multiple_camera_tracking'] = {}
                raw_camera['feature']['human']['human_multiple_camera_tracking']['enabled'] = False
            if raw_camera['feature']['human']['human_multiple_camera_tracking']['enabled']:
                zones = raw_camera['feature']['human']['human_multiple_camera_tracking']['configuration']['zones']
                for zone in zones:
                    zone.update({"type": "tracking"})
                    self.config['counting_zones'].append(zone)
            print("self.config['counting_zones']-------- ",self.config['counting_zones'])

            #--------------------------
            # Others
            # velocity
            if ('vehicle_speed_violation' in raw_camera['feature']['vehicle']):
                self.run['vehicle_speed_violation'] = raw_camera['feature']['vehicle']['vehicle_speed_violation']['enabled']
                self.config['vehicle_speed_violation'] = raw_camera['feature']['vehicle']['vehicle_speed_violation']['configuration']
            else:
                self.run['vehicle_speed_violation'] = False
                self.config['vehicle_speed_violation'] = []

            # print('wwwwwwwwwwww', 'traffic_jam_detection' in raw_camera['feature']['vehicle'],raw_camera['feature']['vehicle']['traffic_jam_detection']['enabled'], raw_camera['feature']['vehicle'])


            if ('traffic_jam_detection' in raw_camera['feature']['vehicle']):
                self.run['traffic_jam_detection']    = raw_camera['feature']['vehicle']['traffic_jam_detection']['enabled']
                self.config['traffic_jam_detection'] = raw_camera['feature']['vehicle']['traffic_jam_detection']['configuration']
            else:
                self.run['traffic_jam_detection'] = False
                self.config['traffic_jam_detection'] = []
            # self.config['human']                = raw_camera['feature']['human']['configuration']
            
            #--------------------------
            # Crowd detection
            self.run['crowd_detection']         = raw_camera['feature']['human']['crowd_detection']['enabled']
            self.config['crowd_detection']      = raw_camera['feature']['human']['crowd_detection']['configuration']

            self.run['virtual_fences']          = raw_camera['feature']['human']['virtual_fences']['enabled']
            self.config['virtual_fences']       = raw_camera['feature']['human']['virtual_fences']['configuration']

            self.run['parking_violation_detection']    = raw_camera['feature']['vehicle']['parking_violation_detection']['enabled']
            self.config['parking_violation_detection'] = raw_camera['feature']['vehicle']['parking_violation_detection']['configuration']

            self.run['way_driving_violation_detection']    = raw_camera['feature']['vehicle']['way_driving_violation_detection']['enabled']
            self.config['way_driving_violation_detection'] = raw_camera['feature']['vehicle']['way_driving_violation_detection']['configuration']


            self.run['license_plate_recognition']       =  raw_camera['feature']['vehicle']['license_plate_recognition']['enabled']
            self.config['license_plate_recognition']    =  raw_camera['feature']['vehicle']['license_plate_recognition']['configuration']

            #--------------------------
            # Statistic
            self.run['heat_map']                          = raw_camera['feature']['statistic']['heat_map']['enabled'] 
            self.run['velocity']                          = raw_camera['feature']['statistic']['velocity']['enabled'] 


            self.run['statistic']['heat_map']             = raw_camera['feature']['statistic']['heat_map']['enabled'] 
            self.run['statistic']['heat_map_direction']   = raw_camera['feature']['statistic']['heat_map_direction']['enabled'] 
            self.run['statistic']['object_counting']      = raw_camera['feature']['statistic']['object_counting']['enabled'] 
            self.run['statistic']['velocity']             = raw_camera['feature']['statistic']['velocity']['enabled'] 
            self.run['statistic']['time']                 = raw_camera['feature']['statistic']["config"]["time"]

            self.config['statistic'] = {}
            self.config['statistic']['heat_map']          = raw_camera['feature']['statistic']['heat_map']['configuration']
            #--------------------------
            #object counter                                        
            self.run['object_counter']          =   raw_camera['feature']['human']['human_counting']['enabled'] or \
                                                    raw_camera['feature']['vehicle']['vehicle_counting']['enabled'] or \
                                                    raw_camera['feature']['human']['human_multiple_camera_tracking']['enabled']
            
            self.run['human_counting']          =   raw_camera['feature']['human']['human_counting']['enabled']    
            self.config['human_counting']       =   raw_camera['feature']['human']['human_counting']['configuration']  

            self.run['vehicle_counting']        =   raw_camera['feature']['vehicle']['vehicle_counting']['enabled'] 
            self.config['vehicle_counting']     =   raw_camera['feature']['vehicle']['vehicle_counting']['configuration']

            
            # ========================
            # TEST
            self.run['object_counter']                  = False
            self.run['heat_map']                        = False
            self.run['crowd_detection']                 = False
            self.run['virtual_fences']                  = True
            self.run['traffic_jam_detection']           = False
            self.run['parking_violation_detection']     = False
            self.run['red_traffic_light']               = False
            self.run['way_driving_violation_detection'] = False
            self.run['vehicle_speed_violation']         = False
            self.run['human_multiple_camera_tracking']  = False

            # self.in_out_lines = []
            # if (raw_camera['feature']['human']['human_counting']['enabled']):
            #     for c in raw_camera['feature']['human']['human_counting']['configuration']['zones']:
            #         self.in_out_lines.append(c['cross_line'])

            # if (raw_camera['feature']['vehicle']['vehicle_counting']['enabled']):
            #     for c in raw_camera['feature']['vehicle']['vehicle_counting']['configuration']['zones']:
            #         self.in_out_lines.append(c['cross_line'])

            #------------------------------

            # Without ai
            self.run['without_ai']              =   (not (self.run['yolo5'] or self.run['retinaface']  or self.run['deeplabv3p']  or self.run['fire_classification'])) 
            
        #----------------------------------
        # Determine the priority feature
        #---------------------------------- 
        if self.run['without_ai']:
            self.priority_feature   = 'without_ai'        
        else:
            if (self.run['retinaface'] and not self.run['yolo5']):
                self.priority_feature   = 'retinaface'  
            elif self.run['fire_classification'] and not self.run['yolo5'] and not self.run['retinaface']:
                self.priority_feature   =  'fire_classification'
            else:
                self.priority_feature   = 'yolo5'

        print("\n")
        aiLogger.warning("The priority of the system of the camera{} is: {}".format(self.id ,self.priority_feature)) 
        #----------------------------------
        # Applications inside Yolo5 
        #---------------------------------- 
        self.run_track_in_yolo5     = True
        
        
        # These options only work if tracking algorithm was turned on, vice versa
        # self.run_heatmap_in_statistic_yolo5   = raw_camera['feature']['yolo5_application']['heatmap']    and self.run_track_in_yolo5 
        # self.run_direction_in_statistic_yolo5 = raw_camera['feature']['yolo5_application']['direction']  and self.run_track_in_yolo5 
        # self.run_count_in_statistic_yolo5     = raw_camera['feature']['yolo5_application']['count']      and self.run_track_in_yolo5 
        # self.run_speed_in_stattistic_yolo5    = raw_camera['feature']['yolo5_application']['speed']      and self.run_track_in_yolo5 
       
    
