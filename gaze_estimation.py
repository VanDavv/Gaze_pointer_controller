'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
import cv2
import sys
import logging as log

class Gaze_estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.initial_w = None
        self.initial_h = None

    def load_model(self, ie):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
   
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_structure, self.model_weights))
        try:
            self.net4 = ie.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        log.info("Loading IR to the plugin...")
        self.exec_net = ie.load_network(network=self.net4, num_requests=0, device_name="MYRIAD")
        
        self.input_name=next(iter(self.exec_net.inputs))
        self.input_shape=self.exec_net.inputs['left_eye_image'].shape
        self.output_name=next(iter(self.exec_net.outputs))
        self.output_shape=self.exec_net.outputs[self.output_name].shape

    def predict(self, r_eye, l_eye, pose):
        '''
        Perform inference.
        '''
        log.info("Performing ge inference...")
        input_dict={}
        input_dict['left_eye_image'] = self.preprocess_input(r_eye)
        input_dict['right_eye_image'] = self.preprocess_input(l_eye)
        input_dict['head_pose_angles'] = pose
        
        outputs=self.exec_net.start_async(request_id=0, inputs=input_dict)
        while True:
            status=self.exec_net.requests[0].wait(-1)
            if status==0:
                break
            else: time.sleep(1)
        coords=self.preprocess_output(outputs)
        self.draw_outputs(coords, r_eye, l_eye)
        return coords

    def preprocess_input(self, image):
        '''
		Preprocess input images and return dictionnary of modified images.
		'''
        # log.info("Preprocessing the input images...")
        try:
            #print(self.input_shape)
            n, c, h, w = self.input_shape
            in_frame = cv2.resize(image, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            return in_frame
        except Exception as e:
             print(str(e))

    def preprocess_output(self, outputs):
        '''
		Preprocess the output and return coodinates of BBox(s).
		'''
        x = self.exec_net.requests[0].outputs['gaze_vector'][0][0]
        y = self.exec_net.requests[0].outputs['gaze_vector'][0][1]
        z = self.exec_net.requests[0].outputs['gaze_vector'][0][2]
        
        return (x, y, z)

    def draw_outputs(self, coords, r_eye, l_eye):
        '''
        Draw Bounding Boxs and texts on images.
        '''
        origin_x_re = r_eye.shape[1]//2
        origin_y_re = r_eye.shape[0]//2
        origin_x_le = l_eye.shape[1]//2 
        origin_y_le = l_eye.shape[0]//2       

        x, y = int(coords[0]*100), int(coords[1]*100)
        cv2.arrowedLine(l_eye, (origin_x_le, origin_y_le), (origin_x_le+x, origin_y_le-y), (255,0,255), 3)
        cv2.arrowedLine(r_eye, (origin_x_re, origin_y_re), (origin_x_re+x, origin_y_re-y), (255,0,255), 3)        
       

    def set_initial(self, w, h):
        self.initial_w = w
        self.initial_h = h 