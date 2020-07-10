'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import sys
import logging as log
import cv2
import time

class Landmark_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

    def load_model(self,ie):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
           
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_structure, self.model_weights))
        try:
            self.net2 = ie.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        #Check supported layers
        if "CPU" in self.device:
            supported_layers = ie.query_network(self.net2, "CPU")
            not_supported_layers = [l for l in self.net2.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Layers are not supported {}:\n {}".
                      format(self.device, ', '.join(not_supported_layers)))
                log.error("Specify cpu extensions using -l")
                sys.exit(1)

        # Load IR to the plugin
        log.info("Loading IR to the plugin...")
        self.exec_net = ie.load_network(network=self.net2, num_requests=0, device_name=self.device)
        
        self.input_name=next(iter(self.exec_net.inputs))
        self.input_shape=self.exec_net.inputs[self.input_name].shape
        self.output_name=next(iter(self.exec_net.outputs))
        self.output_shape=self.exec_net.outputs[self.output_name].shape

    def predict(self, image, draw_flags):
        '''
        Perform inference.
        '''
        w = image.shape[1]
        h = image.shape[0]
        
        log.info("Performing ld inference...")
        feed_dict = self.preprocess_input(image)

        #outputs=self.exec_net.infer(feed_dict)

        outputs=self.exec_net.start_async(request_id=0, inputs=feed_dict)
        while True:
            status=self.exec_net.requests[0].wait(-1)
            if status==0:
                break
            else: time.sleep(1)
        
        coords=self.preprocess_output(outputs)
        if 'ld' in draw_flags:
            self.draw_outputs(coords, image)

        left_eye=image[int(coords[1]*h)-30:int(coords[1]*h)+30, int(coords[0]*w)-30:int(coords[0]*w)+30]
        right_eye=image[int(coords[3]*h)-30:int(coords[3]*h)+30, int(coords[2]*w)-30:int(coords[2]*w)+30]
        nose=(coords[4][0],coords[5][0])
        
        return image, left_eye, right_eye, nose

    def preprocess_input(self, image):
        '''
		Preprocess input images and return dictionnary of modified images.
		'''          
        # log.info("Preprocessing the input images...")
        input_dict={}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        return input_dict

    def preprocess_output(self, outputs):
        '''
		Preprocess the output and return coodinates of BBox(s).
		'''
        r_eye_x = self.exec_net.requests[0].outputs[self.output_name][0][0]
        r_eye_y = self.exec_net.requests[0].outputs[self.output_name][0][1]
        l_eye_x = self.exec_net.requests[0].outputs[self.output_name][0][2]
        l_eye_y = self.exec_net.requests[0].outputs[self.output_name][0][3]
        nose_x = self.exec_net.requests[0].outputs[self.output_name][0][4]
        nose_y = self.exec_net.requests[0].outputs[self.output_name][0][5]    
        return (l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y)

    def draw_outputs(self, coords, image):
        '''
        Draw Bounding Boxs and texts on images.
        '''
        w = image.shape[1]
        h = image.shape[0]

        eye_right_x, eye_right_y, left_eye_x, left_eye_y, nose_x, nose_y=coords
        color = (245, 245, 245)

        cv2.circle(image, (int(nose_x*w), int(nose_y*h)), 2, (0,255,0), thickness=5, lineType=8, shift=0)

        cv2.rectangle(image, (int(eye_right_x*w)-30, int(eye_right_y*h)-30), (int(eye_right_x*w)+30, int(eye_right_y*h)+30), color, 2)       
        cv2.rectangle(image, (int(left_eye_x*w)-30, int(left_eye_y*h)-30), (int(left_eye_x*w)+30, int(left_eye_y*h)+30), color, 2)

