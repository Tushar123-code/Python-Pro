import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2
import threading
import queue


class LeftView(tk.Frame):
    def __init__(self, root):
        
        tk.Frame.__init__(self, root)
        
        self.root = root
       
        self.setup_ui()
        
    def setup_ui(self):
       
        self.output_label = tk.Label(self, text="Webcam Output", bg="black", fg="white")
        self.output_label.pack(side="top", fill="both", expand="yes", padx=10)
        
        
        self.image_label = tk.Label(self)
      
        self.image_label.pack(side="left", fill="both", expand="yes", padx=10, pady=10)
        
    def update_image(self, image):
        self.image_label.configure(image=image)
        self.image = image
    

class RightView(tk.Frame):
    def __init__(self, root):
       
        tk.Frame.__init__(self, root)
        
        self.root = root
        
        self.setup_ui()
        
    def setup_ui(self):
        
        self.output_label = tk.Label(self, text="Face detection Output", bg="black", fg="white")
        self.output_label.pack(side="top", fill="both", expand="yes", padx=10)
        
        
        self.image_label = tk.Label(self)
        
        self.image_label.pack(side="left", fill="both", expand="yes", padx=10, pady=10)
        
        
    def update_image(self, image):
       
        self.image_label.configure(image=image)
        
        self.image = image
        

class AppGui:
    def __init__(self):
        
        self.root = tk.Tk()
        
        self.root.title("Face Detection")
        
        
        self.left_view = LeftView(self.root)
        self.left_view.pack(side='left')
        
       
        self.right_view = RightView(self.root)
        self.right_view.pack(side='right')
        
       
        self.image_width=200
        self.image_height=200
        
       
        self.circle_center = (int(self.image_width/2),int(self.image_height/4))
        self.circle_radius = 15
        self.circle_color = (255, 0, 0)
        
        self.is_ready = True
        
    def launch(self):
       
        self.root.mainloop()
        
    def process_image(self, image):
        
        image = cv2.resize(image, (self.image_width, self.image_height))
        
      
        image = Image.fromarray(image)
        
        #convert image to Tk toolkit format
        image = ImageTk.PhotoImage(image)
        
        return image
        
    def update_webcam_output(self, image):
        
        image = self.process_image(image)

        
        self.left_view.update_image(image)
        
    def update_neural_network_output(self, image):
        
        image = self.process_image(image)
        
        self.right_view.update_image(image)
        
    def update_chat_view(self, question, answer_type):
        self.left_view.update_chat_view(question, answer_type)
        
    def update_emotion_state(self, emotion_state):
        self.right_view.update_emotion_state(emotion_state)
    


import cv2

class VideoCamera:
    def __init__(self):
        
        self.video_capture = cv2.VideoCapture(0)
                
    
    def __del__(self):
        self.video_capture.release()
        
    def read_image(self):
        
        ret, frame = self.video_capture.read()
        
        return ret, frame
    
     
    def release(self):
        self.video_capture.release()
        

def detect_face(img):
    
    face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);
    
    if (len(faces) == 0):
        return img
    
   
    (x, y, w, h) = faces[0]
    
    
    return img[y:y+w, x:x+h]




class WebcamThread(threading.Thread):
    def __init__(self, app_gui, callback_queue):
       
        threading.Thread.__init__(self)
       
        self.callback_queue = callback_queue
        
        
        self.app_gui = app_gui
        
        
        self.should_stop = False
        
        
        self.is_stopped = False
        
       
        self.camera = VideoCamera()
        
    
    def run(self):
        
        while (True):
            
            if (self.should_stop):
                self.is_stopped = True
                break
            
            #read a video frame
            ret, self.current_frame = self.camera.read_image()

            if(ret == False):
                print('Video capture failed')
                exit(-1)
                
            #opencv reads image in BGR color space, let's convert it 
            #to RGB space
            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            #key = cv2.waitKey(10)
            
            if self.callback_queue.full() == False:
                #put the update UI callback to queue so that main thread can execute it
                self.callback_queue.put((lambda: self.update_on_main_thread(self.current_frame, self.app_gui)))
        
        #fetching complete, let's release camera
        #self.camera.release()
        
            
    #this method will be used as callback and executed by main thread
    def update_on_main_thread(self, current_frame, app_gui):
        app_gui.update_webcam_output(current_frame)
        face = detect_face(current_frame)
        app_gui.update_neural_network_output(face)
        
    def __del__(self):
        self.camera.release()
            
    def release_resources(self):
        self.camera.release()
        
    def stop(self):
        self.should_stop = True
    
        




class Wrapper:
    def __init__(self):
        self.app_gui = AppGui()
        
        
        
        
        self.current_frame = None
        
       
        self.callback_queue = queue.Queue()
        
      
        self.webcam_thread = WebcamThread(self.app_gui, self.callback_queue)
        
       
        self.webcam_attempts = 0
        
        
        self.app_gui.root.protocol("WM_DELETE_WINDOW", self.on_gui_closing)
        
        
        self.start_video()
        
        
        self.fetch_webcam_video()
    
    def on_gui_closing(self):
        self.webcam_attempts = 51
        self.webcam_thread.stop()
        self.webcam_thread.join()
        self.webcam_thread.release_resources()
        
        self.app_gui.root.destroy()

    def start_video(self):
        self.webcam_thread.start()
        
    def fetch_webcam_video(self):
            try:
            
                 
                callback = self.callback_queue.get_nowait()
                callback()
                self.webcam_attempts = 0
                
                self.app_gui.root.after(70, self.fetch_webcam_video)
                    
            except queue.Empty:
                if (self.webcam_attempts <= 50):
                    self.webcam_attempts = self.webcam_attempts + 1
                    self.app_gui.root.after(100, self.fetch_webcam_video)

    def test_gui(self):
        
        image, gray = self.read_images()
        self.app_gui.update_webcam_output(image)
        self.app_gui.update_neural_network_output(gray)
        
        
        self.app_gui.update_chat_view("4 + 4 = ? ", "number")
        
        
        self.app_gui.update_emotion_state("neutral")
        
    def read_images(self):
        image = cv2.imread('data/test1.jpg')
    
       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image, gray
    
    def launch(self):
        self.app_gui.launch()
        
    def __del__(self):
        self.webcam_thread.stop()



wrapper = Wrapper()
wrapper.launch()

