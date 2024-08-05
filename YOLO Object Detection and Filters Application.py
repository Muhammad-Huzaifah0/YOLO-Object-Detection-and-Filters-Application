import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from tkinter import ttk

class YOLOFaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection and Filters")
        self.root.geometry("1024x768")

        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""

        # Configure the main grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10 10 10 10")
        self.main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Create a frame for the image and video display
        self.display_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN)
        self.display_frame.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.rowconfigure(0, weight=1)

        # Main display canvas
        self.canvas = tk.Canvas(self.display_frame, bg="black")
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Create a frame for control buttons
        self.controls_frame = ttk.Frame(self.main_frame, padding="10 10 10 10", relief=tk.RAISED)
        self.controls_frame.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.controls_frame.columnconfigure(0, weight=1)
        
        # Create control buttons
        self.create_control_buttons()

        # Status bar for messages
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))

        # Initialize variables
        self.image_path = None
        self.video_path = None
        self.image = None
        self.video_capture = None
        self.net = None
        self.classes = None
        self.output_layers = None
        self.filter_mode = None
        self.detect_objects_flag = False
        self.running = False
        self.thread = None

    def create_control_buttons(self):
        media_buttons = [
            ("Select Weights", self.select_weights),
            ("Select CFG", self.select_cfg),
            ("Select Names", self.select_names),
            ("Select Image", self.select_image),
            ("Select Video", self.select_video),
            ("Live Video", self.toggle_live_video)
        ]

        for i, (text, command) in enumerate(media_buttons):
            btn = ttk.Button(self.controls_frame, text=text, command=command)
            btn.grid(column=0, row=i, pady=5, sticky=(tk.W, tk.E))

        filter_buttons = [
            ("Detect Objects", self.toggle_detect_objects),
            ("Edge Detection", self.toggle_edge_detection),
            ("Sharpen", self.toggle_sharpen),
            ("Gaussian Blur", self.toggle_gaussian_blur),
            ("Brightness", self.toggle_brightness),
            ("Erosion", self.toggle_erosion),
            ("Dilation", self.toggle_dilation),
            ("Sepia Tone", self.toggle_sepia),
            ("Contrast Adjustment", self.toggle_contrast),
            ("Negative", self.toggle_negative),
            ("Emboss", self.toggle_emboss)
        ]

        for i, (text, command) in enumerate(filter_buttons):
            btn = ttk.Button(self.controls_frame, text=text, command=command)
            btn.grid(column=0, row=len(media_buttons) + i, pady=5, sticky=(tk.W, tk.E))

    def select_weights(self):
        self.weights_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_cfg(self):
        self.cfg_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_names(self):
        self.names_path = filedialog.askopenfilename()
        self.load_yolo()

    def load_yolo(self):
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")

    def select_image(self):
        self.stop_running()
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.load_image()

    def select_video(self):
        self.stop_running()
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.running = True
            self.thread = threading.Thread(target=self.detect_objects_video)
            self.thread.start()

    def toggle_live_video(self):
        self.stop_running()
        if not self.running:
            self.video_capture = cv2.VideoCapture(0)
            self.running = True
            self.thread = threading.Thread(target=self.show_live_video)
            self.thread.start()

    def stop_running(self):
        self.running = False
        self.detect_objects_flag = False
        self.filter_mode = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.canvas.delete("all")
        cv2.destroyAllWindows()

    def load_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_image(image)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def resize_image(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_height, image_width, _ = image.shape
        scale = min(canvas_width / image_width, canvas_height / image_height)
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)
        return cv2.resize(image, (new_width, new_height))

    def resize_video_frame(self, frame):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        frame_height, frame_width, _ = frame.shape
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black background image of canvas size
        background = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate center offset
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        # Place the resized video frame on the black background
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        return background

    def toggle_detect_objects(self):
        self.detect_objects_flag = not self.detect_objects_flag
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_edge_detection(self):
        self.filter_mode = "edge" if self.filter_mode != "edge" else None
        if self.image_path:
            self.apply_filter_to_image()
            
    def toggle_sharpen(self):
        self.filter_mode = "sharpen" if self.filter_mode != "sharpen" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_gaussian_blur(self):
        self.filter_mode = "gaussian_blur" if self.filter_mode != "gaussian_blur" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_brightness(self):
        self.filter_mode = "brightness" if self.filter_mode != "brightness" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_erosion(self):
        self.filter_mode = "erosion" if self.filter_mode != "erosion" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_dilation(self):
        self.filter_mode = "dilation" if self.filter_mode != "dilation" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_sepia(self):
        self.filter_mode = "sepia" if self.filter_mode != "sepia" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_contrast(self):
        self.filter_mode = "contrast" if self.filter_mode != "contrast" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_negative(self):
        self.filter_mode = "negative" if self.filter_mode != "negative" else None
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_emboss(self):
        self.filter_mode = "emboss" if self.filter_mode != "emboss" else None
        if self.image_path:
            self.apply_filter_to_image()

    def detect_objects_video(self):
        self.video_capture = cv2.VideoCapture(self.video_path)
        while self.video_capture.isOpened() and self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = self.resize_video_frame(frame)

            if self.detect_objects_flag and self.net:
                frame = self.apply_yolo(frame)

            frame = self.apply_filter_to_frame(frame)
            self.show_frame_on_canvas(frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()


    def show_live_video(self):
        while self.video_capture.isOpened() and self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = self.resize_video_frame(frame)

            if self.detect_objects_flag:
                frame = self.apply_yolo(frame)

            frame = self.apply_filter_to_frame(frame)
            self.show_frame_on_canvas(frame)

        self.stop_running()
    def stop_running(self):
        self.running = False
        self.detect_objects_flag = False
        self.filter_mode = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.canvas.delete("all")
        cv2.destroyAllWindows()

    def apply_yolo(self, image):
        if not self.net:
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")
            return image

        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if (x, y, w, h) and isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def apply_filter_to_frame(self, frame):
        if self.filter_mode == "edge":
            frame = cv2.Canny(frame, 100, 200)
        elif self.filter_mode == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        elif self.filter_mode == "gaussian_blur":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.filter_mode == "brightness":
            alpha = 1.2
            beta = 50
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        elif self.filter_mode == "erosion":
            kernel = np.ones((5, 5), np.uint8)
            frame = cv2.erode(frame, kernel, iterations=1)
        elif self.filter_mode == "dilation":
            kernel = np.ones((5, 5), np.uint8)
            frame = cv2.dilate(frame, kernel, iterations=1)
        elif self.filter_mode == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, sepia_filter)
            frame = np.clip(frame, 0, 255)
        elif self.filter_mode == "contrast":
            alpha = 1.5
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        elif self.filter_mode == "negative":
            frame = cv2.bitwise_not(frame)
        elif self.filter_mode == "emboss":
            kernel = np.array([[ -2, -1,  0],
                            [ -1,  1,  1],
                            [  0,  1,  2]])
            frame = cv2.filter2D(frame, -1, kernel)

        return frame

    def apply_filter_to_image(self):
        if not self.image_path:
            return
        image = cv2.imread(self.image_path)
        if self.detect_objects_flag:
            image = self.apply_yolo(image)
        image = self.apply_filter_to_frame(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def show_frame_on_canvas(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOFaceDetectionApp(root)
    root.mainloop()
