import numpy as np
import cv2
from src.hardware.icamera import ICamera

class ROI:
    def __init__(self, x, y, size, scale):
        self.x = x
        self.y = y
        self.size = size
        self.scale = scale

        # Initial values
        self.com = np.array([int(np.round(x / scale)), int(np.round(y / scale))]) # Centre of mass in global coordinates
        self.com_local = np.copy(self.com) # Local centre of mass
        self.mean_intensity = 0

        self.camera_roi_set = False

        self.roi = None

    def analyse(self, image_raw, detect_blobs=False):
        if not self.camera_roi_set:
            x, y = self.x / self.scale, self.y / self.scale
            roi_size_scaled = self.size / self.scale
            x1, x2 = int(np.round(max(0, x - roi_size_scaled))), int(np.round(min(image_raw.shape[1], x + roi_size_scaled)))
            y1, y2 = int(np.round(max(0, y - roi_size_scaled))), int(np.round(min(image_raw.shape[0], y + roi_size_scaled)))
            self.roi = image_raw[y1:y2, x1:x2]

            self.com = self.get_com(self.roi, x1, y1, x, y, detect_blobs=detect_blobs)
        else:
            self.roi = np.copy(image_raw)
            self.com = self.get_com(self.roi, detect_blobs=detect_blobs)

        self.mean_intensity = self.get_mean_int(self.roi)

    def get_com(self, roi, x1 = 0, y1 = 0, x = 0, y = 0, detect_blobs=False):
        if detect_blobs:
            # 1) normalize & convert to 8-bit so cv2.threshold can work
            roi_8u = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 2) apply Otsu’s threshold (output 'thresh' is 0 or maxval=1)
            _, thresh = cv2.threshold(
                roi_8u,
                0,             # ignored when using OTSU
                1,             # we want a binary mask of 0/1
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            mask = thresh.astype(bool)

            if mask.any():
                ys, xs = np.nonzero(mask)
                w = roi[ys, xs].astype(float)

                cx_raw = (xs * w).sum() / w.sum()
                cy_raw = (ys * w).sum() / w.sum()

                cx = cx_raw + x1
                cy = cy_raw + y1
            else:
                cx, cy = float(x), float(y)
                cx_raw, cy_raw = cx - x1, cy - y1
        else:
            moments = cv2.moments(roi, binaryImage=False)
            if moments["m00"] != 0:
                cx_raw = moments["m10"] / moments["m00"]
                cy_raw = moments["m01"] / moments["m00"]
                cx = cx_raw + x1
                cy = cy_raw + y1
            else:
                cx, cy = x, y

        self.com_local = np.array([cx_raw, cy_raw]) # Store local coordinates
        return np.array([cx, cy])

    def get_mean_int(self, roi):
        return np.mean(roi)

    def get_roi(self, image_raw):
        if not self.camera_roi_set:
            x, y = self.x / self.scale, self.y / self.scale
            roi_size_scaled = self.size / self.scale
            x1, x2 = int(np.round(max(0, x - roi_size_scaled))), int(np.round(min(image_raw.shape[1], x + roi_size_scaled)))
            y1, y2 = int(np.round(max(0, y - roi_size_scaled))), int(np.round(min(image_raw.shape[0], y + roi_size_scaled)))
            roi = image_raw[y1:y2, x1:x2]
        else:
            roi = image_raw

        return roi
    
    def set_camera_roi(self, camera: ICamera):
        x_scaled, y_scaled = self.x / self.scale, self.y / self.scale
        roi_size_scaled = self.size / self.scale

        x_corner = int(np.round(max(0, x_scaled - roi_size_scaled)))
        y_corner = int(np.round(max(0, y_scaled - roi_size_scaled)))
                       
        roi_full_size = int(np.round(2 * roi_size_scaled))

        camera.set_roi(x_corner, y_corner, roi_full_size, roi_full_size)

        self.camera_roi_set = True

    def plot(self, image_col, num=None):
        # Draw the ROI
        x1, x2 = max(0, self.x - self.size), min(image_col.shape[1], self.x + self.size)
        y1, y2 = max(0, self.y - self.size), min(image_col.shape[0], self.y + self.size)
        cv2.rectangle(image_col, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw the centre of mass
        cx, cy = int(np.round(self.com[0] * self.scale)), int(np.round((self.com[1] * self.scale)))
        cv2.line(image_col, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 0, 255), 2)
        cv2.line(image_col, (cx + 5, cy - 5), (cx - 5, cy + 5), (0, 0, 255), 2)

        # Draw the number
        if num is not None:
            cv2.putText(image_col, str(num + 1), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        


class ROISelector:
    def __init__(self, image, roi_size=5, default_width=800, threshold_min=1e-3, window_name="Image"):
        self.min_roi_size = roi_size
        self.roi_size = roi_size
        self.default_width = default_width
        self.rois: list[ROI] = []
        self.preview_roi = None
        self.threshold = 1.0
        self.threshold_min = threshold_min
        self.window_name = window_name

        self.image_raw = np.copy(image)

        # Preprocess the image
        self.image_col, self.scale = self._preprocess_image(image, self.threshold)

        # Create the window and callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._select_points, {"image": image})

    @staticmethod
    def print_instructions():
        print("Select points in the image")
        print("Drag the mouse to select a region of interest (ROI) around the point.")
        print("Use the mouse wheel to adjust the intensity threshold.")
        print("Press 'Esc' to finish selection.")

    def update(self):
        display = self._draw_points(self.image_col, self.rois, self.preview_roi, self.scale)
        cv2.imshow(self.window_name, display)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def check_esc_key(self):
        key = cv2.waitKey(1) & 0xFF
        esc_pressed = key == 27
        return esc_pressed

    def _draw_points(self, image, rois, preview_roi=None, scale=1.0):
        img = image.copy()

        for i, roi in enumerate(rois):
            roi.plot(img, num=i)

        if preview_roi:
            x, y = preview_roi
            preview_roi_obj = ROI(x, y, self.roi_size, scale)
            is_close = any(self._is_close_to_existing_roi(existing_roi, preview_roi_obj) for existing_roi in self.rois)
            color = (0, 0, 255) if is_close else (0, 255, 255)
            cv2.rectangle(img, (x - self.roi_size, y - self.roi_size),
                          (x + self.roi_size, y + self.roi_size), color, 1)

        return img

    def _select_points(self, event, x, y, flags, param):
        state = "idle"

        if event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                state = "dragging"
            else:
                state = "preview"

        elif event == cv2.EVENT_LBUTTONDOWN:
            state = "start_drag"

        elif event == cv2.EVENT_LBUTTONUP:
            state = "end_drag"

        elif event == cv2.EVENT_MOUSEWHEEL:
            state = "scroll"

        if state == "preview":
            self.preview_roi = (x, y)

        elif state == "start_drag":
            self.roi_size = self.min_roi_size
            self.drag_start = (x, y)
            self.preview_roi = self.drag_start

        elif state == "dragging":
            self.roi_size = self.min_roi_size + int(np.linalg.norm(np.array((x, y)) - np.array(self.drag_start)))

        elif state == "end_drag":
            self.roi_size = self.min_roi_size + int(np.linalg.norm(np.array((x, y)) - np.array(self.drag_start)))
            roi = ROI(self.drag_start[0], self.drag_start[1], self.roi_size, self.scale)

            # Remove existing ROI if its centre of mass is close to the new ROI's centre of mass
            for i, existing_roi in enumerate(self.rois):
                if self._is_close_to_existing_roi(existing_roi, roi):
                    self.rois.pop(i)
                    print(f"Removed ROI {i + 1} at {existing_roi.com}")
                    self.preview_roi = None
                    return

            # Else, add the new ROI
            roi.analyse(param["image"])
            self.rois.append(roi)
            print(f"Added ROI {len(self.rois)} at {roi.com}")
            self.preview_roi = None
            self.roi_size = self.min_roi_size

        elif state == "scroll":
            wheel_dir = flags > 0
            self._adjust_threshold(increase=wheel_dir)

    def _is_close_to_existing_roi(self, existing_roi, roi: ROI):
        min_roi_size_scaled = int(np.round(self.min_roi_size / self.scale))
        is_close = abs(existing_roi.com[0] - roi.com[0]) < min_roi_size_scaled and abs(existing_roi.com[1] - roi.com[1]) < min_roi_size_scaled
        return is_close

    def _preprocess_image(self, image, threshold):
        intensity_thresholded = np.minimum(image, np.amax(image) * threshold)
        height, width = intensity_thresholded.shape[:2]
        scale = self.default_width / width
        new_height = int(height * scale)
        image_resized = cv2.resize(intensity_thresholded, (self.default_width, new_height))
        image_scaled = cv2.normalize(image_resized, None, 0, 255, cv2.NORM_MINMAX)
        image_uint = image_scaled.astype(np.uint8)
        image_col = cv2.cvtColor(image_uint, cv2.COLOR_GRAY2BGR)

        return image_col, scale

    def _adjust_threshold(self, increase):
        thres_start = self.threshold

        if increase:
            self.threshold = min(self.threshold * 1.1, 1.0)
        else:
            self.threshold = max(self.threshold / 1.1, self.threshold_min)

        if thres_start != self.threshold:
            print(f"Image threshold: {self.threshold}")
            self.image_col, _ = self._preprocess_image(self.image_raw, self.threshold)
