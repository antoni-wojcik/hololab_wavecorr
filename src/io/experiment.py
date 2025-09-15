import os, glob, re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys, inspect

from src.io import img_handler, santec_io

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout  # Keep original stdout

    def write(self, text):
        self.stdout.write(text)  # Print to console
        self.file.write(text)  # Save to file

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ExpIO:
    def __init__(self, name, description = None, log = True, path = None):
        self.name = name
        self.description = description
        self.start_time = datetime.now()
        
        # Path to the experiment folder if any data is saved
        self.path = path

        self.tee = None

        # Walk the call stack to find the first caller outside this file
        stack = inspect.stack()

        for frame_info in stack:
            caller_file = os.path.abspath(frame_info.filename)
            if caller_file != os.path.abspath(__file__):
                break

        self.caller_path = caller_file

        if log:
            self._start_logging()
            
            # Copy the caller file into the experiment folder for reproducibility
            if self.caller_path and os.path.exists(self.caller_path):
                caller_filename = os.path.basename(self.caller_path)
                destination = os.path.join(self.path, caller_filename)
                with open(self.caller_path, 'r') as src, open(destination, 'w') as dst:
                    dst.write(src.read())

    def load_target(self, name, extension = 'png'):
        path = os.path.join(os.path.dirname(__file__), '../../data/targets', f'{name}.{extension}')

        if extension == 'svg':
            target = img_handler.load_svg(path)
        else:
            target = img_handler.load_image(path)

        return target

    def save_hologram(self, values, name="holo", separate_dir=False):
        self._make_dir()
        path = self._get_unique_path(name, 'csv', separate_dir)
        values_ushort = np.array(values, dtype=np.uint16)
        santec_io.save_hologram_csv(values_ushort, path)
        return path

    def save_figure(self, fig: plt.Figure, name="fig", separate_dir=False):
        self._make_dir()
        path = self._get_unique_path(name, 'png', separate_dir)
        fig.savefig(path)
        return path

    def save_npy(self, data: np.ndarray, name, separate_dir=False):
        self._make_dir()
        path = self._get_unique_path(name, 'npy', separate_dir)
        np.save(path, data)
        return path

    def save_image(self, image, name, separate_dir=False, extension='png'):
        self._make_dir()
        path = self._get_unique_path(name, extension, separate_dir)
        img_handler.save_image(image, path)
        return path

    def save_text(self, text, name="text", separate_dir=False):
        self._make_dir()
        path = self._get_unique_path(name, 'txt', separate_dir)
        with open(path, 'w') as f:
            f.write(text)
        return path

    def _start_logging(self):
        self._make_dir()
        path = self._get_unique_path("log", 'txt')
        self.tee = Tee(path)
        sys.stdout = self.tee
        sys.stderr = self.tee

    def _stop_logging(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def _get_unique_path_file(self, name, extension):
        """
        Generate a unique file path in self.path with the pattern:
          - first: name.extension
          - on 2nd: rename existing to name_0.extension, return name_1.extension
          - thereafter: name_n.extension for next available n
        """
        base = os.path.join(self.path, name)
        plain = f"{base}.{extension}"
        # only consider suffixes that match numeric pattern: base_\n.ext
        pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.{re.escape(extension)}$")
        all_suffixes = glob.glob(f"{base}_*.{extension}")
        numbered = [f for f in all_suffixes if pattern.match(f)]

        # if no plain file and no numbered files exist, use plain name
        if not os.path.exists(plain) and not numbered:
            return plain

        # if plain exists but _0 is missing, rename plain -> base_0.ext
        first = f"{base}_0.{extension}"
        if os.path.exists(plain) and not os.path.exists(first):
            os.rename(plain, first)
            # return next slot
            return f"{base}_1.{extension}"

        # scan existing numbered and pick next index
        nums = [int(pattern.match(f).group(1)) for f in numbered]
        next_idx = max(nums) + 1 if nums else 1
        return f"{base}_{next_idx}.{extension}"

    def _get_unique_path_dir(self, name, extension):
        """
        Create (if needed) a subdirectory under self.path named `name`,
        then place files inside named 0.extension, 1.extension, ...
        Returns the full path to the next available numeric filename.
        """
        dir_path = os.path.join(self.path, name)
        os.makedirs(dir_path, exist_ok=True)

        # list existing files matching *.extension
        existing = [f for f in os.listdir(dir_path) if f.endswith(f".{extension}")]
        nums = []
        for f in existing:
            stem, ext = os.path.splitext(f)
            if stem.isdigit():
                nums.append(int(stem))

        next_idx = max(nums) + 1 if nums else 0
        return os.path.join(dir_path, f"{next_idx}.{extension}")

    def _get_unique_path(self, name, extension, separate_dir=False):
        """
        Unified entry point for unique path generation:
          - if separate_dir=False: use file-based naming (_get_unique_path_file)
          - if separate_dir=True: collect in subdirectory (_get_unique_path_dir)
        """
        if separate_dir:
            return self._get_unique_path_dir(name, extension)
        else:
            return self._get_unique_path_file(name, extension)

    def _make_dir(self):
        if self.path is None:
            date_str = self.start_time.strftime('%Y-%m-%d')
            base_path = os.path.join(os.path.dirname(__file__), '../../data/experiments/', f'{date_str}_{self.name}')
            os.makedirs(base_path, exist_ok=True)

            run_index = 0
            while os.path.exists(os.path.join(base_path, f'run_{run_index}')):
                run_index += 1

            self.path = os.path.join(base_path, f'run_{run_index}')
            os.makedirs(self.path)

            self._save_experiment_info()

    def _save_experiment_info(self):
        self._make_dir()
        path = self._get_unique_path('info', 'txt')
        with open(path, 'w') as f:
            f.write(f'Experiment: {self.name}\n')
            # Write start time in year month day hour minute second format
            f.write(f'Start time: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            # Write which experiment it was on that day
            f.write(f'Experiment number: {self.path.split("_")[-1]}\n')
            # Write the description if it was provided
            if self.description:
                f.write(f'Description: {self.description}\n')
            else:
                f.write('No description provided\n')
            f.write('\n')
