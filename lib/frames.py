import os
import time
import json
import glob
import subprocess
import shlex
import cv2
from PIL import Image, ImageDraw, ImageFont
import imagehash
import numpy as np
from matplotlib import pyplot as plt 
from pathlib import Path
from urllib.parse import urlparse
import faiss
from lib import util
from lib.frame import Frame
from lib import frame_utils
import re
import boto3
from lib import image_utils
import faiss
from functools import cmp_to_key
import numpy as np
from numpy import dot
from numpy.linalg import norm
from IPython.display import display
from IPython.display import Image as DisplayImage

config = {
   "FRAME_PATH": "frames",
   "TITAN_MODEL_ID": 'amazon.titan-embed-image-v1',
   "TITAN_PRICING": 0.00006
}

# Must be h.264 video
class VideoFrames:
  def __init__(self, video_file, bucket, max_res=(750, 500), sample_rate_fps=1, force=False):
    self.frames = []
    self.video_file = video_file
    self.filtered = False
    self.filtered_config = None 
    self.stream_info = self.get_stream_info(video_file, force)
    self.start_ms = int(float(self.stream_info['format']['start_time'])*1000)
    self.duration_ms = self.stream_info['video_stream']['duration_ms']
    self.end_ms = self.start_ms + self.duration_ms
    self.object = util.upload_object(bucket, self.video_stem(), video_file)
    self.fps_extract_frames(force, max_res=max_res, sample_rate_fps=sample_rate_fps)


  def make_vector_store(self):
    dimension = len(self.frames[0]['titan_multimodal_embedding'])
    self.vector_store = self.create_index(dimension)
    #self.calculate_similar_frames()

  def video_stem(self):
      return Path(self.video_file).stem

  def video_dir(self):
      return Path(self.video_file).parent

  def video_asset_dir(self):
      return os.path.join(self.video_dir(), self.video_stem())

  def frame_dir(self):
      return os.path.join( self.video_asset_dir(), config["FRAME_PATH"])

  def get_stream_info(self, video_file, force):
    """
      Extract and return detailed information about a video file. 
      The function uses the `ffprobe` command-line tool (part of FFmpeg) to extract detailed metadata about the given video, including 
       - Whether the video is progressive
       - Width and height
       - Duration in milliseconds
       - Number of frames
       - Frame rate
       - Sample aspect ratio
       - Display aspect ratio
       - Display width
       The information extract is saved in a JSON file for downstream consumption.
       Input: 
           video_file - video file locally on disk
           force - flag to indicate whether to recreate the video metadata.
        Output: 
           None
    """
    
    stream_info_file = os.path.join(self.video_asset_dir(), 'stream_info.json')

    # check to see if stream info already exists
    if os.path.exists(stream_info_file) and force == False:
        with open(stream_info_file, 'r', encoding="utf-8") as f:
            stream_info = json.loads(f.read())
        print(f"  get_stream_info: stream_info.json already exists. Use force=True to if you want to overwrite the existing stream_info. SKIPPING...")
        return stream_info
    
    video = urlparse(self.video_file)

    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(self.video_file):
        raise Exception('input video does not exist')
        
    util.mkdir(self.video_asset_dir())

    command_string = f'ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(self.video_file)}'
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    child_process = subprocess.Popen(
        shlex.split(command_string),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = child_process.communicate()

    stream_info = json.loads(str(stdout, 'utf-8'))
    stream_info['format']['filename'] = self.video_file

    # parse video_info
    video_stream = list(filter(lambda x: (x['codec_type'] == 'video'), stream_info['streams']))[0]

    progressive = bool('field_order' not in video_stream or video_stream['field_order'] == 'progressive')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration_ms = int(float(video_stream['duration']) * 1000)
    num_frames = int(video_stream['nb_frames'])
    framerate = util.to_fraction(video_stream['r_frame_rate'])
    sample_aspect_ratio = util.to_fraction(video_stream['sample_aspect_ratio'])
    display_aspect_ratio = util.to_fraction(video_stream['display_aspect_ratio'])
    display_width = int((width * sample_aspect_ratio.numerator) / sample_aspect_ratio.denominator)

    stream_info['video_stream'] = {
        'duration_ms': duration_ms,
        'num_frames': num_frames,
        'framerate': (framerate.numerator, framerate.denominator),
        'progressive': progressive,
        'sample_aspect_ratio': (sample_aspect_ratio.numerator, sample_aspect_ratio.denominator),
        'display_aspect_ratio': (display_aspect_ratio.numerator, display_aspect_ratio.denominator),
        'encoded_resolution': (width, height),
        'display_resolution': (display_width, height),
    }

    util.save_to_file(stream_info_file, stream_info)

    return stream_info

  def fps_extract_frames(self, force, max_res=(750, 500), sample_rate_fps=1):
    """
    Extract individual frames from a video file as JPEG images.

    Args:
        max_res (tuple, optional): The maximum resolution for the extracted frames. Defaults to (750, 500).
        sample_rate_fps (int, optional): The desired frame rate (frames per second) for the extracted frames.
            Defaults to 1.

    Returns:
        list: A list of file paths for the extracted JPEG frames.

    Raises:
        Exception: If the input video URL is not a valid local file path or HTTPS URL.
        Exception: If the input video file does not exist (for local file paths).

    Description:
        This function extracts individual frames from a video file and saves them as JPEG images
        in a 'frames' directory within the same directory as the video file.

        The function takes the video URL, the previously extracted stream_info (from the probe_stream function),
        an optional max_res parameter to specify the maximum resolution for the extracted frames, and an optional
        sample_rate_fps parameter to specify the desired frame rate for the extracted frames.

        If the video is interlaced, the function applies deinterlacing using the 'yadif' filter from FFmpeg.
        It then downscales the frames to the specified max_res using the 'scale' filter.

        The extracted frames are saved with sequential filenames (e.g., frames.0000001.jpg, frames.0000002.jpg, etc.)
        in the 'frames' directory. If the 'frames' directory already exists, the function skips the extraction process
        and returns the existing JPEG files.

        The function performs input validation to ensure that the provided video URL is a valid local file path
        or HTTPS URL, and that the file exists if it's a local file path.
    """
    video = urlparse(self.video_file)
    video_path = video.path
    video_dir = self.video_dir()
    frames_file = os.path.join(self.video_asset_dir(), 'frames.json')

    # check to see if the frame info already exists
    if os.path.exists(frames_file) and force == False:
        with open(frames_file, 'r', encoding="utf-8") as f:
            self.frames = json.loads(f.read())
        print(f"  extract_frames: frames.json already exists. Use force=True to if you want to force running frame extraction. SKIPPING...")
        return
    
    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(self.video_file):
        raise Exception('input video does not exist')
    
    frame_dir = self.frame_dir()
    util.mkdir(frame_dir)
    # Can't skip becasue we need the timestamps from the ffmpage command even if we have frames 
    #if len(self.frames) > 0 and force == False:
    #   print(f"  extract_frames: found frames. SKIPPING... Use the force=True option to regenerate frames")
    
    t0 = time.time()
    video_filters = []
    video_stream = self.stream_info['video_stream']
    
    # This is a video filter option that sets the frame rate to the specified frames per second. 
    video_filters.append(f"select='eq(n,0) + not(eq(n,0))*gte(t-prev_selected_t,{ 1/sample_rate_fps })'")
    video_filters.append(f"showinfo")
    
    
    # need deinterlacing
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')
    
    # downscale image
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")
    '''
    !ffmpeg -i Netflix_Open_Content_Meridian.mp4 -vf "select='if(eq(n,0),1,floor(t)-floor(prev_selected_t))',scale=392:220" -vsync 0 -qmin 1 -q:v 1 -f image2 ./Netflix_Open_Content_Meridian/frames/frames%07d.jpg
    '''
    command = [
        'ffmpeg',
        '-i',
        shlex.quote(self.video_file),
        '-vf',
        f"{','.join(video_filters)}",
        '-vsync',
        '0',
        '-qmin',
        '1',
        '-q:v',
        '1',
        
        '-f',
        'image2',
        f"{shlex.quote(frame_dir)}/frames%07d.jpg"
    ]
    print(f" Extracting frames ...")
    print(f"  Resizing: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")
    
    self.frames = []
    try:
        # user input is wrapped in shlex above
        # nosemgrep: dangerous-subprocess-use-audit
        result = subprocess.run(command, shell=False, stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, universal_newlines=True, check=True)
        
        jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
        
        index = 0
        for line in result.stdout.splitlines():
            #print(line)
            if "pts_time" in line:
                timestamp = float(line.split("pts_time:")[1].split()[0])
                newFrame = {}
                newFrame['image_file'] = jpeg_frames[index]
                newFrame['timestamp_millis'] = int(timestamp * 1000)
                newFrame['id'] = index
                self.frames.append(newFrame)
                index+=1
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stdout}")

    util.save_to_file(frames_file, self.frames)
    
    t1 = time.time()
    print(f"  Elapsed time: {round(t1 - t0, 2)}s")
    print(f"  Frames extracted: {len(self.frames)}")
    
    return 

  def composite_image(self, start_index, end_index, cols, variation="image", folder='composite-images'):
        """
        Create an image grid using the image frames from the given start and end index 
        and organized in the given number of columns.
        Args:
           start_index - start index of the picture frames in the current object
           end_index - end index of the picture frames in the current object
           cols - number of columns to organize the image grid
           variation - variation for the image grid
           folder - the folder to store the generated image grid.
        Returns:
           the file name path for generated image grid.
        """
        image_files = []
        frames = self.frames[start_index:end_index]
    
        composite_folder = os.path.join(self.video_asset_dir(), folder)
        os.makedirs(composite_folder, exist_ok=True)
        
        composite_image, grid_layout = frame_utils.create_grid_image(frames,
                         cols,
                         border_width = 2,
                         border_outline = (0, 0, 0),
                         burn_timecode = False)
        
        
        composite_file_name = os.path.basename(frames[0]["image_file"]).split(".")[0] + os.path.basename(frames[-1]["image_file"]).split(".")[0] + '.jpg'
        composite_image_file = os.path.join(composite_folder, composite_file_name)
        composite_image.save(composite_image_file)
        composite_image.close()
        
        return composite_image_file

  def create_composite_images(self, frames, folder, prefix='shot_'):
        folder = os.path.join(self.video_asset_dir(), folder)
        os.makedirs(folder, exist_ok=True) 
        composite_images = frame_utils.create_composite_images(
                frames,
                output_dir = folder,
                prefix=prefix,
                max_dimension = (1568, 1568),
                burn_timecode = False
                )
        return composite_images
      
  def display_frames(self, start=0, end=0, ncol=10):
        
        composite_image_file = self.composite_image(start, end, ncol) # creates a grid of images from 0-100 in 10 columns
        # Display the composite image 
        display(DisplayImage(filename=composite_image_file))

  def calculate_similar_frames(self, min_similarity = 0.80, max_interval = 30):
      
      frame_embeddings = [frame['titan_multimodal_embedding'] for frame in self.frames]

      for idx, frame in enumerate(self.frames):
          similar_frames = self.search_similarity(idx, min_similarity = min_similarity, time_range = max_interval)
          frame['similar_frames'] = similar_frames
      
    
  def create_index(self, dimension):
        index = faiss.IndexFlatIP(dimension) # cosine similarity

        self.index_frames(index)
        return index

  def index_frames(self, index):
        for frame in self.frames:
            embedding = np.array([frame['titan_multimodal_embedding']])
            index.add(embedding)
      
        return index
    
  def cosine_similarity(self, a, b):
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        return cos_sim

  def search_similarity(self, idx, k = 20, min_similarity = 0.80, time_range = 30):
        
        embedding = np.array([self.frames[idx]['titan_multimodal_embedding']])
    
        D, I = self.vector_store.search(embedding, k)
    
        similar_frames = [
            {
                'idx': int(i),
                'similarity': float(d)
            } for i, d in zip(I[0], D[0])
        ]
    
        # filter out lower similiarity
        similar_frames = list(
            filter(
                lambda x: x['similarity'] > min_similarity,
                similar_frames
            )
        )
    
        similar_frames = sorted(similar_frames, key=lambda x: x['idx'])
    
        # filter out frames that are far apart from the current frame idx
        filtered_by_time_range = [{
            'idx': idx,
            'similarity': 1.0
        }]
        # filtered_by_time_range = [similar_frames[0]]
    
        for i in range(0, len(similar_frames)):
            prev = filtered_by_time_range[-1]
            cur = similar_frames[i]
    
            if abs(prev['idx'] - cur['idx']) < time_range:
                   filtered_by_time_range.append(cur)
    
        return filtered_by_time_range

  def collect_similar_frames(self, start_frame_id, end_frame_id, min_similarity = 0.80, max_interval = 30):
        """
        Creates a sorted list of frames between the given frame IDs based on similarity.

        Args: 
           start_frame_id - the start of the frame index as a boundary
           end_frame_id - the end of the frame index as a boundary
        Returns:
           sorted list of all similar frames.
        
        """
        similar_frames = []
        for frame_id in range(start_frame_id, end_frame_id+1):
            similar_frames_ids = [frame['idx'] for frame in self.search_similarity(frame_id, min_similarity = min_similarity, time_range = max_interval)]
            similar_frames.extend(similar_frames_ids)
        # unique frames in shot
        return sorted(list(set(similar_frames)))

  def set_shot_ids(self, shots):
        frames_file = os.path.join(self.video_asset_dir(), 'frames.json')
        for s, shot in enumerate(shots):
              for i in range(shot['start_frame_id'], shot['end_frame_id']+1):
                  self.frames[i]['shot_id'] = s
        util.save_to_file(frames_file, self.frames)
      
          
  def collect_related_shots(self, frame_ids):
        """
        Creates a sorted list of shots ids based on the similar image frames within a shot.
        Args:
           frame_ids - collection of frame ids to find the unique shots from.
        Returns:
           sorted list of unique short IDs.
        """
        related_shots = []
        for frame_id in frame_ids:
            if ('shot_id' in self.frames[frame_id]):
                related_shots.append(self.frames[frame_id]['shot_id'])
        # unique frames in shot
        return sorted(list(set(related_shots)))

  def make_titan_multimodal_embeddings(self):
        """
        Creates an image embedding for a frame. This function uses titan multimodal embedding model 
        to create an embedding for the image frame.
        
        Args:
           None
        
        """

        video = urlparse(self.video_file)
        video_path = video.path
        video_dir = self.video_dir()
        frames_file = os.path.join(self.video_asset_dir(), 'frames.json')
      
        titan_model_id = config['TITAN_MODEL_ID']
        accept = 'application/json'
        content_type = 'application/json'

        bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

        self.cost_embeddings = 0

        for frame in self.frames:
            with Image.open(frame['image_file']) as image:
                input_image = image_utils.image_to_base64(image)
    
            model_params = {
                'inputImage': input_image,
                'embeddingConfig': {
                    'outputEmbeddingLength': 384 #1024 #384 #256
                }
            }
    
            body = json.dumps(model_params)
    
            response = bedrock_runtime_client.invoke_model(
                body=body,
                modelId=titan_model_id,
                accept=accept,
                contentType=content_type
            )
            response_body = json.loads(response.get('body').read())
    
            frame['titan_multimodal_embedding'] = response_body['embedding']
            frame['titan_multimodal_embedding_model_id'] = titan_model_id
            self.cost_embeddings = self.cost_embeddings + 0.00006

        util.save_to_file(frames_file, self.frames)

        # save embeddings
        util.save_to_file(fastpath_embeddings_file, self.frames)
        self.store_fastpath_results("frames-embeddings.json")
        
        return

  def load_titan_multimodal_embeddings(self):
        """
        Loads precomputed image embeddings for a video. 
        
        Args:
           None
        
        """

        self.load_fastpath_results("frames-embeddings.json")           
        self.cost_embeddings = len(self.frames) + 0.00006
      
        return
  
  def store_fastpath_results(self, result_file_name):
      
        fastpath_dir = f"./fastpath/{ self.video_stem() }"
        fastpath_file = f"./fastpath/{ self.video_stem() }/{ result_file_name }"
    
        # check to see if there is a fastpath folder for this video 
        if not os.path.exists(fastpath_dir):
            util.mkdir(fastpath_dir)

        # save results
        util.save_to_file(fastpath_file, self.frames)
    
        return
      
  def load_fastpath_results(self, result_file_name):
      
        fastpath_dir = f"./fastpath/{ self.video_stem() }"
        fastpath_file = f"./fastpath/{ self.video_stem() }/{ result_file_name }"
    
        # check to see if the frame info already exists
        if os.path.exists(fastpath_file):
            with open(fastpath_file, 'r', encoding="utf-8") as f:
                self.frames = json.loads(f.read())
            print(f"  load_fastpath_results: loaded frames from { fastpath_file }")
        else:
            print(f"  load_fastpath_results: no stored frames exist at { fastpath_file }.  Run with FASTPATH=False to generate results")  
            
        return