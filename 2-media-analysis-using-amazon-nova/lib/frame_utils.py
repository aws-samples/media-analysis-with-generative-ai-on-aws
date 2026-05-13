import os
import json
import math
import base64
from pathlib import Path
from io import BytesIO
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont
from shutil import rmtree
from IPython.display import display
      
def image_to_base64(image):
    """
    Convert a PIL image to base64 encoding
    :param image: a PIL image
    :return: base64 encoding of the image
    """
    
    buff = BytesIO()
    image.save(buff, format='JPEG')
    return base64.b64encode(buff.getvalue()).decode('utf8')

def image_to_bytes(image):
    """
    Convert a PIL image to bytes
    :param image: a PIL image
    :return: bytes of the image
    """

    buff = BytesIO()
    image.save(buff, format='JPEG')
    return buff.getvalue()

def create_grid_image(frames,
                      #max_ncol = 10,
                      max_ncol,
                      border_width = 2,
                      border_outline = (0, 0, 0),
                      burn_timecode = False):
    """
    Create a grid image from a list of image files
    :param frames: list of image files
    :param max_ncol: maximum number of columns
    :param border_width: width of the border
    :param border_outline: outline color of the border
    :param burn_timecode: whether to burn in timecode to the image
    :return: a PIL image object and a list of tuples containing the image file and its coordinate
    """

    should_resize = len(frames) > 100

    with Image.open(frames[0]['image_file']) as image:
        width, height = image.size

    if should_resize:
        width = width // 2
        height = height // 2

    ncol = max_ncol
    if len(frames) < max_ncol:
        ncol = len(frames)

    nrow = len(frames) // ncol
    if len(frames) % ncol > 0:
        nrow += 1

    # grid layout
    grid_layout = []

    # Create a new image to hold the grid
    grid_width = width * ncol
    grid_height = height * nrow
    grid_image = Image.new('RGB', (grid_width, grid_height))

    draw = ImageDraw.Draw(grid_image)
    # Paste the individual images into the grid
    for i, frame in enumerate(frames):
        with Image.open(frame["image_file"]) as image:
            if should_resize:
                image = image.resize((width, height))
            if burn_timecode:
                seconds = int(Path(image_file).stem.split('.')[1]) - 1
                timecode = to_hhmmssms(seconds * 1000)
                image = burn_in_timecode(image, timecode)
            x = (i % ncol) * width
            y = (i // ncol) * height
            grid_image.paste(image, (x, y))
            

        # draw border
        draw.rectangle((x, y, x + width, y + height), outline=border_outline, width=border_width)

        coord = (x, y, width, height)
        grid_layout.append((frame['image_file'], coord))

    return [grid_image, grid_layout]

def create_composite_images(frames,
                            output_dir = 'composite-images',
                            prefix = "frame_",
                            max_dimension = (1568, 1568),
                            burn_timecode = False
                            ):
    """
    Create composite images from a list of image files
    :param frames: list of image files
    :param output_dir: output directory
    :param max_dimension: maximum dimension of the composite image. 1568x1568 fits within common vision model input limits before downscaling.
    :param burn_timecode: whether to burn in timecode to the image
    :return: a list of dictionaries containing the file name and layout of the composite image
    """
    
    mkdir(output_dir)

    with Image.open(frames[0]["image_file"]) as image:
        width, height = image.size

    ncol, nrow = max_dimension[0] // width, max_dimension[1] // height
    grid_size = ncol * nrow

    composite_images = []
    for i in range(0, len(frames), grid_size):
        frames_per_image = frames[i:i+grid_size]
        composite_image, image_layout = create_grid_image(
            frames_per_image,
            ncol,
            burn_timecode = burn_timecode)

        # save the composite image
        start_frame = os.path.basename(frames[i]['image_file']).split('.')[0]
        end_frame = os.path.basename(frames[min((i+grid_size), len(frames)-1)]['image_file']).split('.')[0]
        name = f"{prefix}{start_frame}-{end_frame}.jpg"
        jpeg_file = os.path.join(output_dir, name)
        composite_image.save(jpeg_file)
        composite_image.close()

        composite_images.append({
            'file': jpeg_file,
            'layout': image_layout,
            'width': width,
            'height': height
        })

    return composite_images

def plot_composite_images(composite_images):
    """
    Plot composite images for display purpose
    :param composite_images: a list of composite images
    """

    for idx, composite_image in enumerate(composite_images):
        print(f"Composite image ##{idx + 1}")

        if isinstance(composite_image, str):
            with Image.open(composite_image) as image:
                width, height = image.size
                width = round((width // 3) / 2) * 2
                height = round((height // 3) / 2) * 2
                with image.resize((width, height)) as resized_image:
                    display(resized_image)
        else:
            width, height = composite_images.size
            width = round((width // 3) / 2) * 2
            height = round((height // 3) / 2) * 2
            with composite_images.resize((width, height)) as resized_image:
                display(resized_image)

def burn_in_timecode(image, timecode):
    """
    Burn in timecode to the image
    :param image: a PIL image
    :param timecode: timecode to burn in
    :return: a PIL image with timecode burned in
    """

    with Image.new('RGB', (96, 28)) as tc_image:
        default_font = ImageFont.load_default()
        draw = ImageDraw.Draw(tc_image)
        draw.text((4, 4), timecode, align='center', color=(255, 255, 255), font=default_font)
        image.paste(tc_image, (2, 2))
        tc_image.close()

    return image

def to_fraction(s):
    """
    Convert a string or float to a Fraction object
    """

    if isinstance(s, str):
        return Fraction(s.replace(':', '/')) 
    return Fraction(s)

def to_hhmmssms(milliseconds, with_msec = True):
    """
    Convert milliseconds to hours, minutes, seconds, and milliseconds
    :param milliseconds: time in milliseconds
    :param with_msec: whether to include milliseconds
    :return: formatted time string
    """

    hh = math.floor(milliseconds / 3600000)
    mm = math.floor((milliseconds % 3600000) / 60000)
    ss = math.floor((milliseconds % 60000) / 1000)
    ms = math.ceil(milliseconds % 1000)
    if not with_msec:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def to_milliseconds(timestamp):
    """
    Convert a timestamp string to milliseconds
    """

    hh, mm, ss = timestamp.split(':')
    ss, ms = ss.split('.')
    hh, mm, ss, ms = map(int, (hh, mm, ss, ms))

    return (((hh * 3600) + (mm * 60) + ss) * 1000) + ms

def mkdir(directory):
    """
    Create a directory if it doesn't exist. Ignore any exception
    """

    try:
        os.mkdir(directory)
    except:
        pass

def rmdir(directory):
    """
    Remove a directory if it exists. Ignore any exception
    """

    try:
        rmtree(directory)
    except:
        pass

def save_to_file(output_file, data):
    """
    Save data to a file
    :param output_file: the path to save the file
    :param data: the data to save
    :return: the output file
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, ensure_ascii=False)

    return output_file

def box_in_box(box_s, box_l):
    """
    Check if one bounding box is inside another bounding box.
    :param box_s: smaller bounding box
    :param box_l: larger bounding box
    :return: True if box_s is inside box_l, False otherwise
    """

    l1, t1, w1, h1 = box_s
    l2, t2, w2, h2 = box_l
    return l1 >= l2 and t1 >= t2 and l1 + w1 <= l2 + w2 and t1 + h1 <= t2 + h2

def scale_bbox(bbox, factor):
    """
    Scale bounding box
    :param bbox: bounding box (left, top, width, height)
    :param factor: scaling factor (x, y)
    :return: scaled bounding box
    """

    if isinstance(factor, tuple):
        factor_x, factor_y = factor
    else:
        factor_x, factor_y = factor, factor

    l, t, w, h = bbox
    delta_w, delta_h = ((w * factor_x) - w) / 2, ((h * factor_y) - h) / 2
    new_x0, new_y0 = max(0, l - delta_w), max(0, t - delta_h)
    new_x1, new_y1 = l + w + delta_w, t + h + delta_h

    new_l = new_x0
    new_t = new_y0
    new_w = new_x1 - new_x0
    new_h = new_y1 - new_y0

    return (new_l, new_t, new_w, new_h)

def search(items, field, value):
    """
    Search item in list of items by field and value
    :param items: list of items
    :param field: field to search
    :param value: value to search
    :return: item if found, None otherwise
    """

    for item in items:
        if item[field] == value:
            return item
    return None

def search_box_in_grid_layouts(bbox, grid_layouts):
    """
    Search box in grid layouts
    :param bbox: bounding box (left, top, width, height)
    :param grid_layouts: list of grid layouts
    :return: grid layout if found, None otherwise
    """

    for grid_layout in grid_layouts:
        _, grid_bbox = grid_layout

        if box_in_box(bbox, grid_bbox):
            return grid_layout

    return None

def create_grid_image_from_files(image_files, max_ncol = 10, border_width = 2):
    should_resize = len(image_files) > 50

    with Image.open(image_files[0]) as image:
        width, height = image.size

    ncol = max_ncol
    if len(image_files) < max_ncol:
        ncol = len(image_files)

    nrow = len(image_files) // ncol
    if len(image_files) % ncol > 0:
        nrow += 1
    
    # Create a new image to hold the grid
    grid_width = width * ncol
    grid_height = height * nrow
    grid_image = Image.new("RGB", (grid_width, grid_height))

    draw = ImageDraw.Draw(grid_image)
    # Paste the individual images into the grid
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        if should_resize:
            image = image.resize((width, height))
        x = (i % ncol) * width
        y = (i // ncol) * height
        grid_image.paste(image, (x, y))
        # draw border
        draw.rectangle((x, y, x + width, y + height), outline=(0, 0, 0), width=border_width)
    
    return grid_image
    
def skip_frames(frames, max_frames = 80):
    if len(frames) < max_frames:
        return frames

    # miminum step = 2
    skip_step = max(round(len(frames) / max_frames), 2)

    output_frames = []
    for i in range(0, len(frames), skip_step):
        output_frames.append(frames[i])
    
    return output_frames

def plot_shots(frames, num_shots):
    mkdir('shots')

    shots = [[] for _ in range(num_shots)]
    for frame in frames.frames:
        shot_id = frame['shot_id']
        file = frame['image_file']
        shots[shot_id].append(frame)

    for i in range(len(shots)):
        shot = shots[i]
        num_frames = len(shot)
        skipped_frames = skip_frames(shot)
        skipped_frames = skip_frames(shot)
        skipped_frames_files = [frame['image_file'] for frame in skipped_frames]
        grid_image = create_grid_image_from_files(skipped_frames_files, 10)
        #print(f'grid_image {json.dumps(grid_image)}')
        w, h = grid_image.size
        if h > 440:
            grid_image = grid_image.resize((w // 2, h // 2))
        w, h = grid_image.size
        print(f"Shot #{i:04d}: {num_frames} frames ({len(skipped_frames)} drawn) [{w}x{h}]")
        grid_image.save(f"shots/shot-{i:04d}.jpg")
        display(grid_image)
        grid_image.close()
        print('====')
