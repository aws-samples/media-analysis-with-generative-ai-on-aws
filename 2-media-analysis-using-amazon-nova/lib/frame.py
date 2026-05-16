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
from lib import util
from lib import image_utils
import re
import boto3
import faiss
from functools import cmp_to_key
import numpy as np
from numpy import dot
from numpy.linalg import norm

TITAN_PRICING = 0.00006  # retained for reference; not used

# Amazon Nova Multimodal Embeddings — unified text/image/video/audio embedding model.
# Launched Oct 28, 2025. us-east-1 only (no Geo/Global cross-region inference).
# We pin a dedicated bedrock-runtime client to us-east-1 regardless of the
# learner's workshop region; only these small embedding calls cross regions.
EMBEDDING_MODEL_ID = 'amazon.nova-2-multimodal-embeddings-v1:0'
EMBEDDING_MODEL_REGION = 'us-east-1'
EMBEDDING_DIMENSION = 1024  # Nova MM Embed supports 256 / 512 / 1024 / 3072 via MRL

config = {
   "LAPLACIAN_PATH": "laplacian",
   "LAPLACIAN_SIZE": 18,
   "LAPLACIAN_DISTANCE": 0.2,
   "PHASH_THRESHOLD": 0.2,
   "PHASH_SIZE": 18,
   "PHASH_DISTANCE": 4
}

class Frame:
    def __init__(self, id, image_file, timestamp_millis):
        self.id = id
        self.image_file = image_file
        self.timestamp_millis = int(timestamp_millis)
        self.laplacian_variance = self.compute_laplacian_variance(ksize=3)
        self.perceptual_hash = self.compute_phash()
        self.make_laplacian_image(self.image_file, ksize=21)
        self.make_multimodal_embedding()

    def compute_laplacian_variance(self, ksize=3):
        """
        Computes laplacian variant for the given image.
        Args: 
             ksize: Aperture size used to compute the second-derivative filters. see (https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6)

        Returns:
             laplacian variant.
        """
        
        image = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variant = cv2.Laplacian(gray, cv2.CV_64F, ksize).var()

        return round(variant)
    
    def compute_phash(self):
        """
        Compute the perceptual hash for the image file for this object.
        For more information please refer to: https://en.wikipedia.org/wiki/Perceptual_hashing

        """
        with Image.open(self.image_file) as image:
            # imagehash.hex_to_hash(phash)
            phash = str(imagehash.phash(image))
            return phash

        return None

    def make_multimodal_embedding(self):
        """
        Creates an image embedding using Amazon Nova Multimodal Embeddings on Amazon Bedrock.
        The model supports text, image, video, and audio inputs and produces embeddings
        in a unified vector space for cross-modal retrieval.

        Note: Nova Multimodal Embeddings is currently only available in us-east-1, so we
        construct a dedicated bedrock-runtime client pinned to that region. Only this small
        embedding call crosses regions; the rest of the workshop continues to run in the
        learner's default region.

        Args:
           None

        """

        model_id = EMBEDDING_MODEL_ID
        accept = 'application/json'
        content_type = 'application/json'

        bedrock_runtime_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=EMBEDDING_MODEL_REGION,
        )

        with Image.open(self.image_file) as image:
            input_image = image_utils.image_to_base64(image)

        model_params = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'embeddingDimension': EMBEDDING_DIMENSION,
                'image': {
                    'format': 'jpeg',
                    'source': {'bytes': input_image},
                },
            },
        }

        body = json.dumps(model_params)

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get('body').read())

        self.multimodal_embedding = response_body['embeddings'][0]['embedding']
        self.multimodal_embedding_model_id = model_id
        self.multimodal_embedding_cost = 0.0  # see Amazon Bedrock pricing page

        return

    def make_laplacian_image(self, image_file, ksize=3):
        """
        Creates an laplacian image from the given image file.
        Args:
           image_file: input image
           ksize: Aperture size used to compute the second-derivative filters. See (https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6) for more detail.
        """

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variant = cv2.Laplacian(gray, cv2.CV_64F, ksize).var()

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        lap_1 = cv2.Laplacian(image, cv2.CV_64F, ksize)
        lap_1_abs = np.uint(np.absolute(lap_1)) 

        # save the laplacian image to the folder for visualization
        image_path = Path(image_file).parent
        util.mkdir(os.path.join(image_path.parent, config["LAPLACIAN_PATH"]))
        laplacian_file = os.path.join(image_path.parent, config["LAPLACIAN_PATH"], os.path.basename(image_file))
        cv2.imwrite(laplacian_file, lap_1_abs)
        self.laplacian_file = laplacian_file

        return

    def display_laplacian(self, ksize=3):
        image = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        lap_1 = cv2.Laplacian(image, cv2.CV_64F, ksize)
        lap_1_abs = np.uint(np.absolute(lap_1)) 

        titles = ['Original Image', f"Laplacian derivative with ksize={ksize}"]
        images = [image, lap_1_abs]
        plt.figure(figsize=(13,5))
        for i in range(2):
            plt.subplot(1,3, i+1)
            plt.imshow((images[i]).astype(np.uint8), 'gray')
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
        return
