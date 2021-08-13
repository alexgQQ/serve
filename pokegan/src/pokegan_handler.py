import io
import os
import tempfile
import logging

from uuid import uuid4

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import moviepy.editor as mp

from google.cloud import storage
from PIL import Image

from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)


UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET")
VIDEO_TEMPLATE_LOCATION = os.getenv("VIDEO_TEMPLATE_LOCATION")
AUDIO_LOCATION = os.getenv("AUDIO_LOCATION")


def tensor_to_array(tensor):
    image_numpy = tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def auto_canny(image, sigma=0.33):
    """
    Simple canny edge detector with normalization
    """
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)


def get_outline_mask(outline):
    """
    Take a greyscale image array of a black on white object outline to create
    a output array as a mask of the object filled, the mask will be white on black
    """
    outline = cv2.bitwise_not(outline)
    contours, _ = cv2.findContours(outline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(outline, contours, 0, 255, cv2.FILLED)


def edit_video(video_template, outline_image, generated_image, audio_template):
    """
    Creates the video file for the "who's that pokemon"
    images should be 4 channel numpy arrays, video template should be the file location
    for the base pokemon tempalte, returns a object for video saving
    """
    video = mp.VideoFileClip(video_template).set_duration(17)

    sillouete_image = (
        mp.ImageClip(outline_image)
        .set_duration(9)
        .set_start(1)
        .resize(height=256)
        .set_position((120, 80))
    )

    full_image = (
        mp.ImageClip(generated_image)
        .set_duration(7)
        .set_start(10)
        .resize(height=256)
        .set_position((120, 80))
    )

    final_video = mp.CompositeVideoClip([video, sillouete_image, full_image])
    original_audio = final_video.audio
    added_audio = mp.AudioFileClip(audio_template)
    new_audio = mp.CompositeAudioClip([original_audio, added_audio.set_start(10)])
    final_video = final_video.set_audio(new_audio)
    return final_video


def blob_storage(filename):
    """Create a blob for file storage upload"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(UPLOAD_BUCKET)
    return bucket.blob(os.path.join("public", filename))


class Pix2PixHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    outline_image = None
    image_processing = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        # Doing this model evaluation breaks the results from inference
        # self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):

        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image)).convert("RGBA").resize((256, 256))

            # Save original png with background opacity, this is needed for image postprocessing
            self.outline_image = image.copy()

            # Set the 0 opacity sections to white for inference prep
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # 3 is the alpha channel

            # Normalize and turn image data into a correctly sized tensor
            image = np.array(rgb_image)
            # image = auto_canny(image)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = self.image_processing(image)
            torch.reshape(image, (1, 3, 256, 256))
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        return self.model(data, *args, **kwargs)

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        outputs = []

        for result in inference_output:
            result_image = tensor_to_array(result.data)

            # Get contours to create a silhouette for the video
            outline = np.array(self.outline_image)
            outline[outline[:, :, 3] == 0] = 255
            outline_mask = get_outline_mask(cv2.cvtColor(outline, cv2.COLOR_RGBA2GRAY))
            outline = cv2.cvtColor(cv2.bitwise_not(outline_mask), cv2.COLOR_GRAY2RGBA)

            # Mask out the sillhouette and generated image so their background is transparent
            result_image = Image.fromarray(result_image).convert("RGBA")
            result_image = np.array(result_image)
            result_image = cv2.bitwise_and(
                result_image, result_image, mask=outline_mask
            )
            outline = cv2.bitwise_and(outline, outline, mask=outline_mask)

            # Apply clustering and a blur to the generated image. This limits the colors
            # and helps to blend them
            Z = result_image.reshape((-1, 4))
            Z = np.float32(Z)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 10
            ret, label, center = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            center = np.uint8(center)
            res = center[label.flatten()]
            result_image = res.reshape((result_image.shape))
            result_image = cv2.GaussianBlur(result_image, (5, 5), 0)

            # Overlay the original image to the generated image
            result_image = Image.fromarray(result_image.astype("uint8"), "RGBA")
            result_image.paste(self.outline_image, (0, 0), mask=self.outline_image)
            result_image = np.array(result_image)

            final = edit_video(
                VIDEO_TEMPLATE_LOCATION, outline, result_image, AUDIO_LOCATION
            )

            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = f"{uuid4()}.mp4"
                fullpath = os.path.join(tmpdirname, filename)

                final.write_videofile(
                    fullpath, logger=None, write_logfile=False, audio=True
                )

                blob = blob_storage(filename)
                blob.upload_from_filename(fullpath)

            outputs.append({"filename": f"{filename}"})
        return outputs
