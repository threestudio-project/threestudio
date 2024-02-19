from gradio_client import Client
import shutil

client = Client("GRADIO_API_ENDPOINT")

# Define the input parameters for the API endpoint
polar_angle_vertical_rotation_in_degrees = 0
azimuth_angle_horizontal_rotation_in_degrees = 90
zoom_relative_distance_from_center = 0
input_image_of_single_object = "IMAGE_PATH"
preprocess_image_automatically_remove_background_and_recenter_object = True
diffusion_guidance_scale = 3
number_of_samples_to_generate = 3
number_of_diffusion_inference_steps = 75

# Call the predict method with the input parameters
result = client.predict(
    polar_angle_vertical_rotation_in_degrees,
    azimuth_angle_horizontal_rotation_in_degrees,
    zoom_relative_distance_from_center,
    input_image_of_single_object,
    preprocess_image_automatically_remove_background_and_recenter_object,
    diffusion_guidance_scale,
    number_of_samples_to_generate,
    number_of_diffusion_inference_steps,
    api_name="/partial_1"
)

for i, item in enumerate(result[-1]):
    generated_image_path = item['image']
    local_image_path = f"./saved_generated_image_{i + 1}.png"

    # Copy the image from the original location to the specified local path
    shutil.copy(generated_image_path, local_image_path)

    print(f"Generated image {i + 1} saved to {local_image_path}")