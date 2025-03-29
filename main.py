import gradio as gr
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


PROJECT_ID = <<YOUR_PROJECT_ID>>
LOCATION = <<YOUR_LOCATION>>

os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = "False"


vertexai.init(project=PROJECT_ID, location=LOCATION)


def generate_image(api_key, model_choice="gemini", prompt=None, input_image=None, edit_prompt=None):
    
    try:
        
        if model_choice == "gemini":
            model = "gemini-2.0-flash-exp-image-generation"
            client = genai.Client(api_key=api_key)
            if input_image is None:
                contents = prompt
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=['Text', 'Image']
                        )
                    )
                except Exception as e:
                    return f"Error generating image: {e}", None

                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        image = Image.open(BytesIO((part.inline_data.data)))
                        image.save('generated_image.png')
            else:
                image = Image.open(input_image.name)
                text_input = (edit_prompt,)
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=[text_input, image],
                        config=types.GenerateContentConfig(
                            response_modalities=['Text', 'Image']
                        )
                    )
                except Exception as e:
                    return f"Error editing image: {e}", None

                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        image.save('generated_image.png')
                        
        elif model_choice == "imagen-fast":
            model = "imagen-3.0-fast-generate-001"
            if input_image is not None:
                raise Exception("Image editing is not supported for this model.")
            generation_model_fast = ImageGenerationModel.from_pretrained(model)

            image = generation_model_fast.generate_images(
                        prompt=prompt,
                        number_of_images=1,
                        aspect_ratio="3:4",
                        safety_filter_level="block_some",
                        add_watermark=True,
                    )
            image[-1].save("generated_image.png")


        elif model_choice == "imagen":
            model = "imagen-3.0-generate-001"
            if input_image is not None:
                raise Exception("Image editing is not supported for this model.")
            generation_model = ImageGenerationModel.from_pretrained(model)

            image = generation_model.generate_images(
                        prompt=prompt,
                        number_of_images=1,
                        aspect_ratio="3:4",
                        safety_filter_level="block_some",
                        add_watermark=True,
                    )
            image[-1].save("generated_image.png")

        return "Image generated successfully!", 'generated_image.png'

    except Exception as e:
        return f"Error initializing client: {e}", None

    
    return "No image generated.", None

with gr.Blocks() as demo:
    gr.Markdown("# Gemini Image Generation")
    api_key_input = gr.Textbox(label="Enter your API Key", type="password")
    model_choice = gr.Radio(
        ["gemini", "imagen", "imagen-fast"],
        label="Choose a model",
    )
    prompt_input = gr.Textbox(label="Enter your prompt")
    image_input = gr.File(label="Upload an image to edit", file_types=["image"])
    edit_prompt_input = gr.Textbox(label="Enter edit prompt")
    generate_button = gr.Button("Generate Image")
    output_text = gr.Textbox(label="Output Text")
    output_image = gr.Image(label="Generated Image")

    generate_button.click(generate_image, inputs=[api_key_input, model_choice, prompt_input, image_input, edit_prompt_input], outputs=[output_text, output_image])

demo.launch()
