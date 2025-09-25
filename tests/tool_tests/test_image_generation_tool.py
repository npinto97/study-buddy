import sys
from gradio_client import Client


def test_image_generation():
    try:
        client = Client("black-forest-labs/FLUX.1-schnell")
        result = client.predict(
            prompt="A beautiful sunset over the mountains",
            seed=42,
            randomize_seed=False,
            width=512,
            height=512,
            num_inference_steps=5,
            api_name="/infer"
        )

        print("Image generated successfully:", result[0])
        print("Final seed used:", result[1])

    except Exception as e:
        print("Error during image generation:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    test_image_generation()
