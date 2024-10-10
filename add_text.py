from PIL import Image, ImageDraw, ImageFont

def overlay_text_on_image(image_path, output_path, text, x, y, font_size):
    """
    Overlays text on an image at the specified position and font size.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output image.
    - text: str, the text to overlay on the image.
    - x: int, the x-coordinate for the text position.
    - y: int, the y-coordinate for the text position.
    - font_size: int, the font size of the text.

    Returns:
    - None
    """
    # Open the image
    image = Image.open(image_path).convert('RGBA')

    # Create a new image for the text with transparent background
    txt_layer = Image.new('RGBA', image.size, (255,255,255,0))

    # Get a drawing context
    draw = ImageDraw.Draw(txt_layer)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default font if arial.ttf is not found
        font = ImageFont.load_default()

    # Draw the text onto the text layer
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Composite the text layer onto the original image
    combined = Image.alpha_composite(image, txt_layer)

    # Save the result
    combined.convert('RGB').save(output_path, "JPEG")
