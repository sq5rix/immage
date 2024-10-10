from PIL import Image, ImageDraw, ImageFont

def overlay_text_on_image(image_path, output_path, text, x, y, font_size, font_path=None):
    """
    Overlays text on an image at the specified position, font size, and font.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output image.
    - text: str, the text to overlay on the image.
    - x: int, the x-coordinate for the text position.
    - y: int, the y-coordinate for the text position.
    - font_size: int, the font size of the text.
    - font_path: str, path to the .ttf font file. If None, uses the default font.

    Returns:
    - None
    """
    # Open the image
    image = Image.open(image_path).convert('RGBA')

    # Create a new image for the text with transparent background
    txt_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))

    # Get a drawing context
    draw = ImageDraw.Draw(txt_layer)

    # Load the font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except IOError:
        print(f"Font file not found: {font_path}")
        font = ImageFont.load_default()

    # Draw the text onto the text layer
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Composite the text layer onto the original image
    combined = Image.alpha_composite(image, txt_layer)

    # Save the result
    combined.convert('RGB').save(output_path, "JPEG")

# Example usage of the function with font selection

# First, list available fonts to choose from
fonts = list_available_fonts()

# Select a font from the list
font_path = fonts[0]  # You can choose any font from the list

overlay_text_on_image(
    image_path='input.jpg',
    output_path='output.jpg',
    text='Hello, World!',
    x=50,
    y=100,
    font_size=36,
    font_path=font_path
)
