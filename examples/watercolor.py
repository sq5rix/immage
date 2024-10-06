from immage import Immage

def main():
    # Generate watercolor paper texture
    texture_processor = Immage.generate_watercolor_paper_texture()
    texture_image = texture_processor.image
    texture_image.save('pix/texture.png')

    # Open the base image
    processor = Immage.open('pix/low_res_image.png')

    # Apply the texture to the base image
    processor \
        .apply_texture(texture_image) \
        .save('pix/image_with_texture.png')

if __name__ == "__main__":
    main()
