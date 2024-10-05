from immage import Immage

def main():
    """
    testing upscale
    """
    processor = Immage.open('pix/low_res_image.png')
    processor \
        .fractal_upscale(scale_factor=2, octaves=6) \
        .save('pix/upscaled_image.png')

if __name__ == "__main__":
    main()
