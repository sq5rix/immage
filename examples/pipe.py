from immage import Immage

def adjust_brightness(proc):
    return proc.enhance_brightness(1.2)

def main():
    processor = Immage.open('pix/low_res_image.png')
    processor = processor | adjust_brightness | (lambda p: p.sharpen())
    processor.save('pix/output_pipe.png')

if __name__ == "__main__":
    main()
