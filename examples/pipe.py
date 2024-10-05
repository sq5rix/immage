from immage import Immage

def adjust_brightness(proc):
    return proc.enhance_brightness(1.2)

def main():
    processor = Immage.open('input_image.jpg')
    processor = processor | adjust_brightness | (lambda p: p.sharpen())
    processor.save('output_pipe.jpg')

if __name__ == "__main__":
    main()
