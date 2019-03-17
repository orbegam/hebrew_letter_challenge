import numpy as np
from keras.models import load_model
from PIL import Image
import PIL.ImageOps
import argparse


def preprocess_img(image_path):
    im = PIL.ImageOps.invert(Image.open(image_path).convert('L'))
    
    im = im.crop((5, 5, 76, 76))
    
    cutoff = 15
    
    pixels = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if pixels[i,j] < cutoff:
                pixels[i,j] = 0
            elif pixels[i,j] >= cutoff:
                pixels[i,j] = 255
    
    cropped_im = im.crop(im.getbbox())
    padded_im = pad_to_size(cropped_im, 71, 71)
    
    return np.asarray(padded_im)

def pad_to_size(im, new_w, new_h):
    new_im = Image.new("L", (new_w, new_h))
    new_im.paste(im, ((new_w - im.size[0])//2,
                      (new_h - im.size[1])//2))
    return new_im

def predict(final_model, image_path, verbose=False):
    img_arr = preprocess_img(image_path)
    img_arr = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], 1).astype('float32')
    img_arr = img_arr / 255
    result = final_model.predict(img_arr)
    if verbose:
        print(result[0])
        
    return np.argmax(result)

def main(image_path, verbose):
    try:
        final_model = load_model('final_model_00175.h5')
    except:
        print("No model file found. Train the network once first, or upload your model as 'final_model.h5'")
        return
    
    print(predict(final_model, image_path, verbose))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--verbose', action='store_const', const=True, default=False)
    args = parser.parse_args()
    main(args.image_path, args.verbose)
