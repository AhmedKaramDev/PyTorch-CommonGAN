import argparse
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data", help="input data")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--output", type=str, default="reshaped_images", help="size of each image dimension")

opt = parser.parse_args()
for image_type in ["jpg", "png", "jpeg"]:
    images_list =glob.glob("{}/*.{}".format(opt.data, image_type))

    i = 0
    for img in images_list:
        try:
            image = cv2.imread(img)
            re_img = cv2.resize(image, (512, 512))
            cv2.imwrite("{}/{}/{}.{}".format(opt.data, opt.output, i, image_type), re_img)
            i+=1

        except:
            print("missed image")