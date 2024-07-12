import argparse
import os 
from PIL import Image
import numpy as np

normal_height = 28
normal_width = 28
COLORS = 16

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser


# return dictionary generated from dsv file
def read_dsv(directory, dsv_file):
    ret_dict = {}
    with open(f"{directory}/{dsv_file}", 'r') as file:
        for line in file:
            line = line.strip().split(':')
            ret_dict[line[0]] = line[1]
    return ret_dict


# write in dsv from dictionary got from another func
def write_dsv(new_file, clasificated):
    with open(new_file, 'w') as f:
        for img, classicator in clasificated.items():
            f.write(f"{img}:{classicator}\n")    


#return numpy array representation of given png(as a path)
def read_png(png_path):
    return np.array(Image.open(png_path)).astype(int).flatten() // COLORS #  make image less bright


# return
#  - prob of class in samples
#  - prob of definite color in pixel in image
def train(train_path):

    # convert dsv to dict {img: class}
    dsv_train_data = read_dsv(train_path, "truth.dsv")

    class_count = {} # dict {class: count}
    class_image = {} # dict {class: example image(as np.array)}

    # filling dicts
    for img, cls in dsv_train_data.items():
        # create pair class: pixels (as ndarray)
        if cls not in class_count:
            class_count[cls] = 0
            class_image[cls] = np.zeros((COLORS, normal_height * normal_width))
        
        # update num of class members
        class_count[cls] += 1
        
        # update num of definite pixels in class type image
        png = read_png(f"{train_path}/{img}")
        for idx, color in enumerate(png):
            class_image[cls][color][idx] += 1

    # num of all samples
    num_of_img = np.sum(np.array(list(class_count.values())))

    # normalising (getting probs)
    probs_of_classes = {}
    probs_of_image = {}

    for cls in class_count.keys():

        # prob of class in samples
        probs_of_classes[cls] = class_count[cls] / num_of_img
        
        # prob of each color in pixel in class example of image
        probs_of_image[cls] = (class_image[cls] + 1) / (class_count[cls] + COLORS) # laplacian correction

    return probs_of_classes, probs_of_image


# prob_class - dict {class: prob of class} 
# prob_pixel - dict {class: type prob image}
# new_image - ndarray
def naive_bayes(prob_class, prob_image, new_image):

    # get png as np_array
    new_png = read_png(new_image)
    
    # set max
    max_value = float("-inf")
    max_class = ''

    # get class with max value (corelation)
    for cls, prob_of_class in prob_class.items():
        value = np.log(prob_of_class) + np.sum( np.array( list(np.log(prob_image[cls][color][idx]) for idx, color in enumerate(new_png) )) )
        if value > max_value:
            max_value = value
            max_class = cls

    return max_class


def main():

    parser = setup_arg_parser()
    args = parser.parse_args()

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    print("Running Naive Bayes classifier")
    
    # get all files of test_path without .dsv file 
    test_data = os.listdir(args.test_path)
    if "truth.dsv" in test_data: test_data.remove("truth.dsv")

    # train classificator
    prob_class, prob_class_img = train(args.train_path)

    # create dict with elements {test_img : class}
    clasificated_data = {img : naive_bayes(prob_class, prob_class_img, f"{args.test_path}/{img}") for img in test_data}
    write_dsv(args.o, clasificated_data)

    return 0
        
        
if __name__ == "__main__":
    main()
    
