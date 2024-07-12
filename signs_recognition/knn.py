import argparse
import os 
from PIL import Image
import numpy as np
import csv

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-k', type=int, 
                        help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
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
    return np.array(Image.open(png_path)).astype(int).flatten()


# euclidean distance
def get_dist(png1, png2):
    diff = png1 - png2
    final = np.sqrt(np.sum(np.square(diff)))
    return final


# k - nearest neighbours algo
def knn(known_images, known_data, known_classes, new_image, k):

    # read image
    png = read_png(new_image)

    # get distance
    distances = np.array([get_dist(png, known_images[img]) for img in known_data.keys()])

    # get labels for each image
    labels = np.array(list(known_data.values()))

    # get first k sorted indicies of distance
    knn_indices = np.argpartition(distances, k, axis=0)[:k]

    # get most often label 
    knn_labels = np.array(list(known_classes[cls] for cls in labels[knn_indices]))
    predicted_label = list(known_classes.keys())[np.bincount(knn_labels).argmax()]

    return predicted_label


def main():

    parser = setup_arg_parser()
    args = parser.parse_args()

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    
    print(f"Running k-NN classifier with k={args.k}")
    
    # get all files of test_path without .dsv file
    test_data = os.listdir(args.test_path)
    if "truth.dsv" in test_data: test_data.remove("truth.dsv")

    # get dsv train data as dict
    dsv_file_train = read_dsv(args.train_path, "truth.dsv")

    # get all unique classes
    classes = np.unique(np.array(list(dsv_file_train.values())))
    classes = {cls: idx for idx, cls in enumerate(classes)}
    
    # Load all images from known_data into memory
    known_images = {img: read_png(f"{args.train_path}/{img}") for img in dsv_file_train.keys()}

    # get reults and write them down into .dsv file
    clasificated_data = {img : knn(known_images, dsv_file_train, classes, f"{args.test_path}/{img}", args.k) for img in test_data}
    write_dsv(args.o, clasificated_data)

    return 0

if __name__ == "__main__":
    main()
