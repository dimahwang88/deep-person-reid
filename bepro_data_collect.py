import shutil
import os
import glob
import cv2

training_data_path = '/home/dmitriy.khvan/pytorch-beginner/training/'
validation_data_path = '/home/dmitriy.khvan/pytorch-beginner/validation/'

training_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/bounding_box_train'
query_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/query'
test_data_dest = '/home/dmitriy.khvan/deep-person-reid/reid-data/bepro/test'

train_dir_list = os.listdir(training_data_path)
valid_dir_list = os.listdir(validation_data_path)

def create_test_dataset(pid):
    pid = pid +1 

    count_q = 0
    count_t = 0

    for d in valid_dir_list:
        path2dir = os.path.join(validation_data_path, d, '1')
        
        if os.path.isdir(path2dir):
            pid_list = os.listdir(path2dir)

            for p in pid_list:
                path2pid = os.path.join(path2dir, p)
                
                pid = pid + 1
                for filename in os.listdir(path2pid):
                    path2file = os.path.join(path2pid, filename)

                    imgidx = int(filename.split('.')[0])

                    image = cv2.imread(path2file)
                    
                    new_filename = '%d_%s_%d.jpg' % (pid, '0', imgidx)

                    if imgidx == 0:
                        #copy to query
                        path2dest = os.path.join(query_data_dest, new_filename)
                        count_q = count_q + 1
                    else:
                        #copy to test
                        path2dest = os.path.join(test_data_dest, new_filename)
                        count_t = count_t + 1

                    print (path2dest)

                    image_res = cv2.resize(image, (64,128))
                    cv2.imwrite(path2dest, image_res)

    print (count_q)
    print (count_t)

def create_training_dataset():
    count = 0
    pid = 1

    for d in train_dir_list:

        folder_indices = [0, 1]
        
        for fi in folder_indices:
            path2dir = os.path.join(training_data_path, d, str(fi))
            
            if os.path.isdir(path2dir):
                pid_list = os.listdir(path2dir)

                for p in pid_list:
                    path2pid = os.path.join(path2dir, p)
                    
                    pid = pid + 1
                    for filename in os.listdir(path2pid):
                        path2file = os.path.join(path2pid, filename)

                        imgidx = int(filename.split('.')[0])
                        img_indices = [0, 5, 10, 15, 20, 25, 29]

                        if imgidx in img_indices:
                            image = cv2.imread(path2file)
                            if image.shape[0] < 110:
                                continue
                            new_filename = '%d_%s_%d.jpg' % (pid, '1', imgidx)
                            path2dest = os.path.join(training_data_dest, new_filename)
                            print (path2dest)
                            image_res = cv2.resize(image, (64,128))
                            cv2.imwrite(path2dest, image_res)
                            count = count + 1

    print (count)
    return pid


if __name__ == "__main__":
    pid = create_training_dataset()
    # create_test_dataset(pid)
