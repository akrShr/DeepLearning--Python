import numpy as np
import matplotlib.pyplot as plt 
import json
import random
import math

class ImageGenerator:
    
    def __init__(self, file_path, json_path, batch_size, image_size, rotation = False, 
                 mirroring = False, shuffle = False):
        self.file_path = file_path
        self.json_path = json_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.labels = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 
                             4 : 'deer', 5 : 'dog', 6 : 'frog', 7 : 'horse', 8 : 'ship', 
                             9 : 'truck' }
        self.images_info=[]
        self.read_jsonfile()

        """
        As per document we have to mirror images if mirroring is set true
        Mirroring to be done horizontally, vertically and diagonally for a batch. We use list called 'mirroring_option' for it.
        Since mirroring to be on random images we generate random number between 0 and batch_size and put in a set called 'num_mirror'.
        The set represents those images on which mirroring will happen. 
        Since mirroring is to be three types, set contains 3 random numbers as image number.
        We sort set so the sequence is in ascending order when we pop image number from sequence.
        """
        if self.mirroring :
            self.mirroring_option = ["horizontal","vertical","double"]
            self.num_mirror = set()
            length_num = len(self.num_mirror)
            while length_num < len(self.mirroring_option):
                self.num_mirror.add(random.randint(0, self.batch_size))
                length_num =len(self.num_mirror)
            self.num_mirror = sorted(self.num_mirror, reverse=True)

        """
        As per document we have to rotate images if rotation is set true
        Rotation to be done 90, 180 and 270 for a batch. We use list called 'rotation_option' for it.
        Since rotation to be on random images we generate random number between 0 and batch_size and put in a set called 'num_rotation'.
        The set represents those images on which rotation will happen. 
        Since rotation is to be three types, set contains 3 random numbers as image number.
        We sort set so the sequence is in ascending order when we pop image number from sequence.
        """
        if self.rotation :
            self.rotation_option = [90, 180, 270]
            self.num_rotation = set()
            length_rot = len(self.num_rotation)
            while length_rot < len(self.rotation_option):
                self.num_rotation.add(random.randint(0, self.batch_size))
                length_rot = len(self.num_rotation)
            self.num_rotation = sorted(self.num_rotation, reverse=True)




    """
    read json file and stores the key(image file identifier) with its respective image label into a tuple.
    The tuple is added to list 'images_info' to be used while loading image file in next().
    """
    def read_jsonfile(self):
        with open(self.json_path) as json_file:  
            data = json.load(json_file)
        for key in data.keys():
            #img_label = self.class_name(data.get(key))
            img_label = data.get(key)
            self.images_info.append((key, img_label))
        """
        If shuffling of data is true we shuffle the list items so the batches has random entry of images
        """
        if self.shuffle :
            random.shuffle(self.images_info)
        
    
    def next(self): 
        start= 0
        end=start+self.batch_size
        labels=np.zeros(self.batch_size)
        batch=np.zeros((self.batch_size,self.image_size[0],self.image_size[1],self.image_size[2]))
        
        if self.mirroring:
            mir_num = self.num_mirror.pop()
        if self.rotation:
            rot_num = self.num_rotation.pop()

        while start < end:
            key,img_label = self.images_info.pop(0)
            img_path = self.file_path + '/' + key + '.npy'
            img_file = np.load(img_path)

            x,y,z=img_file.shape
            if self.image_size[0]>x or self.image_size[1]>y or self.image_size[2]>z :
                img_file = np.resize(img_file,(self.image_size[0],self.image_size[1],self.image_size[2]))

            if self.mirroring and start==mir_num :
                mir_opt=self.mirroring_option.pop()
                if mir_opt == "horizontal":
                    img_file = np.flip(img_file, 1)
                elif mir_opt == "vertical":
                    img_file = np.flip(img_file, 0)
                else:
                    img_file = np.flip(img_file)
                if len(self.num_mirror)>0:
                    mir_num = self.num_mirror.pop()

            if self.rotation and start==rot_num :
                rot_opt=self.rotation_option.pop()
                if rot_opt == 90:
                    img_file = np.rot90(img_file, k=1, axes=(0,1))
                elif rot_opt == 180:
                    img_file = np.rot90(img_file, k=2, axes=(0,1))
                else:
                    img_file = np.rot90(img_file, k=3, axes=(0,1))
                if len(self.num_rotation)>0:
                    rot_num = self.num_rotation.pop()

            batch[start:] = img_file
            labels[start]= img_label
            self.images_info.append((key,img_label))
            start += 1

        batch_info = (batch, labels)
        return batch_info
        

    def class_name(self, int_label):
        return self.labels.get(int_label)

    def show(self):
        batch=self.next()
        images=batch[0]
        labels_num=batch[1]
        labels=[]
        for i in labels_num:    
            labels.append(self.class_name(i))

        if self.batch_size % 2 == 0:
            row = self.batch_size // 2
        else:
            row = int(math.ceil(self.batch_size / 2))
            
        fig, axs = plt.subplots(row, 2,figsize=(2,6))
        
        for i in range(row):
            axs[i, 0].imshow((images[(i*2)]).astype(np.uint8), interpolation="none") 
            axs[i, 0].set_title(labels[i*2])
            axs[i, 0].axis('off')
    
            axs[i, 1].imshow((images[((i*2)+1)]).astype(np.uint8), interpolation="none")    
            axs[i, 1].set_title(labels[(i*2)+1])
            axs[i, 1].axis('off')
    
        plt.tight_layout()
        




