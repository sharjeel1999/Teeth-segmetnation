import numpy as np
import matplotlib.pyplot as plt
import cv2


number_colour_code = [
        [0, 204, 204], # 18
        [255, 153, 51], # 17
        [0, 204, 0], # 16
        [102, 255, 255], # 15
        [102, 255, 102], # 14
        [178, 102, 255], # 13
        [255, 255, 51], # 12
        [0, 0, 102], # 11
        [255, 51, 51], # 21
        [153, 255, 51], # 22
        [102, 178, 255], # 23
        [204, 0, 0], # 24
        [0, 204, 102], # 25
        [102, 204, 0], # 26
        [204, 102, 0], # 27
        [204, 204, 0], # 28
        [102, 102, 255], # 38
        [102, 0, 102], # 37
        [0, 102, 102], # 36
        [0, 51, 102], # 35
        [0, 102, 0], # 34
        [102, 102, 0], # 33
        [102, 51, 0], # 32
        [0, 0, 51], # 31
        [51, 51, 0], # 41
        [51, 0, 0], # 42
        [0, 51, 51], # 43
        [51, 25, 0], # 44
        [0, 51, 0], # 45
        [51, 0, 51], # 46
        [102, 0, 0], # 47
        [102, 0, 51]  # 48
    ]


number_colour_code = np.array(number_colour_code)

data_path = 'C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\correct_samples.npy'
data = np.load(data_path, allow_pickle=True)

print('data shape: ', data.shape)

automatic_numbered_data = []
samples_for_manual_numbering = []

for sample in data:
    image, mask, instance, weights, _ = sample
    
    # fig, ax = plt.subplots(1)
    # ax.imshow(image)
    # ax.text(5, 5, 'Image', bbox={'facecolor': 'white', 'pad': 10})
    # plt.show()
    
    # fig, ax = plt.subplots(1)
    # ax.imshow(instance)
    # ax.text(5, 5, 'Instance', bbox={'facecolor': 'white', 'pad': 10})
    # plt.show()
    
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(instance)
    # plt.show()
    
    flat_instance = instance.reshape(65536, 3)
    un = np.unique(flat_instance, axis = 0)
    
    if len(un) == 33:
        print('length unique: ', len(un))
        
        up_part = instance[0:128, :, :]
        down_part = instance[128:256, :, :]
        down_part = np.flip(down_part, 0)
        conc_part = np.concatenate((up_part, down_part), axis = 1)
        
        teeths_in_order = []

        
        for j in range(512):
            for i in range(50, 100):
                combined_copy = np.copy(conc_part)
                pix = combined_copy[i, j, :]
                
                rr, gg, bb = pix
                
                if rr != 0 or gg != 0 or bb != 0:
                    teeths_in_order.append([rr, gg, bb])
                
        teeths_in_order = np.array(teeths_in_order)
        unique_teeths_sorted = np.unique(teeths_in_order, axis = 0, return_index = True)[1]
        unique_teeths_order = [teeths_in_order[ind, :] for ind in sorted(unique_teeths_sorted)]
        
        numbered_mask = np.zeros((256, 256, 3))
    print('len teeths: ', len(unique_teeths_order))
    
    
    for i, teeth in enumerate(unique_teeths_order):
        r, g, b = teeth
 
        instance_copy = np.copy(instance)
        
        lower = np.array([r-1, g-1, b-1])
        upper = np.array([r+1, g+1, b+1])
        masked = cv2.inRange(instance_copy, lower, upper)

        numbered_mask_copy = np.copy(numbered_mask)
        

        numbered_mask_r = np.where(masked == 255, number_colour_code[i, 0], 0)
        numbered_mask_g = np.where(masked == 255, number_colour_code[i, 1], 0)
        numbered_mask_b = np.where(masked == 255, number_colour_code[i, 2], 0)
        

        numbered_mask_r = np.expand_dims(numbered_mask_r, axis = 2)
        numbered_mask_g = np.expand_dims(numbered_mask_g, axis = 2)
        numbered_mask_b = np.expand_dims(numbered_mask_b, axis = 2)
        
        if i == 0:
            numbered_mask = np.concatenate((numbered_mask_r, numbered_mask_g, numbered_mask_b), axis = 2)
        else:
            numbered_mask_c = np.concatenate((numbered_mask_r, numbered_mask_g, numbered_mask_b), axis = 2)
            numbered_mask = numbered_mask + numbered_mask_c
            
    
    # fig, ax = plt.subplots(1)
    # ax.imshow(numbered_mask)
    # ax.text(5, 5, 'Numbered mask', bbox={'facecolor': 'white', 'pad': 10})
    # plt.show()
    
    # plt.imshow(numbered_mask)
    # plt.show()
    
    flat_numbered = numbered_mask.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    #print('vals: ', unnn)
    if len(unnn) == 33:
        print('unique numbered: ', len(unnn))
        automatic_numbered_data.append([image, mask, instance, numbered_mask, weights, 1])
    else:
        samples_for_manual_numbering.append([image, mask, instance, weights, 1])
        print('----- unique numbered: ', len(unnn))
        fig, ax = plt.subplots(1)
        ax.imshow(conc_part)
        ax.text(5, 5, 'Conc part', bbox={'facecolor': 'white', 'pad': 10})
        plt.show()

automatic_numbered_data = np.array(automatic_numbered_data)
print('Automaticaly numbered data: ', len(automatic_numbered_data))

samples_for_manual_numbering = np.array(samples_for_manual_numbering)
print('samples to be numbered manualy: ', len(samples_for_manual_numbering))

np.save('Automatic_numbered_data.npy', automatic_numbered_data)
np.save('Samples_for_manual_numbering.npy', samples_for_manual_numbering)

