import numpy as np
import matplotlib.pyplot as plt

main_samples = np.load('C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\correct_numbered_samples.npy', allow_pickle=True)
manually_corrected_samples = np.load('C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\Manually_numbered_samples.npy', allow_pickle=True)
manually_corrected_samples_2 = np.load('C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\Manually_numbered_samples_2.npy', allow_pickle=True)
missed_again_samples = np.load('C:\\Users\\Sharjeel\\Desktop\\codes\\teeth_numbered_data\\number_missed_again_samples.npy', allow_pickle=True)

print('Main samples shape: ', main_samples.shape)
print('Manually numbered samples shape: ', manually_corrected_samples.shape)
print('Manually numbered samples 2 shape: ', manually_corrected_samples_2.shape)

# image, mask, instance, numbered, weight, _ = main_samples[114]

# plt.imshow(instance)
# plt.show()

# plt.imshow(numbered)
# plt.show()


missed_again = []


total_combined_data = []

for i, sample in enumerate(main_samples):
    image, mask, instance, numbered, weight, _ = sample
    
    flat_numbered = numbered.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    #print('Num vals: ', len(unnn))
    
    if len(unnn) == 33:
        total_combined_data.append([image, mask, instance, numbered, weight, 1])
    else:
        missed_again.append([image, mask, instance, weight, 1])
        print('First: ', len(unnn), ' at ', i)
    
for i, sample in enumerate(manually_corrected_samples):
    image, mask, instance, numbered, weight, _ = sample
    
    flat_numbered = numbered.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    #print('Num vals: ', len(unnn))
    
    if len(unnn) == 33:
        total_combined_data.append([image, mask, instance, numbered, weight, 1])
    else:
        missed_again.append([image, mask, instance, weight, 1])
        print('Second: ', len(unnn), ' at ', i)
    
for i, sample in enumerate(manually_corrected_samples_2):
    image, mask, instance, numbered, weight, _ = sample
    
    flat_numbered = numbered.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    #print('Num vals: ', len(unnn))
    
    if len(unnn) == 33:
        total_combined_data.append([image, mask, instance, numbered, weight, 1])
    else:
        missed_again.append([image, mask, instance, weight, 1])
        print('Third: ', len(unnn), ' at ', i)
    
for i, sample in enumerate(missed_again_samples):
    image, mask, instance, numbered, weight, _ = sample
    
    flat_numbered = numbered.reshape(65536, 3)
    unnn = np.unique(flat_numbered, axis = 0)
    #print('Num vals: ', len(unnn))
    
    if len(unnn) == 33:
        total_combined_data.append([image, mask, instance, numbered, weight, 1])
    else:
        missed_again.append([image, mask, instance, weight, 1])
        print('fourth: ', len(unnn), ' at ', i)

total_combined_data = np.array(total_combined_data)

print('Total data shape: ', total_combined_data.shape)

missed_again = np.array(missed_again)
#np.save('Missed_again.npy', missed_again)
