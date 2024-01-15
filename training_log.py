# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:14:53 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Printing the training history
"""
import re


# Define the path to your text file
file_path = 'pix2pix_training_log_202401031030.txt'

# Initialize an empty dictionary to store the values
data_dict = {'Epoch': [], 'Iteration': [], 'Dis_real_loss': [], 'Dis_real_accuracy': [],
             'Dis_fake_loss': [], 'Dis_fake_accuracy': [], 'Gen_loss': []}

list_ptrn = r'loss=([\d.]+),accuracy=([\d.]+)'


# Read the text file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Split the line into relevant parts
        parts = line.split('>')

        # Extract values and update the dictionary
        epoch = int(parts[1].split('/')[0].strip())
        
        iteration = int(parts[3].split('>')[0].strip())

        match_real = re.search(list_ptrn, parts[4].strip())    
        dis_real_loss, dis_real_accuracy = float(match_real.group(1)), float(match_real.group(2))
        
        match_fake = re.search(list_ptrn, parts[5].strip())
        dis_fake_loss, dis_fake_accuracy = float(match_fake.group(1)), float(match_fake.group(2))
        
        gen_loss = float(parts[6].split('[')[1][:-2])

        data_dict['Epoch'].append(epoch)
        data_dict['Iteration'].append(iteration)
        data_dict['Dis_real_loss'].append(dis_real_loss)
        data_dict['Dis_real_accuracy'].append(dis_real_accuracy)
        data_dict['Dis_fake_loss'].append(dis_fake_loss)
        data_dict['Dis_fake_accuracy'].append(dis_fake_accuracy)
        data_dict['Gen_loss'].append(gen_loss)
       
# Print the resulting dictionary
print(data_dict)