import os
#---------------------------------------works good
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import numpy as np
import torch.utils.data as utils


num_epochs = 100
batch_size = 64
learning_rate = 1e-3
img_transform = transforms.Compose([transforms.ToTensor()])    #-------------------------------------we do not want to normalize
training_loss = 0
testing_loss = 0


myfilename = "../data/data_4_10_20_new.txt"
f=open(myfilename,'r')
line = f.readline()
line_cnt = 1
states = []
temp_line=[]
temp_line_for_1_state = []
updating_string = []
count = 0

#---------------------------for ninja technique, we need this demo string
demo_string = "a"
updating_string.append(demo_string)


while line:

    demo_string = demo_string + line
    if ("]" in line):

        demo_string = demo_string.replace("[","") #-----------
        demo_string = demo_string.replace("\n","")
        demo_string = demo_string.replace("]","")
        demo_string = demo_string.replace("a","")

        updating_string[count] = demo_string.split()

        count = count + 1
        demo_string = "a"
        updating_string.append(demo_string)

    line = f.readline()

del updating_string[-1]     #removing the "a" added at the end of the loop above
states = updating_string
length_of_total_states = len(states)
print("The total number of states is ",length_of_total_states)
training_states = states[:int(length_of_total_states * 0.9)]
testing_states = states[int(length_of_total_states * 0.9):]

#------------------------------------Check if the data is inconsistent
for i in range(len(training_states)):

    if(len(training_states[i]) != 40):
        #print(len(states[i]))
        print("Training data error: Not all joint states have the same number of elements. Position ", i," has inconsistent data.")

for i in range(len(testing_states)):

    if(len(testing_states[i]) != 40):
        #print(len(states[i]))
        print("Testing data error: Not all joint states have the same number of elements. Position ", i," has inconsistent data.")

#--------------------------------------convert a list of lists into a list of numpy arrays to build a custom dataset for pytorch


def states_list(states_input):
    states_list_x = []
    states_list_y = []

    for item in range(len(states_input)):

        states_list_x.append(np.asarray(states_input[item], dtype=np.float32))

    for item in range(len(states_input)):#-------------------------------------------------works as lable, need to provide to make a data set
        states_list_y.append(np.asarray([0], dtype=np.float32))
    return states_list_x, states_list_y

training_states_list_x, training_states_list_y = states_list(training_states)
testing_states_list_x, testing_states_list_y = states_list(testing_states)
# print(length_of_total_states)
# print(len(training_states_list_x))
# print(len(testing_states_list_x))
training_tensor_x = torch.stack([torch.Tensor(i) for i in training_states_list_x]) # transform to torch tensors
training_tensor_y = torch.stack([torch.Tensor(i) for i in training_states_list_y])
testing_tensor_x = torch.stack([torch.Tensor(i) for i in testing_states_list_x]) # transform to torch tensors
testing_tensor_y = torch.stack([torch.Tensor(i) for i in testing_states_list_y]) # transform to torch tensors
#tensor_x is a list of lists.
#tensor_x[0] has a length of 40

custom_training_dataset = utils.TensorDataset(training_tensor_x, training_tensor_y) # create your datset
custom_training_dataloader = utils.DataLoader(dataset=custom_training_dataset,batch_size = batch_size,shuffle = True) # create your dataloader

custom_testing_dataset = utils.TensorDataset(testing_tensor_x, testing_tensor_y) # create your datset
custom_testing_dataloader = utils.DataLoader(dataset=custom_testing_dataset,batch_size = batch_size,shuffle = True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(len(training_states[0]), 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32),
                                     nn.LeakyReLU(),
                                     # nn.Linear(32, 16),
                                     # nn.LeakyReLU(),
                                     # nn.Linear(16, 10),
                                     # nn.LeakyReLU()

                                     #------------------------------------------Encoder ends here

                                     )

        # self.encoder = nn.Sequential(nn.Linear(len(states[0]),50),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(50, 30),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(30, 20),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(20, 10),
        #                              nn.LeakyReLU(),
        #                              # nn.Linear(64, 32),
        #                              # nn.LeakyReLU()  # ------------------------------------------Encoder ends here
        #
        #                              )

        self.decoder = nn.Sequential(#nn.Linear(10, 16),
                                     #nn.LeakyReLU(),
                                     #nn.Linear(16, 32),
                                     #nn.LeakyReLU(),
                                     nn.Linear(32, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, len(training_states[0]))#------------------------------------------Decoder ends here


        )
        # self.decoder = nn.Sequential(nn.Linear(10, 20),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(20, 30),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(30, 50),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(50, len(states[0])),
        #                              nn.LeakyReLU(),
        #                              # nn.Linear(512, len(states[0]))
        #                              # # ------------------------------------------Decoder ends here
        #
        #                              )

    def forward(self, x):

        x = self.encoder(x)
                                #------------------------may be we can put tolerance stuff on latent dimensions
        x = self.decoder(x)
        return x


model = AutoEncoder().cuda()
criterion = nn.L1Loss()#--------------Which loss to use, L1 or MSE, L1 loss makes more sense because we do not have outliers in our data
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-5)

#print(len(custom_dataloader.dataset))#-----------------102464
for epoch in range(num_epochs):
    for index,data in enumerate(custom_training_dataloader):
        features_of_a_training_batch, training_labels = data
        #print(len(features)) gives 64 which is the batch size
        #print(len(features[0])) gives 40 which is the dimension of the state space
        features_of_a_training_batch = Variable(features_of_a_training_batch).cuda()
        # ----------------------------------------------forward
        training_output = model(features_of_a_training_batch)
        #check how good the model was at the end of every epoch
        if(index == len(custom_training_dataloader)-1):
            print("For Training Data: \n")
            print("In the original data, 1st element of the last batch: ",features_of_a_training_batch[1])
            print("Predicted Values to compare: ",training_output[1])


        training_loss = criterion(training_output, features_of_a_training_batch)#------------------take care of the order of the output and the sample

        # ----------------------------------------------backward
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
    # ===================log========================

    #-----------------------------Check how good you are doing after every epoch with testing data
    for index,data in enumerate(custom_testing_dataloader):
        features_of_a_testing_batch, labels = data

        features_of_a_testing_batch = Variable(features_of_a_testing_batch).cuda()
        # ----------------------------------------------forward
        testing_output = model(features_of_a_testing_batch)
        if(index == len(custom_testing_dataloader)-1):
            print("For Testing Data:\n")
            print("In the original data, 1st element of the last batch: ",features_of_a_testing_batch[1])
            print("Predicted Values to compare: ",testing_output[1])


        testing_loss = criterion(testing_output, features_of_a_testing_batch)#------------------take care of the order of the output and the sample




    #Testing Code here
    #-------------------------------------------------------------------------------
    print("epoch [{}/{}], ".format(epoch+1,num_epochs))
    print('Training loss:{:.4f}'.format(training_loss.data.item()))
    print('Testing loss:{:.4f}'.format(testing_loss.data.item()))
#torch.save(model.state_dict(), './autoencoder.pth')