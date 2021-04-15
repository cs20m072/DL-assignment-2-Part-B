# deep-learning-assignment-2
Assignment 2 Part 2 of deep learning course

run the command in LINUX: train.py --epochs=20 --freeze_cnn=False --learning_rate=0.0001 --model_name=inception --optimizer=adam --weight_decay=0.0001
If run in windows due to naming convention issues it might throw some error, but works fine in LINUX.

#droput reference:
https://arxiv.org/pdf/1207.0580.pdf
http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf


# transfer learning reference
https://www.educative.io/edpresso/what-are-the-strategies-for-using-transfer-learning
https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751


It is assumed that  training data directory is stored inside /data directory 
training_directory = '/data/inaturalist_12K/train'
test_directory='data/inaturalist_12K/val'


#Code is executed on Linux platform , so on Windows it may give error on accessing the files


5)  train.py 
  	training models
	run: python train.py --epochs=20 --freeze_cnn=False --learning_rate=0.001 --model_name=inception --optimizer=sgd --weight_decay=0.01
	or run :  python train.py	
	num_linear is number of neurons in linear layer

CNN.py - contains code for nueral net in pytorch
dataLoader.py  : data loading using DataLoader class

pretrained_model.py  :  implementation of 6 pretrained models

results directory contains , images of guided backpropagation.