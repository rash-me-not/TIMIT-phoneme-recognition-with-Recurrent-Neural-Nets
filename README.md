# TIMIT-phoneme-recognition-with-Recurrent-Neural-Nets
In this project, we are experimenting with the Recurrent Neural networks on TIMIT Data Corpus using a variety of architectures in order to predict the phonemes. 

Due to the huge size of the data, we are training the model on minibatches of data (as per the specified batch size in the model), and padding every batch with series of zeros to match the longest time step in the sequence. 

We are using K Fold cross validation to generalize the performance of the dataset against each fold independently. In this lab we could not experiment beyond 2 and 3 fold models due to hardware and computational issues. Each recurrent neural network is preceded with a masking layer, followed by recursive layers like LSTM , GRU or CuDNNLSTM and ended with a Dense softmax layer output for every phoneme

