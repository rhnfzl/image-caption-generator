# Image Caption Generation

#### Part of Assignment of Deep Learning Course (2IMM10) at TU/e 

## Objective

Construct a Long-Short-Term-Memory (LSTM) network which takes an image representation obtained from a convolutional neural network (ConvNet) as input, and produces a caption describing the image. This task is based on [Show and Tell: A Neural Image Caption Generator(/reading/show_and_tell_a_neural_image_caption_generator.pdf). 


### Data: Flickr8k

The Flickr8k dataset contains 8091 RGB images and 5 human-provided textual descriptions for each image (captions). For this task, the dataset has already been [preprocessed](https://surfdrive.surf.nl/files/index.php/s/kOIDM5tQPzv6IID).

* All images have been rescaled to 128 × 128 RGB.
* Punctuation and special tokens have been removed from the captions.
* Words which occur less than 5 times in the whole corpus have been removed.
* All words have been converted to lower case


### Task 1: Generate Neural Codes

Generate ConvNet representations (neural codes) for all images in the Flickr8k dataset. To this end, use the last convolutional layer (’Conv_1’) of MobileNetV2 pretrained on Imagenet (The pretrained MobileNetV2 can conveniently be downloaded within Keras). This layer contains 4 × 4 × 1280 features, yielding codes of length 20480.

### Task 2: Analyze Captions

Retrieve some information from the captions. In particular:

* Find and report the maximal caption length.
* Construct a collection of all words occurring in the captions and count their occurrences. Report the 10 most frequent words. Do you note a bias in the dataset?
* Include the special word ’_’ (the stop word, signaling the end of the captions) in the collection of words.
* How many unique words are there in the corpus, including ’_’?
* Construct a mapping (dictionary) from words to integers as follows:
	* Stop word ’_’ → 0
	* Most frequent word → 1
	* Second most frequent word → 2
	* …
* Construct an inverse mapping (dictionary), which maps integers back to words.

### Task 3: Train Model

Implement the model from the paper. In particular:

* Embed both the image codes and each word in a 512 dimensional space.

	* For the image codes use a fully connected layer, mapping the codes of length 20480 to 512 features. This layer should be subject to training.
	* Embed the integer encoded words using an Embedding layer (which is essentially a lookup table) of length 512. This layer should also be subject to training.
	
*  Use the image and caption embeddings as inputs to an LSTM as discussed in the paper. Use 500 units for the LSTM.

* Use a fully connected layer with softmax activation mapping the output of the LSTM to a distribution over words (in their integer encoding).

* How does the input and output need to be organized? For how many time steps T should the LSTM be unrolled? For each time step, ```t = 0, . . . , T − 1``` , which embedding should be input to the LSTM and what should be the target?

Train the model by minimizing crossentropy.

* Use Adam with a learning rate 0.001.
* Learn for maximal 100 epochs. Use early stopping with patience 1, providing the separate validation set.
* Use dropout with rate 0.5 for the LSTM.
* Evaluate and report the final training and validation loss.

### Task 4: Generate Test Captions

Implement a greedy decoder model as described in the paper (“beam search with a beam size of 1”). The decoder is akin to the trained model from Task 3. However, rather than providing image codes and captions, the decoder takes only the image codes as input.

* Equip the decoder with the weights from the trained model.
* Use the decoder to predict captions for all test images.
* Show 10 random test images and their predicted captions. Categorize the predictions as in Figure 5 in the paper.
* Compute and report the BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores over the test set.



