# RNN Book Writer
A RNN machine learning project to write a book

# Main Idea
By giving a book or a text file, like a math book, the idea is to learn from it, and then ask the ML to write its own book, based on the style learned from the given book

# Results
We have got excellent results, with consistent chapters written by our ML.

# Conceptions of RNN
  * It's a supervised Machine Learning
	* It's like a short term memory, like our brain(Frontal Lobe)
	* We basically create a new dimension on our ANN, and flaten them, to create just like a sigle node, from tons.
	* Then, we create loops in the noods itself.
	* This makes the nodes to remember some memory, in short time. 
	* Can identify if a comment is positive or negative.
	* We can use RNN to translation, because we need the information of previous words, to translate correct.
	* Can be used to subtitle a movie.

# Combining CNN + ANN + RNN
	* With this combination, we can use CNN to filter images + ANN to classify the image. 
	* Then, we use those output to RNN, which can then describe the picture
	* For example, CNN + ANN can identify a dog in a picture
	* And RNN can describe the color and what's the dog doing in the picute, like jumping, running, etc.

# Types of RNN
	1) One to many
		* One input, like a dog
		* Various output, like classification and action describes

	2) Many to one
		* Many input, and one single output.
		* Like a comment as input, and yes or no for output, saying if it's a bad or good comment.

	3) Many to Manay
		* Many input and many output.
		* Like a phrase as input, and the same phrase as output, but translated to another language.

# Vanishing Gradient --> Big problem in RNN
	* As the gradient descend goal to find the global minimum, as described before.
	* In RNN we have a similar idea, but now, previous information goes to next or come back to the node itself.
	* The problem is because in the formula for RNN, we multiply the weights and values, thourthout your RNN
	* But the problem is that we start with low values for weights, and this cause a problem in the formula,
	* Cause your final value gets lower and lower to fast(Cost function). Then, we may get wrong weights
	* What if we start with hight values for weights?
		* Then, you may get done fast, but exploding your RNN, with wrong weights and bad training

	* Solution
		* For Exploding Gradiente
			* Use the idea of Penalties
			* Gradient Clipping( Maximum value for the propragation)
		* For vanishing Gradiente
			* Weight Initialization
			* Echo state Networks
			* Long short term Memory Networks(LSTMs) --> Best option


# LSTMs - Long short term Memory Networks --> Very complicated
	* It makes Wrec = 1. This way it avoids vanishing or exploding. 
	* It makes a continuos line crossing over the RNN.
	* It's very similar to eletronics, where we have tubs.
	* The main top tube(Wrec = 1), passes the memory
	* And the tubes conecting to it, represents new memory that the node may gets
	* Each node has all this estruct
	* Where it gets as input the ouput of previous node.

# LSTMs Aplications
	* Write a book
	* describes pictures
