# Identifying self-reported worldwide tweets potential Lyme disease cases: A deep learning modelling approach enhanced with sentimental words through emojis.

Authors: Elda K. E. Laison<sup>1</sup>, Mohammed H. Ibrahim<sup>1,2</sup>, Srikanth Boligarla<sup>3</sup>, Jiaxin F. Li<sup>3</sup>, Raja Mahadevan<sup>3</sup>, Austen Ng<sup>3</sup>, Lee W. Yi<sup>3</sup>, Jang Park<sup>3</sup>, Yijun Yin,<sup>3</sup>, Bouchra R Nasri<sup>1,4,5,6</sup>.
1.	Department of Social and Preventive Medicine, École de Santé Publique, University of Montreal, Montréal, Canada. 
2.	Department of Mathematics, Faculty of Science, Zagazig University, Zagazig, Egypt.
3.	Harvard Extension School, Harvard University, Cambridge, United States.
4.	Centre de recherches mathématiques, Montréal, Canada.
5.	Centre de recherche en santé publique, Montréal, Canada.
6.	Data Informatics Center of Epidemiology, PathCheck, Cambridge, United States.

## Description of data file:
We have put three (2) files in the folder named “data” in the suppository on GitHub. These files are described below:  
 
-	*Tweet_Geolocations_clean*: is a csv file that contains all the cleaned tweets with the tweet_id, the country of origin and the corresponding of the label class (0 or 1). 
-	*Tweet_counts*: is a csv file containing information about the tweets collected from Twitter API after the cleaning process, after having been regrouped according to the https://github.com/nasrilab/Twitter_study/tree/main/data countries where they originate from and classified into Lyme counts and non-Lyme counts. 


## Description of the oversampling method: 
Using a (Generative Adversarial Network) GAN-based method to
oversample tweets in minor classes (i.e., generate synthetic tweets
that belong to the underrepresented class). Install tensorflow (which also includes Keras): pip install tensorflow. 
•	Data Preparation: First, you need to have a dataset of tweets with class labels, and you should split it into a major class and a minor class. For this example, let’s assume you have already preprocessed the data and have two lists: major_class_tweets and minor_class_tweets. 
•	Data Vectorization: Convert the tweet data into numerical representations (e.g., using word embeddings like Word2Vec, GloVe, or tokenization techniques like TF-IDF). Additionally, create a label vector for the minor class to distinguish between real and synthetic tweets later. Run the attached code to oversample the tweets after passing the minor_class_tweets as input.
The oversampling code is available on the GitHub link as the folder “code” that contains two files (GAN_oversampling.py and Oversampling_tweets2.py). We have used Python for the  coding of the data. 



