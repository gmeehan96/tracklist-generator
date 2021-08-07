# Tracklist Generator
This repository contains the code for a Tracklist Generator model trained on house music tracklists scraped from [1001Tracklists](https://www.1001tracklists.com/), a DJ tracklist database and aggregator. The resulting graph-based recommender system leverages the unique structure of tracklist data, incorporating information about the songs played, who those songs were made by, and which DJs were playing them. The final model draws from a library of over 30,000 songs to generate tracklists given a seed song and a seed DJ, or generate personalised tracklists given user preferences. To see the generator in action visit the [Tracklist Generator webpage](https://tracklist-generator.azurewebsites.net)\* (open on desktop for best experience). 

## Code
The code is broken down into four notebooks outlining the steps taken to process the data and build the model. 

- The [Data Preparation notebook](https://github.com/gmeehan96/tracklist-generator/blob/main/1.%20Data_Preparation.ipynb) describes how co-occurrence information extracted from the tracklists is processed and saved to be used in subsequent models. This notebook also creates the population of 'users' in the recommendation system by treating the DJs as users, building a click-matrix based on the songs that they play. 
- The [Embeddings notebook](https://github.com/gmeehan96/tracklist-generator/blob/main/2.%20Embeddings.ipynb) trains a recommendation system which expands on the [Multi-Graph Convolution Collaborative Filtering model described by Sun et. al](https://arxiv.org/abs/2001.00267). Our version expands the typical bipartite user-item recommendation scenario by adding a third type of vertex, namely the artists of the songs, and also by adding a matrix factorisation loss to the training objective in the style of [GloVe word embeddings](https://nlp.stanford.edu/pubs/glove.pdf). The resulting system produces recommendation scores for each user-song pair.
- The [Tracklist Model notebook](https://github.com/gmeehan96/tracklist-generator/blob/main/3.%20Tracklist_Model.ipynb) trains the main model which underpins the Tracklist Generator. The model follows a similar approach to language models in NLP, where, given a corpus of sentences, the training task is to predict the next word in the sentence given the previous words as input. Our model provides a novel approach to this task by simulataneously training a hierarchical clustering of the underlying 'vocabulary' (in our case, songs), which the tracklist model uses to help its predictions. The clustering method used is based on Tsitsulin et. al's [Deep Modularity Networks](https://arxiv.org/abs/2006.16904), expanded into a hierarchy and with a novel regularization term. The architecture of the Tracklist Model can be found in the diagram below; see the notebook for a more in-depth explanation.
- Finally, the [Recommendation Notebook](https://github.com/gmeehan96/tracklist-generator/blob/main/4.%20Recommendation.ipynb) demonstrates how the recommender system and tracklist model can be combined to generate tracklists which are realistic but also influenced by the user's taste and the original seed song. The graph-based approach to the recommender system means that it is also possible to generate tracklists for unseen users - all that is needed is a list of some of their liked or preferred songs. To try it yourself, check out the [webpage](https://tracklist-generator.azurewebsites.net).

### Tracklist Model Architecture
![architecture](https://user-images.githubusercontent.com/45896163/128604803-43a9fa9c-ada2-4621-b1ab-b7ef305eaa14.jpg)

If you have any questions or comments, please get in touch via email to gregor.meehan at gmail.com.

\*Note that this webpage is a lightweight web application intended for demonstration purposes only. All underlying data belongs to 1001Tracklists and has not been included in the repository. 
