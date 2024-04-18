Search Engine for Movies
--------------------------------------------------------------------------------------------------
***Domain specific Information Retrieval System***

**Problem Statement**:

The task is to build a search engine which will cater to the needs of a particular domain. You have
to feed your IR model with documents containing information about the chosen domain. It will
then process the data and build indexes. Once this is done, the user will give a query as an input.
You are supposed to return top 10 relevant documents as the output.

**About the project**

Dataset used - [Kaggle-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)

Have a look at the file [Design Architecture](https://github.com/JuiP/Information-Retrieval/blob/main/Design%20Architecture.pdf). It includes the concepts used along with the modified implementation of the TF-IDF ranking.

Project By:
- **Vedant Dhoble**: Email- <vedantmanoj.d19@iiits.in>
--------------------------------------------------------------------------------------------------
**How to run the code**
--------------------------------------------------------------------------------------------------

1. Clone the repository : [https://github.com/VedanT-27/Information-Retrieval-main.git](https://github.com/VedanT-27/Information-Retrieval-main.git)
2. cd Information-Retrieval
3. Run files in the order: 

              python3 preprocess.py
              python3 tfidf.py
              python3 server.py
4. In your browser go to `http://localhost:3000/`
5. Type your query in the search bar and wait till it returns the relevant documents :)

---------------------------------------------------------------------------------------------------
**Dependencies/modules used**
---------------------------------------------------------------------------------------------------
- python-time
- nltk
- pandas
- pickle5
- Numpy
- heapq_max
- flask
- os-sys
