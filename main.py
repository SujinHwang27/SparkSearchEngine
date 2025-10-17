import wget
import tarfile
import re
import math
import pprint
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


# wget.download('http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz')
# file = tarfile.open('MovieSummaries.tar.gz') 
# file.extractall('./') 
# file.close()

nltk.download('punkt_tab')
# nltk.download('stopwords')

conf = SparkConf().setMaster("local").setAppName("main")
sc = SparkContext(conf = conf)

''' Load text data from plot_summaries.txt file '''
plots = sc.textFile("MovieSummaries/plot_summaries.txt")    # "movieId\tSummary"
N = plots.count()   # 42306

''' Load Movie metadata '''
movie_metadata = sc.textFile("MovieSummaries/movie.metadata.tsv").map(lambda line: line.split("\t"))
movie_names = movie_metadata.map(lambda cols: (cols[0], cols[2]))


''' Broadcast stopwords set'''
stop_words = set(stopwords.words('english'))
broadcast_stopwords = sc.broadcast(stop_words)


def filter_stopwords(text):
    ''' 
    Tokenize and filter stopwords 
    '''
    # clean non-alnums out but keep dashed words
    text = text.replace('-', ' ')
    tokens = word_tokenize(text.lower())   
    filtered_tokens = [w for w in tokens if len(w)>0 and w.isalnum() and w not in broadcast_stopwords.value]

    return filtered_tokens



def build_tfidf(plots):
    '''
    Compute tfidf for each terms and docs

    Output: (term_i, (doc_j, tfidf_if))
    '''

    # Map into key-value pairs (movieId, plot summary)
    movie_plots = plots.map(lambda x:(x.split("\t")[0], x.split("\t")[1]))

    # Remove stopwords from plot summary
    movie_plots_clean = movie_plots.mapValues(lambda x: filter_stopwords(x))

    # Map into key-value pairs (movieId, term)
    movie_term = movie_plots_clean.flatMapValues(lambda x:x)

    # Flip them into (term, movieId) 
    term_movie = movie_term.map(lambda x: (x[1], x[0]))
    
    # Reduce into term-frequency ((term, movieId), term-frequency)
    tf = term_movie.map(lambda x: ((x[0], x[1]),1)).reduceByKey(lambda x,y:x+y).map((lambda x: (x[0][0], (x[0][1], x[1]))))

    # Calculate idf
    df = tf.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b:a+b)
    idf = df.map(lambda x: (x[0], math.log(N / x[1])))
    tf_idf = tf.join(idf).map(lambda x: (x[0], (x[1][0][0], x[1][0][1]*x[1][1])))


    return tf_idf, idf
                                                                                  
def query_short(query, tf_idf):
    ''' Return top5 (movieId, tf-idf) for a single term query '''
    result = tf_idf.filter(lambda x: x[0]==query).takeOrdered(5, key=lambda x:-x[1][1])
    result_kv = [(x[1][0], x[1][1]) for x in result]
    result_rdd = sc.parallelize(result_kv)

    result_named = result_rdd.join(movie_names).map(lambda x: (x[1][1], x[1][0])).sortBy(lambda x: -x[1])

    return {query: result_named.collect()}


def process_short_queries(queryfile, tf_idf):
    # read short query file into a Python list
    with open(queryfile, "r") as file:
        short_queries = [q.strip() for q in file.readlines()]

    res = [query_short(q, tf_idf) for q in short_queries]

    return res



def cosine_sim(doc, query_vec, query_norm):
    """
    Compute cosine similarity between one document and the query.
    doc: (docId, (doc_terms_dict, doc_norm))
    query_vec: dict(term -> tfidf)
    query_norm: float
    """
    doc_id, (doc_terms, doc_norm) = doc
    # dot product
    dot = sum(doc_terms.get(t, 0.0) * w for t, w in query_vec.items())
    if doc_norm == 0 or query_norm == 0:
        return (doc_id, 0.0)
    return (doc_id, dot / (doc_norm * query_norm))


def process_single_query(sc, query, tf_idf, idf):
    """Process one multi-term query and return top-5 docs"""
    # remove stopwords
    query_terms = filter_stopwords(query)

    # build query term frequency
    query_tf = (sc.parallelize(query_terms)
                  .map(lambda x: ((x, 'q'), 1))
                  .reduceByKey(lambda x, y: x + y)
                  .map(lambda x: (x[0][0], (x[0][1], x[1]))))

    # query tf-idf
    query_tf_idf = query_tf.join(idf).map(
        lambda x: (x[0], (x[1][0][0], x[1][0][1] * x[1][1]))
    )

    query_vector = dict(query_tf_idf.map(lambda x: (x[0], x[1][1])).collect())
    query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))

    # build doc vectors once
    doc_dicts = tf_idf.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(dict)
    doc_norms = doc_dicts.mapValues(lambda vec: math.sqrt(sum(v**2 for v in vec.values())))
    doc_vectors = doc_dicts.join(doc_norms)

    # compute cosine similarities
    query_doc_cossim = doc_vectors.map(lambda doc: cosine_sim(doc, query_vector, query_norm))

    # top 5
    top5 = query_doc_cossim.takeOrdered(5, key=lambda x: -x[1])

    # result_kv = [(x[1][0], x[1][1]) for x in top5]
    result_rdd = sc.parallelize(top5)

    result_named = result_rdd.join(movie_names).map(lambda x: (x[1][1], x[1][0])).sortBy(lambda x: -x[1])

    return {query: result_named.collect()}


def process_long_queries(sc, queryfile, tf_idf, idf):
    """Process multiple queries from a file and return dict"""
    with open(queryfile, "r") as f:
        queries = [line.strip() for line in f if line.strip()]

    results = [process_single_query(sc, q, tf_idf, idf) for q in queries]

    return results

tf_idf, idf = build_tfidf(plots)
tf_idf = tf_idf.cache()
idf = idf.cache()

# process short queries and display
short_queries_res = process_short_queries("short_queries.txt", tf_idf)

long_queries_res = process_long_queries(sc, "long_queries.txt", tf_idf, idf)


pprint.pprint(short_queries_res)
pprint.pprint(long_queries_res)

# Write out queries and the results in output.json file
with open("output.json", "w") as f:
    json.dump(short_queries_res, f, indent=4)
    json.dump(long_queries_res, f, indent=4)

sc.stop()


                                                                                       


