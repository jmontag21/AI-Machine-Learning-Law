from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json, re, nltk.stem




#                   precision    recall  f1-score   support
#
# CitationSentence       0.99      0.95      0.97        87
# EvidenceSentence       0.91      0.93      0.92       243
#  FindingSentence       0.82      0.79      0.80        52
#LegalRuleSentence       0.83      0.93      0.87        94
#ReasoningSentence       0.69      0.57      0.62        67
#         Sentence       0.76      0.76      0.76        37
#
#      avg / total       0.87      0.87      0.87       580
def load_file(DATA_FILE):
    global data, labels
    
    data = [ ]
    labels = [ ]
    
    global all_types
    all_types = ['CitationSentence',
     'EvidenceSentence',
      'FindingSentence',
    'LegalRuleSentence',
    'ReasoningSentence',
             'Sentence']
    
     
    global WHAT, WHATLABEL, NOTWHAT, NOTWHATLABEL, TOKEN_PATTERN
    global REMOVE_DIGITS, REMOVE_PUNCT, STEM
    WHAT="MultiClass"
    #WHAT='FindingSentence'
    #WHAT='EvidenceSentence'
    #WHAT='LegalRuleSentence'
    #WHAT='ReasoningSentence'
    WHATLABEL = WHAT
    NOTWHAT="non-"+WHAT
    NOTWHATLABEL= NOTWHAT
    TOKEN_PATTERN='(?u)\\b\\w\\w+\\b'
    REMOVE_DIGITS=True
    REMOVE_PUNCT=False
    STEM=False
    
    print("Classifier for", WHAT)
    
    global filter_out
    filter_out = set()
    #filter_out = set( [ all_types[0], all_types[1], all_types[3] ] )
    #filter_out = set( [ all_types[0], all_types[1], all_types[3], all_types[5] ] )
    
    print("Filtering out:", filter_out)
    
    #DATA_FILE="tagged_all_sentences.json"
    #DATA_FILE="all_sentences.json"
    
    print("DATA_FILE =", DATA_FILE)
    global allsentences, sentences, stemmer, saved_sentences
    allsentences = json.loads(open(DATA_FILE).read())
    sentences = allsentences['sentences']
    print("Loaded",len(sentences),"sentences.")
    stemmer = nltk.stem.SnowballStemmer('english')
    saved_sentences = []
    
    if REMOVE_DIGITS:
        print("Scrubbing out digits.")
    if REMOVE_PUNCT:
        print("Scrubbing out punctuation.")
    if STEM:
        print("Stemming is on.")    
    for sentence in sentences:
        st = sentence['text']
        if "Citation Nr" in st:
            continue
        if st.endswith("FINDINGS OF FACT"):
            continue
        if len(filter_out.intersection(sentence['rhetRole'])) != 0:
            continue
        if REMOVE_DIGITS:
            s = re.sub(r'\d+', ' ', st)
        if REMOVE_PUNCT:
            s = re.sub(r'\W+', ' ', s)
        
        if STEM:
            sw = s.split()
            s = ""
            for w in sw:
                if len(w) > 1:
                    s += stemmer.stem(w) + " "       
    
        #if s == s.strip():
            #continue
        saved_sentences.append(sentence['text'])    
        data.append(s)
        if WHAT == "MultiClass":
            labels.append(sentence['rhetRole'][0])
        elif WHAT in sentence['rhetRole']:
            labels.append(WHATLABEL)
        else:
            labels.append(NOTWHATLABEL)
    
    print("Num sentences after scrubbing and filtering=", len(data))
    
    global vectorizer, features, features_nd, names
    vectorizer = CountVectorizer(analyzer = 'word', lowercase = 'False', stop_words=None, token_pattern=TOKEN_PATTERN, ngram_range=(1,2), binary = False)
    print(vectorizer)
    print("Vectorizing ...", end="")    
    features = vectorizer.fit_transform(data)
    features_nd = features.toarray()
    names = vectorizer.get_feature_names()
    print("done.")

def multi(N, randkeys, TRAIN_SIZE, model):
    if N > len(randkeys):
        print("Not enough randkeys, exiting")
        return None

    TEST_SIZE=1-TRAIN_SIZE
    print(" TRAIN_SIZE =", round(TRAIN_SIZE,2))
    print(" TEST_SIZE =", round(TEST_SIZE,2))
    print(model)

    accuracy = []
    class_rep = [ ]
    conf_mat =[ ]
    
    for rep in range(N):
        print("Run #", rep)
#        print("Randomly splitting training & test ...", end="")
        RAND_KEY=randkeys[rep]
        print(" RAND_KEY =", RAND_KEY)
        
        X_train, X_test, y_train, y_test = train_test_split(features_nd, labels, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RAND_KEY)
#        print("done.")
        
#        print("Test set size =", len(X_test))
        
        model = model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X_test)
        
#        features_as_list = features_nd.tolist()
#        for i in range(len(X_test)):
#            #if y_pred[i] != y_test[i]:
#            if y_pred[i] != y_test[i] and y_test[i] == "ReasoningSentence":
#                print("*"*50)
#                print("Labeled as", y_pred[i])
#                print("Actual is", y_test[i])
#                Xtestl = X_test[i].tolist()
#                ind = features_as_list.index(Xtestl)
#                print("Data =", data[ind])
    
        print("Accuracy score:", accuracy_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        class_rep.append(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        conf_mat.append(confusion_matrix(y_test, y_pred).tolist())
    
    print("All done.")
    return accuracy, class_rep, conf_mat
    
model = None

def try_fit():
    multi(1, (1234,), 0.9, model)
    
def go_svm():
    global model
    print("Support Vector Machine ...")
    from sklearn.svm import SVC
    model = SVC(gamma='scale', kernel='linear')
    
def try_svm():
    go_svm()
    try_fit()

def go_log_reg():
    global model
    print("Logistic Regression ...")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l1', multi_class='ovr', solver='liblinear')
    
    
def Kmean():
    global model 
    print("KmeanMachine")
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=6, init='k-means++', n_init=6, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    
def try_log_reg():
    go_log_reg()
    try_fit()

def go_KNN():
    global model
    print("KNearestNeighbor ...")
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(15)

def go_NB():
    global model
    print("Naive Bayes ...")
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

#ComplementNB
def go_CNB():
    global model
    print("Complement Naive Bayes ...")
    from sklearn.naive_bayes import ComplementNB
    model = ComplementNB()
    
def go_DT():
    global model
    print("Decision Tree ...")
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy')

def go_AdaBoost():
    global model
    print("AdaBoost ...")
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100)
    
from sklearn.feature_selection import SelectFromModel

def top(N):
    model2 = SelectFromModel(model, prefit=True, max_features=N)
    chosen = model2.get_support(True)
    return [names[i] for i in chosen]
    
def ranked(N):
    ans = [ ]
    for i in range(1,N+1):
        tops = top(i)
        for x in tops:
            if x not in ans:
                ans.append(x)
    return ans

def viz():
    import graphviz
    from sklearn import tree
    import re
    dot_data = tree.export_graphviz(model, class_names=["Finding", "Non-Finding"], out_file=None)
    a = re.findall("X\[\d+\]", dot_data)
    sd = dot_data.replace('<= 0.5', 'appears')
    sd = sd.replace('False', 'T')
    sd = sd.replace('True', 'F')
    for v in a:
        num = int(v[2:-1])
        name = names[num]
        sd = sd.replace(v,name)
    graph = graphviz.Source(sd)
    graph.render("ASAIL-names")

def extractscores(rep, n_classes=6):
    lines = rep.split("\n")[2:n_classes+2]
    scores = [ ]
    for line in lines:
        scores.append([float(x) for x in line.split()[1:4]])
    return scores
    
def avg_class_rep(reports_list, n_classes=6):
    totals = []
    for i in range(n_classes):
        totals.append([0] * 3)
        
    for rep in reports_list:
        scores = extractscores(rep, n_classes)
        for r in range(len(totals)):
            for c in range(len(totals[0])):
                totals[r][c] += scores[r][c]
    
    for r in range(len(totals)):
        for c in range(len(totals[0])):
            totals[r][c] = round(totals[r][c]/len(reports_list),2)
    return totals

def avg_conf_mat(mats_list, n_classes = 6):
    totals = [ ]
    for row in range(len(mats_list[0])):
        totals.append([0] * len(mats_list[0][0]))
    for mat in mats_list:
        for r in range(len(totals)):
            for c in range(len(totals[0])):
                totals[r][c] += mat[r][c]
                
    for r in range(len(totals)):
        for c in range(len(totals[0])):
            totals[r][c] = round(totals[r][c]/len(mats_list),2)
    return totals

def avg_acc_score(acc_list):
    return sum(acc_list)/len(acc_list)

def all_avgs(results, n_classes=6):
    print("Accuracy score summary\n", avg_acc_score(results[0]))
    print("\nClassification report summary")
    for x in avg_class_rep(results[1], n_classes):
        print(x)
    print("\nConfusion matrix summary")
    for x in avg_conf_mat(results[2], n_classes):
        print(x)
        
        
        

def ranked_for_all():
    for X in [go_svm, go_log_reg, go_NB]:
        X()
        multi(1, [1220], 0.9, model)
        print("Ranked features for: " + str(X))
        ranks = ranked(20)
        for i in range(len(ranks)):
            print(str(i+1)+". " + ranks[i])
    

def vocab_counts():
    tokens = 0
    bigrams = 0
    trigrams = 0
    for w in names:
        if " " not in w:
            tokens += 1
        else:
            neww = w.replace(" ", "-", 1)
            if " " in neww:
                trigrams += 1
            else:
                bigrams += 1
    return tokens, bigrams, trigrams

def label_distribution():
    lab_distrib= { }
    for label in labels:
        if label in lab_distrib:
            lab_distrib[label] += 1
        else:
            lab_distrib[label] = 1
    return lab_distrib






#if __name__ == "__main__":
