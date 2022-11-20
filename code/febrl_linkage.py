import recordlinkage as rl, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from recordlinkage.preprocessing import phonetic
from numpy.random import choice
import collections, numpy
from IPython.display import clear_output
from sklearn.model_selection import train_test_split, KFold

import generator_dataset_febrl as gd
import plots as pl

def generate_true_links(df): 
    # although the match_id column is included in the original df to imply the true links,
    # this function will create the true_link object identical to the true_links properties
    # of recordlinkage toolkit, in order to exploit "Compare.compute()" from that toolkit
    # in extract_function() for extracting features quicker.
    # This process should be deprecated in the future release of the UNSW toolkit.
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    processed = 0
    for match_id in df["match_id"].unique():
        if match_id != -1:    
            processed = processed + 1
            # print("In routine generate_true_links(), count =", processed)
            # clear_output(wait=True)
            linkages = df.loc[df['match_id'] == match_id]
            for j in range(len(linkages)-1):
                for k in range(j+1, len(linkages)):
                    indices_1 = indices_1 + [linkages.iloc[j]["rec_id"]]
                    indices_2 = indices_2 + [linkages.iloc[k]["rec_id"]]    
    links = pd.MultiIndex.from_arrays([indices_1,indices_2])
    return links

def generate_false_links(df, size):
    # A counterpart of generate_true_links(), with the purpose to generate random false pairs
    # for training. The number of false pairs in specified as "size".
    df["rec_id"] = df.index.values.tolist()
    indices_1 = []
    indices_2 = []
    unique_match_id = df["match_id"].unique()
    for j in range(size):
            false_pair_ids = choice(unique_match_id, 2)
            candidate_1_cluster = df.loc[df['match_id'] == false_pair_ids[0]]
            candidate_1 = candidate_1_cluster.iloc[choice(range(len(candidate_1_cluster)))]
            candidate_2_cluster = df.loc[df['match_id'] == false_pair_ids[1]]
            candidate_2 = candidate_2_cluster.iloc[choice(range(len(candidate_2_cluster)))]    
            indices_1 = indices_1 + [candidate_1["rec_id"]]
            indices_2 = indices_2 + [candidate_2["rec_id"]]  
    links = pd.MultiIndex.from_arrays([indices_1,indices_2])
    return links

def swap_fields_flag(f11, f12, f21, f22):
    return int((f11 == f22) and (f12 == f21))

def extract_features(df, links):
    c = rl.Compare()
    c.string('given_name', 'given_name', method='jarowinkler', label='y_name')
    c.string('given_name_soundex', 'given_name_soundex', method='jarowinkler', label='y_name_soundex')
    c.string('given_name_nysiis', 'given_name_nysiis', method='jarowinkler', label='y_name_nysiis')
    c.string('surname', 'surname', method='jarowinkler', label='y_surname')
    c.string('surname_soundex', 'surname_soundex', method='jarowinkler', label='y_surname_soundex')
    c.string('surname_nysiis', 'surname_nysiis', method='jarowinkler', label='y_surname_nysiis')
    c.exact('street_number', 'street_number', label='y_street_number')
    c.string('address_1', 'address_1', method='levenshtein', threshold=0.7, label='y_address1')
    c.string('address_2', 'address_2', method='levenshtein', threshold=0.7, label='y_address2')
    c.exact('postcode', 'postcode', label='y_postcode')
    c.exact('day', 'day', label='y_day')
    c.exact('month', 'month', label='y_month')
    c.exact('year', 'year', label='y_year')
        
    # Build features
    feature_vectors = c.compute(links, df, df)
    return feature_vectors

def generate_train_X_y(df, train_true_links):
    # This routine is to generate the feature vector X and the corresponding labels y
    # with exactly equal number of samples for both classes to train the classifier.
    pos = extract_features(df, train_true_links)
    train_false_links = generate_false_links(df, len(train_true_links))    
    neg = extract_features(df, train_false_links)
    X = pos.values.tolist() + neg.values.tolist()
    y = [1]*len(pos)+[0]*len(neg)
    X, y = shuffle(X, y, random_state=0)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_model(modeltype, modelparam, train_vectors, train_labels, modeltype_2):
    if modeltype == 'svm': # Support Vector Machine
        model = svm.SVC(C = modelparam, kernel = modeltype_2)
        model.fit(train_vectors, train_labels) 
    elif modeltype == 'lg': # Logistic Regression
        if modeltype_2 == "l1":
            model = LogisticRegression(C=modelparam, penalty = modeltype_2,class_weight=None, dual=False, fit_intercept=True, 
                                   intercept_scaling=1, max_iter=5000, multi_class='ovr', 
                                   n_jobs=1, random_state=None, solver='liblinear')
                                      
        if modeltype_2 == "l2":
            model = LogisticRegression(C=modelparam, penalty = modeltype_2,class_weight=None, dual=False, fit_intercept=True, 
                                   intercept_scaling=1, max_iter=5000, multi_class='ovr', 
                                   n_jobs=1, random_state=None)

        model.fit(train_vectors, train_labels)
    elif modeltype == 'nb': # Naive Bayes
        model = GaussianNB()
        model.fit(train_vectors, train_labels)
    elif modeltype == 'nn': # Neural Network
        model = MLPClassifier(solver='lbfgs', alpha=modelparam, hidden_layer_sizes=(256, ), 
                              activation = modeltype_2,random_state=None, batch_size='auto', 
                              learning_rate='constant',  learning_rate_init=0.001, 
                              power_t=0.5, max_iter=10000, shuffle=True, 
                              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                              nesterovs_momentum=True, early_stopping=False, 
                              validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.fit(train_vectors, train_labels)
    return model

def classify(model, test_vectors):
    result = model.predict(test_vectors)
    return result

    
def evaluation(test_labels, result):
    true_pos = np.logical_and(test_labels, result)
    count_true_pos = np.sum(true_pos)

    true_neg = np.logical_and(np.logical_not(test_labels),np.logical_not(result))
    count_true_neg = np.sum(true_neg)

    false_pos = np.logical_and(np.logical_not(test_labels), result)
    count_false_pos = np.sum(false_pos)

    false_neg = np.logical_and(test_labels,np.logical_not(result))
    count_false_neg = np.sum(false_neg)

    precision = count_true_pos/(count_true_pos+count_false_pos)
    sensitivity = count_true_pos/(count_true_pos+count_false_neg) # sensitivity = recall
    confusion_matrix = [count_true_pos, count_false_pos, count_false_neg, count_true_neg]
    no_links_found = np.count_nonzero(result)
    no_false = count_false_pos + count_false_neg
    Fscore = 2*precision*sensitivity/(precision+sensitivity)

    metrics_result = {'no_false':no_false, 'confusion_matrix':confusion_matrix ,'precision':precision,
                     'sensitivity':sensitivity ,'no_links':no_links_found, 'F-score': Fscore, 'true_pos': count_true_pos,
                     'true_neg': count_true_neg, 'false_pos': count_false_pos, 'false_neg': count_false_neg}

    return metrics_result

def blocking_performance(candidates, true_links, df):
    count = 0
    for candi in candidates:
        if df.loc[candi[0]]["match_id"]==df.loc[candi[1]]["match_id"]:
            count = count + 1
    return count


def load_file(path):
    df = pd.read_csv(path, index_col = "rec_id")
    return df

def generate_train_sets(df_train):
    ## TRAIN SET CONSTRUCTION
    train_true_links = generate_true_links(df_train)
    print("Train set size:", len(df_train), ", number of matched pairs: ", str(len(train_true_links)))

    # Preprocess train set
    df_train['postcode'] = df_train['postcode'].astype(str)
    df_train['given_name_soundex'] = phonetic(df_train['given_name'], method='soundex')
    df_train['given_name_nysiis'] = phonetic(df_train['given_name'], method='nysiis')
    df_train['surname_soundex'] = phonetic(df_train['surname'], method='soundex')
    df_train['surname_nysiis'] = phonetic(df_train['surname'], method='nysiis')

    # Final train feature vectors and labels
    X_train, y_train = generate_train_X_y(df_train, train_true_links)
    print("Finished building X_train, y_train")
    return X_train, y_train

def block_test_sets(df_test):
    # Blocking Criteria: declare non-match of all of the below fields disagree
    test_true_links = generate_true_links(df_test)
    leng_test_true_links = len(test_true_links)
    print("Test set size:", len(df_test), ", number of matched pairs: ", str(leng_test_true_links))

    print("BLOCKING PERFORMANCE:")
    blocking_fields = ["given_name", "surname", "postcode"]
    all_candidate_pairs = []
    for field in blocking_fields:
        block = rl.Index()
        block.block(on=field)
        candidates = block.index(df_test)
        print("candidates:", candidates)
        detects = blocking_performance(candidates, test_true_links, df_test)
        all_candidate_pairs = candidates.union(all_candidate_pairs)
        print("Number of pairs of matched "+ field +": "+str(len(candidates)), ", detected ",
            detects,'/'+ str(leng_test_true_links) + " true matched pairs, missed " + 
            str(leng_test_true_links-detects) )
    detects = blocking_performance(all_candidate_pairs, test_true_links, df_test)
    print("Number of pairs of at least 1 field matched: " + str(len(all_candidate_pairs)), ", detected ",
        detects,'/'+ str(leng_test_true_links) + " true matched pairs, missed " + 
            str(leng_test_true_links-detects) )
    return all_candidate_pairs

def generate_test_sets(df_test, all_candidate_pairs):
    ## TEST SET CONSTRUCTION

    # Preprocess test set
    print("Processing test set...")
    print("Preprocess...")
    df_test['postcode'] = df_test['postcode'].astype(str)
    df_test['given_name_soundex'] = phonetic(df_test['given_name'], method='soundex')
    df_test['given_name_nysiis'] = phonetic(df_test['given_name'], method='nysiis')
    df_test['surname_soundex'] = phonetic(df_test['surname'], method='soundex')
    df_test['surname_nysiis'] = phonetic(df_test['surname'], method='nysiis')

    # Test feature vectors and labels construction
    print("Extract feature vectors...")
    df_X_test = extract_features(df_test, all_candidate_pairs)
    vectors = df_X_test.values.tolist()
    labels = [0]*len(vectors)
    feature_index = df_X_test.index
    for i in range(0, len(feature_index)):
        if df_test.loc[feature_index[i][0]]["match_id"]==df_test.loc[feature_index[i][1]]["match_id"]:
            labels[i] = 1
    X_test, y_test = shuffle(vectors, labels, random_state=0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print("Count labels of y_test:",collections.Counter(y_test))
    print("Finished building X_test, y_test")
    return X_test, y_test


def train_test_models(X_train, y_train, X_test, y_test):
    ## BASE LEARNERS CLASSIFICATION AND EVALUATION
    # Choose model
    print("BASE LEARNERS CLASSIFICATION PERFORMANCE:")
    models = {'svm': ['linear','rbf'], 'lg': ['l1','l2'], 'nn': ['relu','logistic']}

    modeltype = 'svm' # choose between 'svm', 'lg', 'nn'
    modeltype_2 = 'linear'  # 'linear' or 'rbf' for svm, 'l1' or 'l2' for lg, 'relu' or 'logistic' for nn
    modelparam_range = [.001,.002,.005,.01,.02,.05,.1,.2,.5,1,5,10,20,50,100,200,500,1000,2000,5000] # C for svm, C for lg, alpha for NN
    
    df_precision = pd.DataFrame()
    df_sensitivity = pd.DataFrame()
    df_fscore = pd.DataFrame()
    df_nb_false = pd.DataFrame()
    
    df_precision["param"] = modelparam_range
    df_sensitivity["param"] = modelparam_range
    df_fscore["param"] = modelparam_range
    df_nb_false["param"] = modelparam_range


    for i in models:
        for j in models[i]:
            print("Model:",i,", Param_1:",j, ", tuning range:", modelparam_range)
            precision = []
            sensitivity = []
            Fscore = []
            nb_false = []

            for modelparam in modelparam_range:
                md = train_model(i, modelparam, X_train, y_train, j)
                final_result = classify(md, X_test)
                final_eval = evaluation(y_test, final_result)
                precision += [final_eval['precision']]
                sensitivity += [final_eval['sensitivity']]
                Fscore += [final_eval['F-score']]
                nb_false  += [final_eval['no_false']]
        
            df_precision[i + "-" + j] = precision
            df_sensitivity[i + "-" + j] = sensitivity
            df_fscore[i + "-" + j] = Fscore
            df_nb_false[i + "-" + j] = nb_false

            print("No_false:",nb_false,"\n")
            print("Precision:",precision,"\n")
            print("Sensitivity:",sensitivity,"\n")
            print("F-score:", Fscore,"\n")
            print("")

    return df_precision, df_sensitivity, df_fscore, df_nb_false


def bagging_model(X_train, y_train, X_test, y_test):
    ## ENSEMBLE CLASSIFICATION AND EVALUATION

    print("BAGGING PERFORMANCE:\n")
    modeltypes = ['svm', 'nn', 'lg'] 
    modeltypes_2 = ['linear', 'relu', 'l2']
    modelparams = [0.005, 100, 0.2]
    
    df_bagging = pd.DataFrame(columns = ['model', 'no_false', 'confusion_matrix', 'precision', 'sensitivity', 
                                            'no_links', 'F-score', 'true_pos', 'true_neg', 'false_pos', 'false_neg'])

    nFold = 10
    kf = KFold(n_splits=nFold)

    model_raw_score = [0]*3
    model_binary_score = [0]*3
    model_i = 0

    for model_i in range(3):
        modeltype = modeltypes[model_i]
        modeltype_2 = modeltypes_2[model_i]
        modelparam = modelparams[model_i]

        print(modeltype, "per fold:")
        iFold = 0
        result_fold = [0]*nFold
        final_eval_fold = [0]*nFold
        
        for train_index, valid_index in kf.split(X_train):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            md =  train_model(modeltype, modelparam, X_train_fold, y_train_fold, modeltype_2)
            result_fold[iFold] = classify(md, X_test)
            final_eval_fold[iFold] = evaluation(y_test, result_fold[iFold])
            print("Fold", str(iFold), final_eval_fold[iFold])
            iFold = iFold + 1
        bagging_raw_score = np.average(result_fold, axis=0)
        bagging_binary_score  = np.copy(bagging_raw_score)
        bagging_binary_score[bagging_binary_score > 0.5] = 1
        bagging_binary_score[bagging_binary_score <= 0.5] = 0
        bagging_eval = evaluation(y_test, bagging_binary_score)
        print(modeltype, "bagging:", bagging_eval)
        print('')
        model_raw_score[model_i] = bagging_raw_score
        model_binary_score[model_i] = bagging_binary_score

        model = modeltypes[model_i] + " - " + modeltypes_2[model_i] + " - " + str(modelparams[model_i])
        df_bagging.loc[model_i] =  [model] + [v for _, v in bagging_eval.items()]

    thres = .99

    print("STACKING PERFORMANCE:\n")
    stack_raw_score = np.average(model_raw_score, axis=0)
    stack_binary_score = np.copy(stack_raw_score)
    stack_binary_score[stack_binary_score > thres] = 1
    stack_binary_score[stack_binary_score <= thres] = 0
    stacking_eval = evaluation(y_test, stack_binary_score)
    print(stacking_eval)

    model = "stack bagging"
    df_bagging.loc[len(df_bagging)] =  [model] + [v for _, v in stacking_eval.items()]

    return df_bagging

def get_params_maxscore(df, model):
    
    col = df.filter(like=["model", model])

    for c in col:
        print(c)

def main():
    PATH_FILES =  "../data/FEBRL/"
    PATH_IMAGES =  "../images/"
    PATH_OUTPUT =  "../output/"

    np.random.seed(42)

    ## if using their files
    #trainset = PATH_FILES + 'febrl3_UNSW.csv' 
    #testset = PATH_FILES + 'febrl4_UNSW.csv'
    
    ## if generating new files
    trainset = gd.generate_files(PATH_FILES, "train")
    testset = gd.generate_files(PATH_FILES, "test")
    #trainset = PATH_FILES + 'febrl3_rep.csv' 
    #testset = PATH_FILES + 'febrl4_rep.csv'

    df_train = load_file(trainset)
    df_test = load_file(testset)

    X_train, y_train = generate_train_sets(df_train)
    all_candidate_pairs = block_test_sets(df_test)
    X_test, y_test = generate_test_sets(df_test, all_candidate_pairs)

    df_precision, df_sensitivity, df_fscore, df_nb_false = train_test_models(X_train, y_train, X_test, y_test)

    pl.show_plot(df_precision, "Hyperparameter - Precision - Scheme A", "precision", "precision_schemeA_1.png")
    pl.show_plot(df_sensitivity, "Hyperparameter - Sensitivity - Scheme A", "sensitivity", "sensitivity_schemeA_1.png")
    pl.show_plot(df_fscore, "Hyperparameter - FScore - Scheme A", "fscore", "fscore_schemeA_1.png")
    pl.show_plot(df_nb_false, "Hyperparameter - Falses - Scheme A", "#falses", "falses_schemeA_1.png")

    df_precision.to_csv(PATH_OUTPUT + 'schemeA_precision_data.csv')
    df_sensitivity.to_csv(PATH_OUTPUT + 'schemeA_sensitivity_data.csv')
    df_fscore.to_csv(PATH_OUTPUT + 'schemeA_fscore_data.csv')
    df_nb_false.to_csv(PATH_OUTPUT + 'schemeA_nb_false_data.csv')

    df_bagging = bagging_model(X_train, y_train, X_test, y_test)
    
    df_bagging.to_csv(PATH_OUTPUT + 'schemeA_bagging_stack.csv')

if __name__ == "__main__":
    main()
