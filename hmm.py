import random
import pandas as pd
import numpy as np

# return dict due to complications when opening file
def countOutputNumerator(twitter_train_tag,twitter_tags) :
    
    # numerator -> count using train and tags, see how many respective word is pegged to that tag
    with open(twitter_train_tag) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
    
    #for denominator
    with open(twitter_tags) as fin:
        twitter_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    twitter_tags_dict = { tag : 0 for tag in twitter_tags}
    #iterate through the predicted tags, store in key-value dict
    dict = {}
    dict_count = {} #denominator
    dict_prob = {} #probability
    delta = 1 # can also be 0.01, 0.1, 1, 10
    for string in predicted_tags :
        string_list = string.split("\t")
        valuestring = string_list[0].lower()
        #filter all the users into the same key
        if valuestring.startswith('@user') :
            valuestring = '@user'
        #filter all the http into the http
        if valuestring.startswith('http') :
            valuestring = 'http'
        twitter_tags_dict[string_list[1]] += 1 #counting denom
        #empty list
        if dict.get(valuestring) == None:
            dict[valuestring] = [string_list[1]]
            dict_count[valuestring] = {}
            dict_count[valuestring][string_list[1]] = 1
        #not empty list
        else :
            dict[valuestring].append(string_list[1])
            # value as a dictionary
            if dict_count[valuestring].get(string_list[1]) :
                dict_count[valuestring][string_list[1]] += 1
            else :
                dict_count[valuestring][string_list[1]] = 1

    dict_prob = dict_count
    for word in dict_prob :
        for tag in dict_prob[word] :
            numerator = dict_count[word][tag] + delta
            denominator = twitter_tags_dict[tag] + delta * (len(dict) +1)
            dict_prob[word][tag] = numerator/denominator
    # print(dict_prob)
    return dict_prob

def dictToTxt(file) :
    with open('naive_output_probs.txt','w') as data :
        for k,v in file.items() :
            data.write("%s:%s\n" % (k,v))
    return data

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):

    #reading twitter_dev_no_tags
    with open(in_test_filename) as fin:
        test_file = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
    
    # print(in_output_probs_filename)
    predict_tag = []
    for word in test_file :
        highest = 0
        tag = ''
        found = False # if the word doesnt match , take away the last letter from the word and check again
        changed_word = word.lower()
        #filter all the users into the same key
        if changed_word.startswith('@user') :
            changed_word = '@user'
        #filter all the http into the http
        if changed_word.startswith('http') :
            changed_word = 'http'
        while found == False :
            if in_output_probs_filename.get(changed_word) != None : 
                for key in in_output_probs_filename[changed_word] :
                    curr_prob = in_output_probs_filename[changed_word][key]
                    if curr_prob > highest : 
                        highest = curr_prob
                        tag = key
                found = True
            elif len(changed_word) > 1:
                changed_word = changed_word[:-1]
            # else :
            #     found = True
            else : # if word does not exist with the training dataset, use random
                tag = random.choice(['@',',','L','~','&','S','N','A','G','$','V','R','X','E','T','M','D','O','Z','!','^','U','P','Y'])
                found = True
        predict_tag.append(tag)
    # print(len(predict_tag))
    text = open("naive_predictions.txt","w")
    for element in predict_tag :
        text.write(element + "\n")
    text.close()
    return text
    pass

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    # j* = argmax P(y = j| x = w)
    # ie. argmax P(y = j, x = w) as denominator P(x = w) constant for each j
    # multiply output probabilities from naive_output_probs.txt by P(y = j)

    output = ""

    # get all tags as list
    with open('twitter_tags.txt') as tags_file:
        tags = [l.strip() for l in tags_file.readlines() if len(l.strip()) != 0]

    # get P(y = j)
    with open(in_train_filename) as train_file:
        train_tags = [l.strip()[-1] for l in train_file.readlines() if len(l.strip()) != 0]
        prob_tags = dict()
        for tag in tags:
            prob_tags[tag] = train_tags.count(tag) / len(train_tags)
    
    with open(in_test_filename) as test_file:
        for l in test_file.readlines():
            l = l.strip().lower()

            # Improvement: preprocessing by clustering
            if '@user' in l:
                l = '@user'

            if 'http' in l:
                l = 'http'
            
            if (len(l) == 0):
                # case: empty line
                output += '\n'
            else:
                # case: word to be processed
                probabilities = dict()
                if (in_output_probs_filename.get(l, None) != None):
                    # case: word found in output prob dict
                    for key in in_output_probs_filename[l]:
                        probabilities[key] = in_output_probs_filename[l][key] * prob_tags[key]
                        highest_prob_tag = max(probabilities, key=lambda k: probabilities[k])
                else:
                    # case: word not found in output prob dict
                    # choose highest probability tag
                    highest_prob_tag = max(prob_tags, key=lambda k: prob_tags[k])

                output += highest_prob_tag + '\n'
    text = open(out_prediction_filename,"w")
    text.write(output)

def transitionProb(twitter_train_tag,twitter_tags, trans_probs_filename):
    ### TRAINING ###
    # Generating the Tweets and their Tags
    traintweets = []
    current = []
    with open(twitter_train_tag, "r") as fin:
        for l in fin:
            if l == "\n": #if the line is empty, end of tweet. 
                traintweets.append(current)
                current = []
            else: # data to be added to current tweet. 
                traindata = l.split("\t") #splits WORD and TAG (seperated by a tab)
                traindata[1] = traindata[1][:-1]
                current.append(tuple(traindata))

    #Model the transition probabilities as a Dataframe
    transitionCount = pd.DataFrame(columns = ["From", "To"])
    for tweet in traintweets:
        for i in range(len(tweet)):
            tag = tweet[i][1]
            if i == 0: # First word in tweet: Start State > Tag
                fromToPlacement = {"From": ["START"], "To": [tag]} #matches the from and to
                temp = pd.DataFrame.from_dict(fromToPlacement) #transforms dictionary into df to be added into transitionCount
                transitionCount = transitionCount.append(temp, ignore_index = True)
            elif i == len(tweet) - 1: # Last word in tweet: Tag > End State
                fromToPlacement = {"From": [tag], "To": ["END"]} #matches the from and to
                temp = pd.DataFrame.from_dict(fromToPlacement) #transforms dictionary into df to be added into transitionCount
                transitionCount = transitionCount.append(temp, ignore_index = True)
            else: # One of the middle words in the tweet
                fromTag = tweet[i-1][1] # extract the previous word's tag
                fromToPlacement = {"From": [fromTag], "To": [tag]} #matches the from and to
                temp = pd.DataFrame.from_dict(fromToPlacement) #transforms dictionary into df to be added into transitionCount
                transitionCount = transitionCount.append(temp, ignore_index = True)
    
    #Using the dataframe, get all counts
    #Total Counts
    transitionCount = transitionCount.groupby(['From', 'To']).size().reset_index()
    transitionCount.columns = ["From", "To", "CountOfTags"]
    uniqueFromTags = transitionCount["From"].unique() #all unique tags in the training set
    #From Counts
    fromTags = transitionCount.drop(columns = "To") #removes the next from transitionCount
    fromTags = transitionCount.groupby(['From']).size().to_frame('CountOfTags')
    fromTags = fromTags.reset_index()

    # Getting a list of unique tags
    uniqueTags = []
    with open(twitter_tags, "r") as fin:
        for l in fin:
            tag = l[:-1]
            uniqueTags += [tag] #adds the tag into the list
    uniqueTagsWithStart = uniqueTags + ["START"]
    uniqueTagswithEnd = uniqueTags + ["END"]
    numUniqueTags = len(uniqueTags) + 2

    #Getting the Transition Probabilities
    sigma = 0.1
    denom = 0.1 * (numUniqueTags+1)
    transitionProb = transitionCount.copy() #replicating the dataframe
    transitionProb = transitionCount.groupby(["From", "To"])['CountOfTags'].sum().rename("ProbabilityOfTags")
    transitionProb = (transitionProb + sigma) / (transitionProb.groupby(level=0).sum() + denom)
    transitionProb.columns = ["From", "To", "ProbabilityOfTags"]
    transitionProb = transitionProb.reset_index()

    for i in uniqueTagsWithStart:
        if i not in uniqueFromTags: #if the tag is not in the training data
            for j in uniqueTagswithEnd:
                probOfTag = sigma / denom
                #Same method as above
                fromToPlacement = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                temp = pd.DataFrame.from_dict(fromToPlacement)
                transitionProb = transitionProb.append(temp, ignore_index = True)
        #for each From tag, get a list of the possible unique To tags
        uniqueTo = transitionCount[transitionCount["From"] == i]["To"].unique()
        if i != "START":
            enumTag = uniqueTagswithEnd
        elif i == "START":
            enumTag = uniqueTags
        for j in enumTag:
            if j not in uniqueTo:
                countOfFrom = fromTags[fromTags["From"]==i].iloc[0]["CountOfTags"]
                probOfTag = sigma / (countOfFrom + denom)
                fromToPlacement = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                temp = pd.DataFrame.from_dict(fromToPlacement)
                transitionProb = transitionProb.append(temp, ignore_index = True)
    
    transitionProb.to_csv(trans_probs_filename, index=None)
    return transitionProb

def outputAndTransitionProbs(twitter_train_tag,twitter_tags, trans_probs_filename):
    countOutputNumerator(twitter_train_tag,twitter_tags)
    transitionProb(twitter_train_tag,twitter_tags, trans_probs_filename)

def toDict(transdf, a, b):
        """
        Converts transition prob to a nested dict
        """
        from collections import defaultdict
        d = defaultdict(dict)

        for i, row in transdf.iterrows():
            d[str(row[a])][str(row[b])] = row.drop([a, b]).to_dict()

        transDict = dict(d)

        return transDict

def initializationStep(states, trans_prob, output_prob, sequence):

    # get START probabilities
    start_prob = trans_prob["START"]
    
    # Define first word
    firstWord = sequence[0]

    # Define statistics for pi and backpointer
    numLength = len(sequence)
    numState = len(states)

    # Creating pi and Backpointer
    pi = np.zeros(shape=(numLength, numState)) #initiates 0s for pi
    backpointer = np.zeros(shape=(numLength, numState)) #initiates 0s for backpointer

    # Iterate through states
    for i, state in enumerate(states):
        # get START -> state probability
        stateProb = start_prob[state]
        ao_v = stateProb['ProbabilityOfTag']

        # get state -> output probability given word
        ## Check if word exists in output probability
        if firstWord in output_prob:
            result_dict = output_prob[firstWord]
        else:
            result_dict = output_prob["NONE"]

        if state in result_dict:
            bv_x1 = result_dict[state]['ProbabilityOfTag']
        else:
            result_dict = output_prob["NONE"]
            bv_x1 = result_dict[state]['ProbabilityOfTag']

        # Calculate Prob
        prob = ao_v*bv_x1
        
        # Store in pi
        pi[0][i] = prob

    
    return [pi, backpointer]

def compute_viterbi(states, trans_prob, output_prob, sequence, pi, backpointer): #viterbi algo

    output_prob["NONE"] = ''

    def find_max(trans_prob, state, states, index, stateIndex, bv_xk, pi):
        # retrieve pi values
        pi_kminus1 = pi[index - 1]

        # set temp holder for results
        argMax = -1
        maxVal = -1

        # enumerate for u
        for priorIndex, prior in enumerate(states):

            # get prior probabilities
            prior_prob = trans_prob[prior]

            # get prior -> state probability
            state_prob = prior_prob[state]
            au_v = state_prob['ProbabilityOfTag']

            # get previous pi
            pi_kminus1_prior = pi_kminus1[priorIndex]

            # calculate result
            piResult = pi_kminus1_prior*au_v*bv_xk
            
            if piResult > maxVal:
                maxVal = piResult
                argMax = priorIndex

        return [maxVal, argMax]

    lastIndex = len(sequence) - 1

    for index, tweet in enumerate(sequence):

        for word in tweet:
            ## Check if word exists in output probability
            if word in output_prob:
                result_dict = output_prob[word]
            else:
                result_dict = output_prob["NONE"]

            # START is covered in zero states
            if index != 0:
                for stateIndex, state in enumerate(states):

                    # Check if state exists in word dict
                    if state in result_dict:
                        bv_xk = result_dict[state]['ProbabilityOfTag']
                    else:
                        result_dict_else = output_prob["NONE"]
                        bv_xk = result_dict_else[state]['ProbabilityOfTag']

                    # finding max and argmax
                    max_ArgMax_result = find_max(trans_prob, state, states, index, stateIndex, bv_xk, pi)
                    pi[index][stateIndex] = max_ArgMax_result[0]
                    backpointer[index][stateIndex] = max_ArgMax_result[1]

                # ensure that probability does not go to zero for super long tweets
                if all(i <= 0.00001 for i in pi[index]):
                    pi[index] = [i * 10000 for i in pi[index]]
    
    return [pi, backpointer]

def getBackPointer(pi, backpointer, sequence, states):
    # Get last state and index
    len_of_sequence = len(sequence)
    pi_list = pi[len_of_sequence-1]
    curr_index = np.argmax(pi_list)
    state_result = [states[curr_index]]
    path = [curr_index]
    prob_path = [pi[len_of_sequence-1][curr_index]]

    # access the relevant state
    for index in range(len_of_sequence-1, 0, -1):
        
        # Get index
        curr_index = int(backpointer[index][curr_index])

        # Get state
        state_result += [states[curr_index]]

        # Get path
        path += [curr_index]

        # Get prob
        prob_path += [pi[len_of_sequence-1][curr_index]]
    
    # reverse to get actual result
    list.reverse(state_result)
    list.reverse(path)
    list.reverse(prob_path)
    
    return [state_result, path, prob_path]

def run_viterbi(states,trans_prob,output_prob, sequence):
    """
    Given a sequence, the possible states, trans probs, and output probs, predicts tags for the sequence
    """
    # Initialise pi and backpointer, and compute results for START
    init_pi, init_backpointer = initializationStep(states, trans_prob, output_prob, sequence)

    # Compute viterbi for the remaining sequence
    pi, backpointer = compute_viterbi(states, trans_prob, output_prob, sequence, init_pi, init_backpointer)
    
    # get the backpointer results, which is a tuple of 3 items: the state_result, the path, and the prob_path
    backpointer_result = getBackPointer(pi, backpointer, sequence, states)
    
    return backpointer_result

def viterbi_predict(in_tags_filename, in_trans_probs_filename, prob_dict, in_test_filename, out_predictions_filename):
   
    # Import the relevant files
    test_data =[]
    current_test = []
    with open(in_test_filename, "r") as f:
        for line in f:
            if line == "\n": 
                test_data.append(current_test)
                current_test = []
            else: 
                current_test.append(line[:-1])

    states = []
    current_state = []
    with open(in_tags_filename, "r") as f:
        for line in f:
            if line == "\n": 
                states.append(current_state)
                current_state = []
            else:
                current_state.append(line[:-1])

    # Convert transition and output probs to dict
    output_prob = prob_dict
    trans_prob = toDict(pd.read_csv(in_trans_probs_filename),"From","To")
    print(trans_prob)

    # Initialise 3 lists to save the results for each tweet in the test data
    state_result = []
    path = []
    prob_path = []

    # iterate through all tweets
    for tweet in test_data:

        viterbi_predictions = run_viterbi(states,trans_prob,output_prob, test_data)

        state_result += viterbi_predictions[0]
        path += viterbi_predictions[1]
        prob_path += viterbi_predictions[2]

    # Write predictions to file
    with open(out_predictions_filename, "w") as f:
        for prediction in state_result:
            f.write(prediction + "\n")

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    pass

def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):
    pass


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''
    
    
    ddir = './' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'
    in_tag_filename = f'{ddir}/twitter_tags.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    
    prob_dict = countOutputNumerator(in_train_filename,in_tag_filename)
    dictToTxt(prob_dict)
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(prob_dict, in_test_filename, naive_prediction_filename)

    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(prob_dict, in_train_filename, in_test_filename, naive_prediction_filename2) # used prob_dict instead of naive_output_probs_filename
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    transitionProb(in_train_filename, in_tag_filename, trans_probs_filename)

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, prob_dict, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    # in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    # in_tag_filename     = f'{ddir}/twitter_tags.txt'
    # out_trans_filename  = f'{ddir}/trans_probs4.txt'
    # out_output_filename = f'{ddir}/output_probs4.txt'
    # max_iter = 10
    # seed     = 8
    # thresh   = 1e-4
    # forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
    #                  max_iter, seed, thresh)

    # trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    # output_probs_filename3 = f'{ddir}/output_probs3.txt'
    # viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
    #                  viterbi_predictions_filename3)
    # correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    # print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    # trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    # output_probs_filename4 = f'{ddir}/output_probs4.txt'
    # viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
    #                  viterbi_predictions_filename4)
    # correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    # print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    # in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    # in_tag_filename     = f'{ddir}/cat_states.txt'
    # out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    # out_output_filename = f'{ddir}/cat_output_probs.txt'
    # max_iter = 1000000
    # seed     = 8
    # thresh   = 1e-4
    # forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
    #                  max_iter, seed, thresh)

    # in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    # in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    # in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    # in_states_filename       = f'{ddir}/cat_states.txt'
    # predictions_filename     = f'{ddir}/cat_predictions.txt'
    # cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
    #             predictions_filename)

    # in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    # ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    # print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()