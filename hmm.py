import pandas as pd
import numpy as np

########### Helper Codes for: Q2(a) ###########
def output_prob(in_train_filename, output_probs_filename, delta):
    ### Extracting all the data from the training set
    trainData = []
    curr = []
    with open(in_train_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                trainData.append(curr)
                curr = []
            else:
                wordtag = l.split("\t")
                wordtag[1] = wordtag[1][:-1]
                curr.append(tuple(wordtag))

    ###  Process the counts in the following way
    # {tag1: {                                  > each unique tag
    #   numAppeared: n,                         > tracks the num of times this tag appears
    #   "words": {word1: x, word2: y,...}       > tracks the words with this tag, and their num of appearences
    # }}
    dictOfTags = {}
    
    for tweet in trainData:
        for wordtag in tweet:
            word = wordtag[0]
            tag = wordtag[1]

            if tag not in dictOfTags:
                dictOfTags[tag] = {"numAppeared": 1, "words": {word: 1}}
            else: 
                dictOfTags[tag]["numAppeared"] += 1
                if word not in dictOfTags[tag]["words"]:
                    dictOfTags[tag]["words"][word] = 1
                else:
                    dictOfTags[tag]["words"][word] += 1

    ### Get the total number of unique words
    num_words = len({word for tag in dictOfTags.values() for word in tag["words"]})

    for tag in dictOfTags:
        yj = dictOfTags[tag]["numAppeared"] # count(y = j)
        # UNSEEN WORDS: let "WordDoesNotExist" be the placeholder for unseen words
        # if a word does not existing in the train set: count(y = j → x = w) = 0 
        dictOfTags[tag]["words"]["WordDoesNotExist"] = delta / (yj + delta * (num_words + 1)) # 
        for word in dictOfTags[tag]["words"]:
            yjxw = dictOfTags[tag]["words"][word] # count(y = j → x = w) 

            bjw = (yjxw + delta) / (yj + delta*(num_words + 1))
            dictOfTags[tag]["words"][word] = bjw # replaces the count of the word to the prob of the word
    
    ### Creating the final output
    result = {"tag": [], "word": [], "prob": []}

    for tag in dictOfTags:
        for word in dictOfTags[tag]["words"]:
            bjw = dictOfTags[tag]["words"][word]
            result["tag"].append(tag)
            result["word"].append(word)
            result["prob"]. append(bjw)
    
    ### Create a dataframe as such:
    # Tag | Word | Probability is stored
    output_probs = pd.DataFrame.from_dict(result)
    output_probs.to_csv(output_probs_filename, index=False) 

########### Helper Codes for: Q4(a) ###########
def trans_prob(in_train_filename, trans_probs_filename, in_tags_filename, delta):
     ### Extracting all the data from the training set
    trainData = []
    curr = []
    with open(in_train_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                trainData.append(curr)
                curr = []
            else:
                wordtag = l.split("\t")
                wordtag[1] = wordtag[1][:-1]
                curr.append(tuple(wordtag))
    
    ### Creating all transition pairs
    # This data will be stored as a dataframe as such:
    # From  | To
    # START | tag1
    # tag1  | tag2 ...

    transitionsPairs = pd.DataFrame(columns = ["From", "To"])
    for tweet in trainData:
        for wordNum in range(len(tweet)): # Using this to be able to track the first and last word.
            toTag = tweet[wordNum][1]
            if wordNum == 0: # If this is the first word
                # Generate the START -> tag pair
                pair = {"From": ["START"], "To": [toTag]} 
            else: # A previous tag has been present
                # Generate the tag -> tag pair
                fromTag = tweet[wordNum - 1][1] # extracts the previous tag
                pair = {"From": [fromTag], "To": [toTag]}

            transitionsPairs = transitionsPairs.append(pd.DataFrame.from_dict(pair), ignore_index=True) # Records the pair

            if wordNum == len(tweet) - 1: # If this is the last word
                # Generate tag -> END pair
                lastpair = {"From": [toTag], "To": ["END"]}
                transitionsPairs = transitionsPairs.append(pd.DataFrame.from_dict(lastpair), ignore_index=True)
    
    ### Converting transition Counts to Probabilities
    # Count(fromTag > toTag)
    transitionCount = transitionsPairs.groupby(['From', 'To']).size().reset_index()
    transitionCount.columns = ["From", "To", "CountOfTags"]
    uniqueFromTags_trained = transitionCount["From"].unique()

    # Count(fromTag)
    fromTags = transitionCount.drop(columns = "To")
    fromTags = transitionCount.groupby(['From']).size().to_frame('CountOfFromTags')
    fromTags = fromTags.reset_index()

    ### Extracting all possible unique tags
    uniqueTags = []
    with open(in_tags_filename, "r") as fin:
        for l in fin:
            tag = l[:-1]
            uniqueTags += [tag]
    uniqueTagsWithStart = uniqueTags + ["START"]
    uniqueTagswithEnd = uniqueTags + ["END"]
    numUniqueTags = len(uniqueTags) + 2

    ### Converting Counts into Transition Probabilities
    denom = delta * (numUniqueTags+1)

    # Create a datafram to initialize the combinations of transitions
    transitionProb = transitionCount.copy()
    transitionProb = transitionCount.groupby(["From", "To"])['CountOfTags'].sum().rename("ProbabilityOfTags")
    transitionProb = (transitionProb + delta) / (transitionProb.groupby(level=0).sum() + denom)
    transitionProb.columns = ["From", "To", "ProbabilityOfTags"]
    transitionProb = transitionProb.reset_index()

    for i in uniqueTagsWithStart: # Goes through all possible unique tag that could be the From tag
        # Creating unseen combinations
        if i not in uniqueFromTags_trained:
            for j in uniqueTagswithEnd:
                probOfTag = delta / denom # smoothing as per Qn 2
                pair = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                transitionProb = transitionProb.append(pd.DataFrame.from_dict(pair), ignore_index=True)
        
        # For each From Tag, get all the possible To Tags
        uniqueTo = transitionCount[transitionCount["From"] == i]["To"].unique()
        if i != "START":
            possibleTo = uniqueTagswithEnd # The next tag can be any of the unique tags + "END" tag
        elif i == "START":
            possibleTo = uniqueTags # The next tag can only be only of another unique tag.
        
        for j in possibleTo: 
            if j not in uniqueTo:
                countOfFrom = fromTags[fromTags["From"]==i].iloc[0]["CountOfFromTags"]
                probOfTag = delta / (countOfFrom + denom)
                pair = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                transitionProb = transitionProb.append(pd.DataFrame.from_dict(pair), ignore_index = True)
    
    transitionProb.to_csv(trans_probs_filename, index=None)
    
    return transitionProb

########### Helper Codes for: Q4(b) ###########
### To transform output_prob and trans_prob into a nested dictionary for easy access.
def toDict(transdf, a, b):
        from collections import defaultdict
        d = defaultdict(dict)

        for i, row in transdf.iterrows():
            d[str(row[a])][str(row[b])] = row.drop([a, b]).to_dict()

        transDict = dict(d)

        return transDict

def initializationStep(states, trans_prob, output_prob, tweet):
    ### This aims to achieve the following
    # for state 1 to N:
    #   viterbi[s, 1] <- initialProb * bj
    #   backpointer <- 0

    ### Looking at START to tag1
    initialProb = trans_prob["START"]
    firstWord = tweet[0]

    numStatesN = len(states)        
    numObservationsT = len(tweet)

    ### Create probability matrix Lattice
    # where columns are observables (words of a sentence in the same sequence as in sentence)
    # & rows as hidden states(all possible POS Tags are known)

    # Initializing all placeholders as 0 first
    vertibiS1 = np.zeros(shape=(numObservationsT, numStatesN))
    backpointer = np.zeros(shape=(numObservationsT, numStatesN))

    # Iterate through each possible state (set of possible tags)
    for i, state in enumerate(states):
        # Transistion Prob
        probOfStartingAtState = initialProb[state]
        aij = probOfStartingAtState["ProbabilityOfTags"]

        # Output Prob
        if firstWord in output_prob: # Word existed in training data
            probsOfWord = output_prob[firstWord] 
        else: # Otherwise, use the WordDoesNotExist smoothing probability
            probsOfWord = output_prob["WordDoesNotExist"]
        
        if state in probsOfWord: # Extract Probability from Nested Dictionary
            bj = probsOfWord[state]['prob']
        else:
            probsOfWord = output_prob["WordDoesNotExist"]
            bj = probsOfWord[state]['prob']
        
        prob = aij * bj
        vertibiS1[0][i] = prob # Initialized the 1 - N state probs, and backpointer remains 0

    return [vertibiS1, backpointer]

def recusionStep(states, trans_prob, output_prob, sequence, viterbiST, backpointer):
    ### Computes the Recursion Step for each 2 to T words, and each 1 to N states

    ### Helper Function: Finds arg max and max for each individual aij and viterbiST[k]
    def findMax(trans_prob, state, states, index, stateIndex, b_st, viterbiST):
        viterbiST_kminus1 = viterbiST[index - 1]

        argMax = -1
        maxVal = -1

        for FromIndex, From in enumerate(states):
            From_prob = trans_prob[From]

            # get From -> state probability
            state_prob = From_prob[state]
            aFromTo = state_prob["ProbabilityOfTags"]

            # get previous viterbiST
            viterbiST_kminus1_From = viterbiST_kminus1[FromIndex]

            # calculate result
            viterbiSTResult = viterbiST_kminus1_From*aFromTo*b_st
            
            if viterbiSTResult > maxVal:
                maxVal = viterbiSTResult
                argMax = FromIndex

        return [maxVal, argMax]

    lastIndex = len(sequence) - 1

    for index, word in enumerate(sequence):
                # Check if word existed in training data
                if word in output_prob:
                    probsOfWord = output_prob[word]
                else: 
                    probsOfWord = output_prob["WordDoesNotExist"] # If not use WordDoesNotExist smoothing probability
                
                if index != 0:
                    for stateIndex, state in enumerate(states):
                        # Check if state exists in training data
                        if state in probsOfWord:
                            b_st = probsOfWord[state]['prob']
                        else:
                            probsOfNoWord = output_prob["WordDoesNotExist"] # If not use WordDoesNotExist smoothing probability
                            b_st = probsOfNoWord[state]['prob']
            
                        # finding max and argmax
                        max_ArgMax_result = findMax(trans_prob, state, states, index, stateIndex, b_st, viterbiST)
                        viterbiST[index][stateIndex] = max_ArgMax_result[0]
                        backpointer[index][stateIndex] = max_ArgMax_result[1]
                    
                    #if all(i <= 0.00001 for i in viterbiST[index]):
                    #    viterbiST[index] = [i * 10000 for i in viterbiST[index]]
    return [viterbiST, backpointer]

def backPointer(viterbiST, backpointer, sequence, states):
    # Get last state and index
    len_of_sequence = len(sequence)
    viterbiST_list = viterbiST[len_of_sequence-1]
    curr_index = np.argmax(viterbiST_list)
    state_result = [states[curr_index]]
    path = [curr_index]
    prob_path = [viterbiST[len_of_sequence-1][curr_index]]

    # access the relevant state
    for index in range(len_of_sequence-1, 0, -1):
        
        # Get index
        curr_index = int(backpointer[index][curr_index])

        # Get state
        state_result += [states[curr_index]]

        # Get path
        path += [curr_index]

        # Get prob
        prob_path += [viterbiST[len_of_sequence-1][curr_index]]
    
    # reverse to get actual result
    list.reverse(state_result)
    list.reverse(path)
    list.reverse(prob_path)
    
    return [state_result, path, prob_path]

### To put together the 3 steps 
def viterbiAlgo(states,trans_prob,output_prob,sequence):
     # Initialization Step
     init_viterbiST, init_backpointer = initializationStep(states, trans_prob, output_prob, sequence)
     
     # Recursion Step
     viterbiST, backpointer = recusionStep(states, trans_prob, output_prob, sequence, init_viterbiST, init_backpointer)
    
     # Calculating the Backpointers
     backpointer_result = backPointer(viterbiST, backpointer, sequence, states)
    
     return backpointer_result

########### Helper Codes for: Q5(a) ###########
### To redefine the output probabilities to account for the @user and http inputs by grouping them
def output_prob2(in_train_filename, output_probs_filename, delta):
    ### Extracting all the data from the training set
    trainData = []
    curr = []
    with open(in_train_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                trainData.append(curr)
                curr = []
            else:
                wordtag = l.split("\t")
                wordtag[1] = wordtag[1][:-1]
                word = wordtag[0]
                tag = wordtag[1]
                changed_word = word.lower()
                if changed_word.startswith('@user') :
                    changed_word = '@user'
                elif changed_word.startswith('http') :
                    changed_word = 'http'
                word = changed_word
                curr.append(tuple(wordtag))

    ###  Process the counts in the following way
    # {tag1: {                                  > each unique tag
    #   numAppeared: n,                         > tracks the num of times this tag appears
    #   "words": {word1: x, word2: y,...}       > tracks the words with this tag, and their num of appearences
    # }}
    dictOfTags = {}
    
    for tweet in trainData:
        for wordtag in tweet:
            word = wordtag[0]
            tag = wordtag[1]

            if tag not in dictOfTags:
                dictOfTags[tag] = {"numAppeared": 1, "words": {word: 1}}
            else: 
                dictOfTags[tag]["numAppeared"] += 1
                if word not in dictOfTags[tag]["words"]:
                    dictOfTags[tag]["words"][word] = 1
                else:
                    dictOfTags[tag]["words"][word] += 1

    ### Get the total number of unique words
    num_words = len({word for tag in dictOfTags.values() for word in tag["words"]})

    for tag in dictOfTags:
        yj = dictOfTags[tag]["numAppeared"] # count(y = j)
        # UNSEEN WORDS: let "WordDoesNotExist" be the placeholder for unseen words
        # if a word does not existing in the train set: count(y = j → x = w) = 0 
        dictOfTags[tag]["words"]["WordDoesNotExist"] = delta / (yj + delta * (num_words + 1)) # 
        for word in dictOfTags[tag]["words"]:
            yjxw = dictOfTags[tag]["words"][word] # count(y = j → x = w) 

            bjw = (yjxw + delta) / (yj + delta*(num_words + 1))
            dictOfTags[tag]["words"][word] = bjw # replaces the count of the word to the prob of the word
    
    ### Creating the final output
    result = {"tag": [], "word": [], "prob": []}

    for tag in dictOfTags:
        for word in dictOfTags[tag]["words"]:
            bjw = dictOfTags[tag]["words"][word]
            result["tag"].append(tag)
            result["word"].append(word)
            result["prob"]. append(bjw)
    
    ### Create a dataframe as such:
    # Tag | Word | Probability is stored
    output_probs = pd.DataFrame.from_dict(result)
    output_probs.to_csv(output_probs_filename, index=False) 

def trans_prob2(in_train_filename, trans_probs_filename, in_tags_filename, delta):
     ### Extracting all the data from the training set
    trainData = []
    curr = []
    with open(in_train_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                trainData.append(curr)
                curr = []
            else:
                wordtag = l.split("\t")
                wordtag[1] = wordtag[1][:-1]
                word = wordtag[0]
                tag = wordtag[1]
                changed_word = word.lower()
                if changed_word.startswith('@user') :
                    changed_word = '@user'
                elif changed_word.startswith('http') :
                    changed_word = 'http'
                word = changed_word
                curr.append(tuple(wordtag))

    
    ### Creating all transition pairs
    # This data will be stored as a dataframe as such:
    # From  | To
    # START | tag1
    # tag1  | tag2 ...

    transitionsPairs = pd.DataFrame(columns = ["From", "To"])
    for tweet in trainData:
        for wordNum in range(len(tweet)): # Using this to be able to track the first and last word.
            toTag = tweet[wordNum][1]
            if wordNum == 0: # If this is the first word
                # Generate the START -> tag pair
                pair = {"From": ["START"], "To": [toTag]} 
            else: # A previous tag has been present
                # Generate the tag -> tag pair
                fromTag = tweet[wordNum - 1][1] # extracts the previous tag
                pair = {"From": [fromTag], "To": [toTag]}

            transitionsPairs = transitionsPairs.append(pd.DataFrame.from_dict(pair), ignore_index=True) # Records the pair

            if wordNum == len(tweet) - 1: # If this is the last word
                # Generate tag -> END pair
                lastpair = {"From": [toTag], "To": ["END"]}
                transitionsPairs = transitionsPairs.append(pd.DataFrame.from_dict(lastpair), ignore_index=True)
    
    ### Converting transition Counts to Probabilities
    # Count(fromTag > toTag)
    transitionCount = transitionsPairs.groupby(['From', 'To']).size().reset_index()
    transitionCount.columns = ["From", "To", "CountOfTags"]
    uniqueFromTags_trained = transitionCount["From"].unique()

    # Count(fromTag)
    fromTags = transitionCount.drop(columns = "To")
    fromTags = transitionCount.groupby(['From']).size().to_frame('CountOfFromTags')
    fromTags = fromTags.reset_index()

    ### Extracting all possible unique tags
    uniqueTags = []
    with open(in_tags_filename, "r") as fin:
        for l in fin:
            tag = l[:-1]
            uniqueTags += [tag]
    uniqueTagsWithStart = uniqueTags + ["START"]
    uniqueTagswithEnd = uniqueTags + ["END"]
    numUniqueTags = len(uniqueTags) + 2

    ### Converting Counts into Transition Probabilities
    denom = delta * (numUniqueTags+1)

    # Create a datafram to initialize the combinations of transitions
    transitionProb = transitionCount.copy()
    transitionProb = transitionCount.groupby(["From", "To"])['CountOfTags'].sum().rename("ProbabilityOfTags")
    transitionProb = (transitionProb + delta) / (transitionProb.groupby(level=0).sum() + denom)
    transitionProb.columns = ["From", "To", "ProbabilityOfTags"]
    transitionProb = transitionProb.reset_index()

    for i in uniqueTagsWithStart: # Goes through all possible unique tag that could be the From tag
        # Creating unseen combinations
        if i not in uniqueFromTags_trained:
            for j in uniqueTagswithEnd:
                probOfTag = delta / denom # smoothing as per Qn 2
                pair = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                transitionProb = transitionProb.append(pd.DataFrame.from_dict(pair), ignore_index=True)
        
        # For each From Tag, get all the possible To Tags
        uniqueTo = transitionCount[transitionCount["From"] == i]["To"].unique()
        if i != "START":
            possibleTo = uniqueTagswithEnd # The next tag can be any of the unique tags + "END" tag
        elif i == "START":
            possibleTo = uniqueTags # The next tag can only be only of another unique tag.
        
        for j in possibleTo: 
            if j not in uniqueTo:
                countOfFrom = fromTags[fromTags["From"]==i].iloc[0]["CountOfFromTags"]
                probOfTag = delta / (countOfFrom + denom)
                pair = {"From": [i], "To": [j], "ProbabilityOfTags": [probOfTag]}
                transitionProb = transitionProb.append(pd.DataFrame.from_dict(pair), ignore_index = True)
    
    transitionProb.to_csv(trans_probs_filename, index=None)
    
    return transitionProb

# Implement the six functions below
def naive_predict(in_output_probs_filename2, in_test_filename, out_prediction_filename):

    ### Get output_probs from naive output probs.txt
    output_probs = pd.read_csv(in_output_probs_filename2)

    ### Creates a base of probabilities such that if the test word does not exist, this unseen prob will be used.
    #This is done by extracting the "WordDoesNotExist" from output probs.
    notFoundProb = output_probs[output_probs["word"] == "WordDoesNotExist"].drop("word",axis=1).set_index("tag")

    ### Take in test data 
    testData = []
    curr = []
    with open(in_test_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                testData.append(curr)
                curr = []
            else:     
                word = l[:-1]
                changed_word = word.lower()
                if '@user' in changed_word :
                    changed_word = '@user'
                elif 'http' in changed_word :
                    changed_word = 'http'
                word = changed_word
                curr.append(word)

    with open(out_prediction_filename, "w") as f: # To write out the predictions
        for tweet in testData:
            for word in tweet:
                possibleTags = notFoundProb.copy()

                # Look into output probs gotten from training if the word exists in training set
                trainedProb = output_probs[output_probs.word == word].set_index("tag")
                possibleTags.update(trainedProb)

                # Get the tag with the highest Likelihood
                MLE_tag = possibleTags.idxmax().prob
                f.write(MLE_tag + "\n")
            f.write("\n") # Adds space after every Tweet

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    # From Q2: P(x = w|y = j)
    # RHS of equation: j∗ = argmaxP(y = j|x = w).
    # Product Rule: P(y = j|x = w) = P(x = w|y = j) * P(y = j) / P(x = w)
    # Since we are finding the argmax of the tag, P(x = w) can be omiited. And we only need to find P(y = j)

    ### Get output probs
    output_probs = pd.read_csv(in_output_probs_filename)
    
    ### Calculating P(y = j)
    trainData = []
    curr = []
    with open(in_train_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                trainData.append(curr)
                curr = []
            else:
                wordtag = l.split("\t")
                wordtag[1] = wordtag[1][:-1]
                word = wordtag[0]
                tag = wordtag[1]
                changed_word = word.lower()
                if changed_word.startswith('@user') :
                    changed_word = '@user'
                elif changed_word.startswith('http') :
                    changed_word = 'http'
                word = changed_word
                curr.append(tuple(wordtag))

    # Getting the counts of each tag in the trining data, storing them as:
    # {tag1: numOfAppearance,
    # tag2: numOfAppearance, ...}
    countyj = {} 
    for tweet in trainData:
        for wordtag in tweet:
            tag = wordtag[1]
            if tag not in countyj:
                countyj[tag] = 1
            else:
                countyj[tag] += 1
    
    num_tags = sum(countyj.values()) #retrives the denominator for computing the prob
    
    ### Converting the counts of countyj into probabilities
    # probyj = countyj / num_tags
    probyj = {}
    for k,v in countyj.items():
        probyj[k] = v/num_tags

    ### Take in test data 
    testData = []
    curr = []
    with open(in_test_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                testData.append(curr)
                curr = []
            else: 
                word = l[:-1]
                changed_word = word.lower()
                if '@user' in changed_word :
                    changed_word = '@user'
                elif 'http' in changed_word :
                    changed_word = 'http'
                word = changed_word
                curr.append(word)
    
     ### Creates a base of probabilities such that if the test word does not exist, this unseen prob will be used.
    notFoundProb = output_probs[output_probs["word"] == "WordDoesNotExist"].drop("word",axis=1).set_index("tag")
    
    with open(out_prediction_filename, "w") as f:
        for tweet in testData:
            for word in tweet:
                tagProb = notFoundProb.copy()
                tagProb.update(output_probs[output_probs.word == word].set_index("tag"))

                # Switching up P(x = w|y = j) to P(y = j|x = w)
                for index, row in tagProb.iterrows():
                    new_prob = row.prob * probyj[index]
                    tagProb.loc[index].prob = new_prob
                
                # Getting the argmax of P(y = j|x = w)
                argmax_tag = tagProb.idxmax().prob
                f.write(argmax_tag + "\n")
            f.write("\n")

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    in_output_probs = pd.read_csv(in_output_probs_filename)
    in_trans_probs = pd.read_csv(in_trans_probs_filename)

    testData = []
    curr = []
    with open(in_test_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                testData.append(curr)
                curr = []
            else:
                curr.append(l[:-1])

    states = []
    with open(in_tags_filename, "r") as f:
        for line in f:
            data = line[:-1] # Remove the newline character
            states += [data]


    # Convert transition and output probs to dict
    output_prob = toDict(in_output_probs,"word","tag")
    trans_prob = toDict(in_trans_probs,"From","To")

    # Initialise 3 lists to save the results for each tweet in the test data
    state_result = []
    path = []
    prob_path = []

    # iterate through all tweets
    for tweet in testData:

        viterbi_predictions = viterbiAlgo(states,trans_prob,output_prob,tweet)

        state_result += viterbi_predictions[0]
        path += viterbi_predictions[1]
        prob_path += viterbi_predictions[2]

    # Write predictions to file
    with open(out_predictions_filename, "w") as f:
        for prediction in state_result:
            f.write(prediction + "\n")
    
def viterbi_predict2(in_tags_filename, in_trans_probs_filename2, in_output_probs_filename2, in_test_filename,
                     out_predictions_filename):
    in_output_probs = pd.read_csv(in_output_probs_filename2)
    in_trans_probs = pd.read_csv(in_trans_probs_filename2)

    testData = []
    curr = []
    with open(in_test_filename, "r") as fin:
        for l in fin:
            if l == "\n":
                testData.append(curr)
                curr = []
            else:
                word = l[:-1]
                changed_word = word.lower()
                if '@user' in changed_word :
                    changed_word = '@user'
                elif 'http' in changed_word :
                    changed_word = 'http'
                word = changed_word
                curr.append(word)

    states = []
    with open(in_tags_filename, "r") as f:
        for line in f:
            data = line[:-1] # Remove the newline character
            states += [data]


    # Convert transition and output probs to dict
    output_prob = toDict(in_output_probs,"word","tag")
    trans_prob = toDict(in_trans_probs,"From","To")

    # Initialise 3 lists to save the results for each tweet in the test data
    state_result = []
    path = []
    prob_path = []

    # iterate through all tweets
    for tweet in testData:
    
        viterbi_predictions = viterbiAlgo(states,trans_prob,output_prob,tweet)

        state_result += viterbi_predictions[0]
        path += viterbi_predictions[1]
        prob_path += viterbi_predictions[2]

    # Write predictions to file
    with open(out_predictions_filename, "w") as f:
        for prediction in state_result:
            f.write(prediction + "\n")

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

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    output_prob(in_train_filename, naive_output_probs_filename, 0.1)

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    trans_prob(in_train_filename, trans_probs_filename, in_tags_filename, 0.1)
    output_prob(in_train_filename, output_probs_filename, 0.1)

    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    output_prob2(in_train_filename, output_probs_filename2, 0.1)
    trans_prob2(in_train_filename, trans_probs_filename2, in_tags_filename, 0.1)

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                    viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    #in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    #in_tag_filename     = f'{ddir}/twitter_tags.txt'
    #out_trans_filename  = f'{ddir}/trans_probs4.txt'
    #out_output_filename = f'{ddir}/output_probs4.txt'
    #max_iter = 10
    #seed     = 8
    #thresh   = 1e-4
    #forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
    #                 max_iter, seed, thresh)

    #trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    #output_probs_filename3 = f'{ddir}/output_probs3.txt'
    #viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    #viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
    #                 viterbi_predictions_filename3)
    #correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    #print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    #trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    #output_probs_filename4 = f'{ddir}/output_probs4.txt'
    #viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    #viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
    #                 viterbi_predictions_filename4)
    #correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    #print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    #in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    #in_tag_filename     = f'{ddir}/cat_states.txt'
    #out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    #out_output_filename = f'{ddir}/cat_output_probs.txt'
    #max_iter = 1000000
    #seed     = 8
    #thresh   = 1e-4
    #forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
    #                 max_iter, seed, thresh)

    #in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    #in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    #in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    #in_states_filename       = f'{ddir}/cat_states.txt'
    #predictions_filename     = f'{ddir}/cat_predictions.txt'
    #cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
    #            predictions_filename)

    #in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    #ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    #print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()
