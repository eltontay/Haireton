import random

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
    delta = 10 # can also be 0.01, 0.1, 1, 10
    for string in predicted_tags :
        string_list = string.split("\t")
        valuestring = string_list[0].lower()
        twitter_tags_dict[string_list[1]] += 1
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
        for key in dict_count[valuestring] :
            numerator = dict_count[valuestring][key] + delta
            denominator = twitter_tags_dict[key] + delta * ((len(dict))+1)
            dict_count[valuestring][key]  = numerator/denominator
    return dict_count

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
    print(len(predict_tag))
    text = open("naive_predictions.txt","w")
    for element in predict_tag :
        text.write(element + "\n")
    text.close()
    return text
    pass

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    pass

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

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

    # naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    # naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    # correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    # print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    # trans_probs_filename =  f'{ddir}/trans_probs.txt'
    # output_probs_filename = f'{ddir}/output_probs.txt'

    # in_tags_filename = f'{ddir}/twitter_tags.txt'
    # viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    # viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
    #                 viterbi_predictions_filename)
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

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
