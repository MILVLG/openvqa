# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import json, pickle
import numpy as np
from collections import defaultdict


def eval(__C, dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, valid=False):
    result_eval_file = result_eval_file + '.txt'

    ans_size = dataset.ans_size

    result_eval_file_fs = open(result_eval_file, 'w')
    for qix in range(dataset.data_size):
        result_eval_file_fs.write(dataset.ix_to_ans[ans_ix_list[qix]])
        result_eval_file_fs.write("\n")
    result_eval_file_fs.close()


    if __C.TEST_SAVE_PRED:
        print('Save the prediction vector to file: {}'.format(ensemble_file))

        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred': pred_list[qix],
            'qid': qix
        } for qix in range(dataset.data_size)]
        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


    if valid:
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]

        true_answers = []
        with open(ques_file_path, 'r') as f:
            questions = json.load(f)['questions']
            for ques in questions:
                true_answers.append(ques['answer'])

        correct_by_q_type = defaultdict(list)

        # Load predicted answers
        predicted_answers = []
        with open(result_eval_file, 'r') as f:
            for line in f:
                predicted_answers.append(line.strip())

        num_true, num_pred = len(true_answers), len(predicted_answers)
        assert num_true == num_pred, 'Expected %d answers but got %d' % (
            num_true, num_pred)

        for i, (true_answer, predicted_answer) in enumerate(zip(true_answers, predicted_answers)):
            correct = 1 if true_answer == predicted_answer else 0
            correct_by_q_type['Overall'].append(correct)
            q_type = questions[i]['program'][-1]['function']
            correct_by_q_type[q_type].append(correct)

        print('Write to log file: {}'.format(log_file))
        logfile = open(log_file, 'a+')
        q_dict = {}
        for q_type, vals in sorted(correct_by_q_type.items()):
            vals = np.asarray(vals)
            q_dict[q_type] = [vals.sum(), vals.shape[0]]
            # print(q_type, '%d / %d = %.2f' % (vals.sum(), vals.shape[0], 100.0 * vals.mean()))
            # logfile.write(q_type + ' : ' + '%d / %d = %.2f\n' % (vals.sum(), vals.shape[0], 100.0 * vals.mean()))

        # Score Summary
        score_type = ['Overall', 'Count', 'Exist', 'Compare_Numbers', 'Query_Attribute', 'Compare_Attribute']
        compare_numbers_type = ['greater_than', 'less_than']
        query_attribute_type = ['query_color', 'query_material', 'query_shape', 'query_size']
        compare_attribute_type =  ['equal_color', 'equal_integer', 'equal_material', 'equal_shape', 'equal_size']
        score_dict = {}
        score_dict['Overall'] = q_dict['Overall']
        score_dict['Count'] = q_dict['count']
        score_dict['Exist'] = q_dict['exist']

        correct_num, total_num = 0, 0
        for q_type in compare_numbers_type:
            correct_num += q_dict[q_type][0]
            total_num += q_dict[q_type][1]
        score_dict['Compare_Numbers'] = [correct_num, total_num]

        correct_num, total_num = 0, 0
        for q_type in query_attribute_type:
            correct_num += q_dict[q_type][0]
            total_num += q_dict[q_type][1]
        score_dict['Query_Attribute'] = [correct_num, total_num]

        correct_num, total_num = 0, 0
        for q_type in compare_attribute_type:
            correct_num += q_dict[q_type][0]
            total_num += q_dict[q_type][1]
        score_dict['Compare_Attribute'] = [correct_num, total_num]

        for q_type in score_type:
            val, tol = score_dict[q_type]
            print(q_type, '%d / %d = %.2f' % (val, tol, 100.0 * val / tol))
            logfile.write(q_type + ' : ' + '%d / %d = %.2f\n' % (val, tol, 100.0 * val / tol))

        logfile.write("\n")
        logfile.close()


