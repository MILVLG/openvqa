# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from collections import defaultdict
from tqdm import tqdm
import os.path
import glob
import json


class GQAEval:
    def __init__(self, __C, result_eval_file, ques_file_path, choices_path=None, EVAL_CONSISTENCY=False):
        ##### Files Loading
        ##########################################################################################

        # self.question_path = __C.QUESTION_PATH[__C.SPLIT[__C.RUN_MODE]]
        # self.val_choices_path = __C.EVAL_PATH['val_choices']
        # self.prediction_path = __C.EVAL_PATH['tmp'] + 'result_run_' + __C.VERSION + '.json'

        # # Load scene graphs
        # print("Loading scene graphs...")
        # scenes = self.loadFile(args.scenes.format(tier=args.tier))

        # Load questions
        print("Loading questions...")
        questions = self.loadFile(ques_file_path)

        # Load choices
        choices = None
        if choices_path is not None:
            print("Loading choices...")
            choices = self.loadFile(choices_path)

        # Load predictions and turn them into a dictionary
        print("Loading predictions...")
        self.predictions = self.loadFile(result_eval_file)
        self.predictions = {p["questionId"]: p["prediction"] for p in self.predictions}

        # Make sure all question have predictions
        for qid in questions:
            if (qid not in self.predictions) and (EVAL_CONSISTENCY or questions[qid]["isBalanced"]):
                print("no prediction for question {}. Please add prediction for all questions.".format(qid))
                raise Exception("missing predictions")

        self.scores = {
            "accuracy": [], # list of accuracies per question (1 if correct else 0). Will be averaged ultimately.
            "binary": [], # list of accuracies per a binary question (1 if correct else 0). Will be averaged ultimately.
            "open": [], # list of accuracies per an open question (1 if correct else 0). Will be averaged ultimately.
            "validity": [], # list of validity per question (1 if valid else 0).
            "plausibility": [], # list of plausibility per question (1 if plausible else 0).
            "consistency": [], # list of consistency scores for entailed questions.
            "accuracyPerStructuralType": defaultdict(list), # list of question accuracies for each structural type (e.g. compare, logic questions).
            "accuracyPerSemanticType": defaultdict(list), # list of question accuracies for each semantic type (e.g. questions about an object, an attribute, a relation).
            "accuracyPerLength": defaultdict(list), # list of question accuracies per question's word number.
            "accuracyPerSteps": defaultdict(list), # list of question accuracies per question's reasoning length (steps number).
            "grounding": [] # list of grounding scores for each question.
        }

        # Initialize golden and predicted histograms per each question group. Used to compute the distribution metric.
        self.dist = {
            "gold": defaultdict(lambda: defaultdict(int)),
            "predicted": defaultdict(lambda: defaultdict(int))
        }

        ##### Main score computation
        ##########################################################################################

        # Loop over the questions and compute mterics
        for qid, question in tqdm(questions.items()):
            gold = question["answer"]
            predicted = self.predictions[qid]

            self.correct = (predicted == gold)
            score = self.toScore(self.correct)

            wordsNum = self.getWordsNum(question)
            stepsNum = self.getStepsNum(question)

            # Compute scores over the balanced dataset (more robust against cheating by making educated guesses)
            if question["isBalanced"]:
                # Update accuracy
                self.scores["accuracy"].append(score)
                self.scores["accuracyPerLength"][wordsNum].append(score)
                self.scores["accuracyPerSteps"][stepsNum].append(score)
                self.scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
                self.scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
                answerType = "open" if question["types"]["structural"] == "query" else "binary"
                self.scores[answerType].append(score)

                if choices_path is not None:
                    # Update validity score
                    valid = self.belongs(predicted, choices[qid]["valid"], question)
                    self.scores["validity"].append(self.toScore(valid))

                    # Update plausibility score
                    plausible = self.belongs(predicted, choices[qid]["plausible"], question)
                    self.scores["plausibility"].append(self.toScore(plausible))

                # Update histograms for gold and predicted answers
                globalGroup = question["groups"]["global"]
                if globalGroup is not None:
                    self.dist["gold"][globalGroup][gold] += 1
                    self.dist["predicted"][globalGroup][predicted] += 1

                if EVAL_CONSISTENCY:
                    # Compute consistency (for entailed questions)
                    self.updateConsistency(qid, question, questions)

        # Compute distribution score
        self.scores["distribution"] = self.chiSquare(self.dist["gold"], self.dist["predicted"]) / 100

        # Average scores over all questions (in the balanced dataset) and print scores

        metrics = [
            "binary",
            "open",
            "accuracy",
            "consistency",
            "validity",
            "plausibility",
            "grounding",
            "distribution"
        ]

        detailedMetrics = [
            ("accuracyPerStructuralType", "Accuracy / structural type"),
            ("accuracyPerSemanticType", "Accuracy / semantic type"),
            ("accuracyPerSteps", "Accuracy / steps number"),
            ("accuracyPerLength", "Accuracy / words number")
        ]

        subMetrics = {
            "attr": "attribute",
            "cat": "category",
            "global": "scene",
            "obj": "object",
            "rel": "relation"
        }
        # average
        for k in metrics:
            if isinstance(self.scores[k], list):
                self.scores[k] = self.avg(self.scores[k]) * 100

        for k, _ in detailedMetrics:
            for t in self.scores[k]:
                self.scores[k][t] = self.avg(self.scores[k][t]) * 100, len(self.scores[k][t])

        self.result_string = []
        self.detail_result_string = []

        # print
        # print("")
        for m in metrics:
            # skip grounding and consistency scores if not requested
            if m == "grounding":
                continue
            if m == "consistency" and not EVAL_CONSISTENCY:
                continue
            if m == "validity" and choices_path is None:
                continue
            if m == "plausibility" and choices_path is None:
                continue

            self.result_string.append("{title}: {score:.2f}{suffix}".format(title=m.capitalize(), score=self.scores[m],
                                                        suffix=" (lower is better)" if m == "distribution" else "%"))
            # print score
            # print("{title}: {score:.2f}{suffix}".format(title=m.capitalize(), score=self.scores[m],
            #                                             suffix=" (lower is better)" if m == "distribution" else "%"))

        for m, mPrintName in detailedMetrics:
            # print("")
            # self.detail_result_string.append('\n')

            # print metric title
            # print("{}:".format(mPrintName))
            self.detail_result_string.append("{}:".format(mPrintName))

            for t in sorted(list(self.scores[m].keys())):
                # set sub-metric title
                tName = t
                if isinstance(self.scores[k], list):
                    tName = subMetrics.get(t, t).capitalize()

                self.detail_result_string.append("  {title}: {score:.2f}{suffix} ({amount} questions)".format(title=tName,
                                                                                   score=self.scores[m][t][0], suffix="%",
                                                                                   amount=self.scores[m][t][1]))
                # # print score
                # print("  {title}: {score:.2f}{suffix} ({amount} questions)".format(title=tName,
                #                                                                    score=self.scores[m][t][0], suffix="%",
                #                                                                    amount=self.scores[m][t][1]))


    def get_str_result(self):
        return self.result_string, self.detail_result_string

    def loadFile(self, name):
        # load standard json file
        if os.path.isfile(name):
            with open(name) as file:
                data = json.load(file)
        # load file chunks if too big
        elif os.path.isdir(name.split(".")[0]):
            data = {}
            chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
            for chunk in chunks:
                with open(chunk) as file:
                    data.update(json.load(file))
        else:
            raise Exception("Can't find {}".format(name))
        return data

    # book to float
    def toScore(self, b):
        return float(1 if b else 0)

    # Compute average of a list
    def avg(self, l):
        if len(l) == 0:
            return 0
        return float(sum(l)) / len(l)

    def wavg(self, l, w):
        if sum(w) == 0:
            return None
        return float(sum(l[i] * w[i] for i in range(len(l)))) / sum(w)

    ##### Question lengths - words numbers and reasoning steps number
    ##########################################################################################

    # Compute question length (words number)
    def getWordsNum(self, question):
        return len(question["question"].split())

    # Compute number of reasoning steps (excluding the final "querying" step which doesn't increase effective reasoning length)
    def getStepsNum(self, question):
        return len([c for c in question["semantic"] if not (any([o in "{}: {}".format(c["operation"], c["argument"])
                                                                 for o in ["exist", "query: name", "choose name"]]))])

    # ##### Functions for question annotations
    # ##########################################################################################
    #
    # # Utility function for converting question annotations string keys to slices
    # def toSlice(self, strSlice):
    #     sliceLims = (int(n) for n in strSlice.split(':'))
    #     return apply(slice, sliceLims)
    #
    # # Utility function for converting question annotations string keys to indexes list:
    # # "1" => [0]
    # # "1:3" => [1, 2]
    # # "4:9:2" => [4, 6, 8]
    # def intsFromSlice(self, strSlice):
    #     slice_obj = get_slice_obj(slicearg)
    #     return (range(slice_obj.start or 0, slice_obj.stop or -1, slice_obj.step or 1))

    ##### Functions for validity and plausibility
    ##########################################################################################

    def belongs(self, element, group, question):
        # normalization ()
        if "Common" in question["types"]["detailed"]:
            group = ["color", "material", "shape"]

        return element in group

    ##### Functions for consistency scores (for entailed questions ("inferred"))
    ##########################################################################################

    def updateConsistency(self, questionId, question, questions):
        inferredQuestions = [eid for eid in question["entailed"] if eid != questionId]

        if self.correct and len(inferredQuestions) > 0:

            cosnsitencyScores = []
            for eid in inferredQuestions:
                gold = questions[eid]["answer"]
                predicted = self.predictions[eid]
                score = self.toScore(predicted == gold)
                cosnsitencyScores.append(score)

            self.scores["consistency"].append(self.avg(cosnsitencyScores))

    ##### Functions for distribution score
    ##########################################################################################

    # Compute chi square statistic of gold distribution vs predicted distribution,
    # averaged over all question groups
    def chiSquare(self, goldDist, predictedDist):
        sumScore, sumOverall = 0, 0

        for group in goldDist:
            score, overall = 0, 0

            for ans in goldDist[group]:
                e = goldDist[group][ans]
                o = predictedDist[group].get(ans, 0)
                score += ((float(o - e) ** 2) / e)
                overall += goldDist[group][ans]

            sumScore += score * overall
            sumOverall += overall

        avgScore = float(sumScore) / sumOverall

        return avgScore

