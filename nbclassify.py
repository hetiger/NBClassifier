import json, ast, sys, math
from nblearn import tokenize, loadStopWords

stopWords = set("")

def readTestData(fileName):
    testData = {}
    f = open(fileName)
    for line in iter(f):
        line = line.rstrip()
        if line:
            lineMap = tokenize(line)
            for reviewID in lineMap:
                testData[reviewID] = lineMap[reviewID]
    f.close()
    return testData

def getClassScore(featureClass, content, model, classProbability):
    score = math.log(classProbability)
    for word in content:
        if word in model:
            wordClassMap = model[word]
            score += math.log(wordClassMap[featureClass])
    return score

def getMostLikelyClass(classScores):
    argMaxScore = float('-inf')
    for featureClass, classScore in classScores.iteritems():
        if classScore > argMaxScore:
            argMaxScore = classScore
            argMaxClass = featureClass
    return argMaxClass

def applyNaiveBayes(classProbabilities, wordProbabilities, testData):
    output = {}
    for reviewID, reviewContent in testData.iteritems():
        classScore = {}
        for featureClass, classProbability in classProbabilities.iteritems():
            classScore[featureClass] = getClassScore(featureClass, reviewContent, wordProbabilities, classProbability)
        argMaxClass = getMostLikelyClass(classScore)
        output[reviewID] = argMaxClass
    return output

def mergeOutput(output1, output2):
    output = []
    for reviewID, reviewClass1 in output1.iteritems():
        reviewOutput = []
        reviewOutput.append(reviewID)
        reviewOutput.append(output1[reviewID])
        reviewOutput.append(output2[reviewID])
        output.append(reviewOutput)
    return output

def writeToFile(output, filePath):
    f = open(filePath, 'w')
    countOutputLines = len(output)
    for reviewOutput in output:
        strReviewOutput = ""
        for word in reviewOutput:
            strReviewOutput += word
            strReviewOutput += " "
        countOutputLines-=1
        if countOutputLines > 0:
            f.write(strReviewOutput.rstrip() + "\n")
        else:
            f.write(strReviewOutput.rstrip())
    f.close()

if __name__ == '__main__':
    # loading stop words
    loadStopWords("stop-words")

    # loading model from nbmodel.txt
    with open("nbmodel.txt") as modelFile:
        model = json.load(modelFile)
    model = ast.literal_eval(json.dumps(model))

    # loading model 1
    model1ClassProbabilities = model["Model1ClassProbabilities"]
    model1WordProbabilities = model["Model1WordProbabilities"]

    # loading model 2
    model2ClassProbabilities = model["Model2ClassProbabilities"]
    model2WordProbabilities = model["Model2WordProbabilities"]

    testTextFile = sys.argv[1]

    # reading test data
    testData = readTestData(testTextFile)

    # applying Naive Bayes Classifier 1
    output1 = applyNaiveBayes(model1ClassProbabilities, model1WordProbabilities, testData)

    # applying Naive Bayes Classifier 2
    output2 = applyNaiveBayes(model2ClassProbabilities, model2WordProbabilities, testData)

    # merging both the outputs
    output = mergeOutput(output1, output2)

    # writing the output to file
    filePath = "nboutput.txt"
    writeToFile(output, filePath)
