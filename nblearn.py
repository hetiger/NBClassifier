import json, sys, string

stopWords = set("")

def readLabels(fileName, classifierIdx):
    labels = {}
    classes = set("")
    classesCount = {}
    f = open(fileName)
    for line in iter(f):
        line = line.rstrip()
        if line:
            lineData = line.split(None, 1)
            lineWords = lineData[1].split()
            reviewID = lineData[0]
            reviewClass = lineWords[classifierIdx]
            labels[reviewID] = reviewClass
            classes = classes.union([reviewClass])
            if reviewClass in classesCount:
                classesCount[reviewClass]+=1
            else:
                classesCount[reviewClass] = 1
    f.close()
    return labels, classes, classesCount

def translateContent(content):

    # removing punctuation
    content = [s.translate(None, string.punctuation) for s in content]

    # removing empty strings
    content = [s for s in content if s]

    # removing numbers
    content = [s for s in content if not (s.isdigit() or s[0] == '-' and s[1:].isdigit())]

    # converting to lower case
    content = [s.lower() for s in content]

    # removing stop words
    translatedContent = []
    for word in content:
        if word not in stopWords:
            translatedContent.append(word)

    return translatedContent

def tokenize(line):
    lineData = line.split(None, 1)
    lineWords = lineData[1].split()
    lineWords = translateContent(lineWords)
    lineMap = {}
    lineMap[lineData[0]] = lineWords
    return lineMap

def getClassesMap(classes):
    classesMap = dict.fromkeys(classes, 0)
    return classesMap

def buildModel(fileName, labels, classes):
    wordCount = {}
    classWordCount = dict.fromkeys(classes, 0)
    f = open(fileName)
    for line in iter(f):
        line = line.rstrip()
        if line:
            review = tokenize(line)
            for reviewID, reviewContent in review.iteritems():
                for word in reviewContent:
                    if word in wordCount:
                        classesMap = wordCount[word]
                    else:
                        classesMap = getClassesMap(classes)

                    # incrementing the count of the class to which this word belongs
                    reviewClass = labels[reviewID]
                    classesMap[reviewClass] += 1
                    wordCount[word] = classesMap
                    classWordCount[reviewClass] += 1
    f.close()
    return wordCount, classWordCount

def isSmoothingRequired(wordCount):
    isRequired = False
    for classesMap in wordCount.values():
        for wordOccurences in classesMap.values():
            if wordOccurences == 0:
                isRequired = True
                break
        else:
            continue
        break
    return isRequired

def addOneSmoothing(wordCount):
    for word, classesMap in wordCount.iteritems():
        for featureClass in classesMap:
           classesMap[featureClass]+=1
           wordCount[word] = classesMap
    return wordCount

def addVocabularyToClassWordCount(wordCount, classWordCount):
    lenVocabulary = len(wordCount)
    for featureClass in classWordCount:
        classWordCount[featureClass]+=lenVocabulary
    return classWordCount

def computeWordProbabilities(wordCount, classWordCount):
    for word, classesMap in wordCount.iteritems():
        for featureClass in classesMap:
           classesMap[featureClass]/=float(classWordCount[featureClass])
           wordCount[word] = classesMap
    return wordCount

def computeClassProbabilities(classesCount):
    totReviews = 0
    #computing total reviews
    for featureClass in classesCount:
        totReviews += classesCount[featureClass]

    # computing class prior probabilities
    for featureClass in classesCount:
        classesCount[featureClass]/=float(totReviews)

    return classesCount

def writeModel(model, filePath):
    with open(filePath, "w") as modelFile:
        json.dump(model, modelFile, sort_keys=True, indent=4, ensure_ascii=False)

def getModel(trainTextFile, trainLabelsFile, classifierIdx):
    labels, classes, classesCount = readLabels(trainLabelsFile, classifierIdx)
    wordCount, classWordCount = buildModel(trainTextFile, labels, classes)

    if isSmoothingRequired(wordCount):
        wordCount = addOneSmoothing(wordCount)
        classWordCount = addVocabularyToClassWordCount(wordCount, classWordCount)

    wordCount = computeWordProbabilities(wordCount, classWordCount)

    # Computing prior probabilities of all the classes
    classProbabilities = computeClassProbabilities(classesCount)

    return classProbabilities, wordCount

def loadStopWords(filePath):
    global stopWords
    f = open(filePath)
    for line in iter(f):
        line = line.rstrip()
        if line:
            words = line.split()
            # removing punctuation
            words = [s.translate(None, string.punctuation) for s in words]
            stopWords = set(words)

if __name__ == '__main__':
    # reading train-text and train-labels file
    trainTextFile = sys.argv[1]
    trainLabelsFile = sys.argv[2]

    # loading stop words
    loadStopWords("stop-words")

    # reading model 1 (for 1st Classifier)
    model1ClassProbabilities, model1WordProbabilities = getModel(trainTextFile, trainLabelsFile, 0)

    # reading model 2 (for 2nd Classifier)
    model2ClassProbabilities, model2WordProbabilities = getModel(trainTextFile, trainLabelsFile, 1)

    # creating data structure for writing the models to file
    model = {}
    model["Model1ClassProbabilities"] = model1ClassProbabilities
    model["Model1WordProbabilities"] = model1WordProbabilities
    model["Model2ClassProbabilities"] = model2ClassProbabilities
    model["Model2WordProbabilities"] = model2WordProbabilities

    # writing the model to file
    writeModel(model, "nbmodel.txt")