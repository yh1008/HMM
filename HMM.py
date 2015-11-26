#author: Emily Hua     09/26/2015   

from __future__ import division
import os,sys
from io import open
import sys 
from string import punctuation
import re
import timeit
reload(sys)  
sys.setdefaultencoding('utf8')

start = timeit.default_timer()

inputFile = open("training.pos", "r")
inputFile1 = open("development.text", "r") #the test data file
outputFile = open("prior.txt", "w") #prior table content
outputFile1 = open("likelihood.txt", "w") #likelihood table content 
outputFile2 = open("sys.txt","w") #the HMM tagging output

tagList = [] #store distinct tags/states
tagFullList = [] #store all of the tags(repeatable) appear in the training file
tagStartList = [] #store tags that are the begining of the sentence
tagEndList = [] #store tags that are the end of the sentence
corpus = [] #store all the words show up in the training corpus

marker = 0 # a marker for the end of sentence
counterS = 0 #for counting total number of tag as the start of the sentence
counterF = 0 # for counting total number of tags ever appeared
countTag = 0 #[prior]count total occurance for each distinct tag
countTag1 = 0 #[likelihood]count total occurance for each distinct tag
countJoint = 0 #count the occurance for A to be followed by B
countStart = 0 #count how many times this tag is the start of the sentence
countEnd = 0 #count how many time this tag is the end of the sentence

#get a list of tag, know there is only 43 distinguishable tag

tagStartList = ["NNP"] #the firt tag in the corpus is NNP 

wordtag = {} #store likelihood table 
print ("computing...")
for line in inputFile:
	s = re.match(r'^\s*$', line)  #find empty line
	if s:
		marker = 1
	else: 
		sentenceList = line.split()
		word = sentenceList[0]
		tag = sentenceList[1]
		if marker == 1:
			tagStartList.append(tag)
			counterS = counterS + 1 
			tagEndList.append(tagFullList[-1])
			marker = 0
		#store full tag list that ever occurs in the training corpus
		tagFullList.append(tag)
			
		#eg.{'NNP': {'Goodwill' : 1, 'house' : 2 , ....}
		corpus.append(word)

		wordtag[tag] = wordtag.get(tag, {})

		wordtag[tag][word] = wordtag[tag].get(word, 0) + 1 			
		#store unrepeted tag in a new list 
		if tag not in tagList:	
			tagList.append(tag)


##################################################################Prior probability table
# calculate the transition probability of <S>
dic = {}
for i in tagList:
	for j in tagStartList:
			if i == j:
				countStart = countStart + 1
	ProbS = countStart/counterS
	dic[i] = ProbS
	countStart = 0
dicP = {"<S>": dic} #store prior table 
#calculate the End prior
dicE = {} #temporarry dic 
#calculate the prior (End|state) = C(state, End)/C(state) 
for i in tagList:
	for j  in range(len(tagEndList)):
		for f in tagFullList:
			if j == 0:
				if i == f:
					countTag = countTag + 1 
		if i == tagEndList[j]:
			countEnd = countEnd + 1 
	#print ("tag: " + i +" is in end of sentence for $ of times: " + str(countEnd))
	#print ("tag: " + i +" is in the corpus: " + str(countTag))
	ProbE = format(countEnd/(countTag*1.0), '.5f')
	dicE.update({i: {"END":ProbE}})
	countEnd = 0
	countTag = 0


#prior = C(t_i-1, ti)/C(t_i-1)
countTag = 0
dicT = {} #temporary dic
for e in range (len(tagList)):
	for f in range(len(tagList)):
		for n in range(len(tagFullList)):
			if f == 0:
				if tagList[e] == tagFullList[n]:
					countTag = countTag + 1 
			if n != (len(tagFullList) -1 ):
				if tagList[e] == tagFullList[n]:
					if tagList[f] == tagFullList[n + 1]:
						countJoint = countJoint + 1
		prior = format(countJoint/float(countTag), '.5f')
		dicT[tagList[f]] = prior 
		countJoint = 0 
	countTag = 0
	dicT["END"] = dicE[tagList[e]]["END"]
	dicP.update({tagList[e]: dicT})
	countJoint = 0
	dicT = {}

#display the prior dicP's content
for i in dicP:
	outputFile.write ("*******************************"+str(i)+'\n')
	for k in dicP[i]:
		outputFile.write (str(k) + " ")
		outputFile.write ( str(dicP[i][k]) + "\n")


outputFile.close()

##################################################################likelihood table
# eg. {'NNP': {'Goodwill' : 0.999}} where 0.999 is P(Goodwill|NNP) 
# P (Goodwill|NNP) = C(NNP, Goodwill)|C(NNP)	

for i in tagList:
	for j in tagFullList:
		if i == j:
			countTag1 = countTag1 + 1
	for k in wordtag[i]: 
		
			p = format(wordtag[i].get(k, 0) / countTag1, '.5f')
			wordtag[i][k] = p
			
	countTag1 = 0 #renew countTag for a new tag	
			
#display the likelihood wordtag's content
for i in wordtag:
	outputFile1.write ("*******************************"+str(i)+'\n')
	for k in wordtag[i]:
		outputFile1.write (str(k) + " ")
		outputFile1.write ( str(wordtag[i][k]) + '\n')

outputFile1.close()
inputFile.close()
##################################################################HMM 
def HMM(wordList):

	w1 = wordList[0] #the first word of the sentence
	qRange = len(tagList)
	wRange = len(wordList)

	viterbi = [[0 for x in range(200)] for x in range(200)] 
	backpointer = [['' for x in range(200)] for x in range(200)] 
	#intialization
	for q in range (qRange):
		#score transition 0 -> q given w1
		if w1 in corpus:
				if w1 in wordtag[tagList[q]]:
					viterbi[q][1] = float(dicP['<S>'][tagList[q]]) * float(wordtag[tagList[q]][w1])
				else: #not in this particular POS tag
					viterbi[q][1] = float(dicP['<S>'][tagList[q]]) * 0 
		else:
			if tagList[q] in punctuation:
				viterbi[q][1] = float(dicP['<S>'][tagList[q]]) * 0  
			else: 
				min_val = min(wordtag[tagList[q]].itervalues())
				viterbi[q][1] = float(dicP['<S>'][tagList[q]]) * float(min_val)
		

		backpointer[q][1] = 0 #stand for q0 (start point)
	#for word w from 2 to T
	maxViterbi = 0
	maxPreviousState = 0 
	maxPreTerminalProb = 0
	for w in range (1, wRange):	
	  for q in range (qRange):
			if wordList[w] in corpus:
				if wordList[w] in wordtag[tagList[q]]:
					wordLikelihood = wordtag[tagList[q]][wordList[w]]
				else:
					wordLikelihood = 0
			else:
				if tagList[q] in punctuation:
					wordLikelihood = 0
					
				else :
					wordLikelihood = min(wordtag[tagList[q]].itervalues())
				#print ("wordLikelihood: " + str(wordLikelihood)) #16 checked
		
				
			#find max verterbi = max (previous * prior * likelihood)		
			maxViterbi = float(viterbi[0][w]) * float(dicP[tagList[0]][tagList[q]]) * float(wordLikelihood)
			maxPreviousState = 0
			for i in range (1, qRange):
				
				if float(viterbi[i][w]) * float(dicP[tagList[i]][tagList[q]]) * float(wordLikelihood) > maxViterbi:
					 maxViterbi = float(viterbi[i][w]) * float(dicP[tagList[i]][tagList[q]])* float(wordLikelihood)
					 maxPreviousState = i #content tagList[i]		

			viterbi[q][w+1] = maxViterbi	
			backpointer[q][w+1] = tagList[maxPreviousState] #points to the matrix x axis (max previous)
			
			maxViterbi = 0
			maxPreviousState = 0 
			maxPreTerminalProb = 0
	#termination step
	#viterbi[qF, T] = max (viterbi[s,T] *as,qF)
	maxPreTerminalProb = float(viterbi[0][wRange] )* float(dicP[tagList[0]]["END"])
	
	maxPreviousState = 0
	for i in range (1, qRange):
		
		if float(viterbi[i][wRange]) * float(dicP[tagList[i]]["END"]) > maxPreTerminalProb:
			maxPreTerminalProb = float(viterbi[i][wRange]) * float(dicP[tagList[i]]["END"])
			maxPreviousState = i
		
	
	viterbi[qRange][wRange+1] = maxPreTerminalProb 
	#store the state that returns the maxPreTerminalProbability
	backpointer[qRange][wRange+1] = tagList[maxPreviousState]

	#return POS tag path 
	pathReverse = [tagList[maxPreviousState]]
	maxPreviousTag = tagList[maxPreviousState]
	
	i = 0
	while i < (wRange -1):
				
		pathReverse.append(backpointer[tagList.index(maxPreviousTag)][wRange - i])
		maxPreviousTag = backpointer[tagList.index(maxPreviousTag)][wRange - i]
		i = i + 1 

	#reverse the path to make it correct
	index = len(pathReverse)
	path = []
	while index >= 1 :
		path.append(pathReverse[index - 1])
		index = index -1 
	return path
			 
#main()
wordList = [] #store words in a sentence
for line in inputFile1:
	
	if line.strip() != '': #if not empty do following 
		sentenceList = line.split()
		word = sentenceList[0]
		wordList.append(word)		
	s = re.match(r'^\s*$', line)  #find empty line

	if s:
		path = HMM(wordList) #list of TAGs returned by HMM function call
		for i in range(len(wordList)):
			outputFile2.write(wordList[i]+"	"+path[i] + "\n")
		outputFile2.write("\n")
		wordList = [] # refresh word list
	
#cleaning up	
outputFile2.close()
inputFile1.close()

stop = timeit.default_timer()
timetakes = format(stop - start, '.3f') 
print("\nit took " + str(timetakes) + " seconds to complete")
print ("lol, finished!")
