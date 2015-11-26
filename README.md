#### HMM
###Hidden Markov model
####Author: Emily Hua
####this is an implementation of using HMM to generate POS tags
####algorithms see http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/hmms.pdf
---------------------
####The program will produce: 
####1. a txt file called "prior.txt" that contains the prior table content; 
####2. a txt file called "likelihood.txt" that contains the likelihood table content.
####3. a txt file called "sys.txt" that contains the POS tagging result (the hardcoded input file is the development.text)
####4. I included a sample output file generated by my system, renamed to sampleOutput.txt
---------------------

#####How to run the program: 
#####//1. notice the training.pos and development.text has to be in the same directory as this program
#####//2. took 30 mins on department's server/14 mins on a machine with 8G RAM
---------------------
$ python HMM.py 
