from org.petuum.jbosen import PsApplication, PsTableGroup
from org.petuum.jbosen.table import IntTable
from org.apache.commons.math3.special import *
from DataLoader import DataLoader
import random
import time
import os
import sys

TOPIC_TABLE = 0
WORD_TOPIC_TABLE = 1

class LdaApp(PsApplication):
	def __init__(self, dataFile, outputDir, numWords, numTopics,
                alpha, beta, numIterations, numClocksPerIteration, staleness):
		self.outputDir = outputDir
		self.numWords = numWords
		self.numTopics = numTopics
		self.alpha = alpha
		self.beta = beta
		self.numIterations = numIterations
		self.numClocksPerIteration = numClocksPerIteration
		self.staleness = staleness
		self.dataLoader = DataLoader(dataFile)
		
	def logDirichlet_vector(self, alpha):
		sumLogGamma = 0.0
		logSumGamma = 0.0
		for value in alpha:
			sumLogGamma += Gamma.logGamma(value)
			logSumGamma += value
		
		return sumLogGamma - Gamma.logGamma(logSumGamma)

	def logDirichlet_const(self, alpha, k):
		return k * Gamma.logGamma(alpha) - Gamma.logGamma(k*alpha)
	
	def getColumn(self, matrix, columnId):
		# TO DO: 
		# Get the column of wordTopicTable according to columnId
		#
  		# ... fill me out ...x
  		#
  		
  		# return [matrix.get(row, columnId) for row in range(self.numWords)]
  		col = []
  		for row in range(self.numWords):
  			# print row
  			col.append(matrix.get(row, columnId))
  		return col
		pass
		
	def getRow(self, matrix, rowId):
		# TO DO: 
		# Get the row of docTopicTable according to rowId
		#
  		# ... fill me out ...
  		#
  		# i = 0
  		# print "rowID = %d" %rowId
  		# for row in matrix:
  		# for i, row in enumerate(matrix):
  			# if i == rowId:
  				# print row
  				# print type(row)
  				# return row
  				# print i
  				# return matrix[i][:]
  			# i += 1
  		return matrix[rowId][:]
		pass

	def getLogLikelihood(self, wordTopicTable, docTopicTable):
		lik = 0.0
		for k in range(self.numTopics):
			temp = self.getColumn(wordTopicTable, k)
			for w in range(self.numWords):
				 temp[w] += self.beta
		  
			lik += self.logDirichlet_vector(temp)
			lik -= self.logDirichlet_const(self.beta, self.numWords)
	  
		for d in range(len(docTopicTable)):
			temp = self.getRow(docTopicTable, d)
			for k in range(self.numTopics):
				temp[k] += self.alpha

			lik += self.logDirichlet_vector(temp)
			lik -= self.logDirichlet_const(self.alpha, self.numTopics)
		return lik
  
  	# TO DO: 
  	# Sample function
  	#
  	# ... fill me out ...
  	#
  	def sample(self, p, norm):
		sum_p_up_to_k = 0.0
		r = random.random()
		for k in xrange(self.numTopics):
			sum_p_up_to_k += p[k]/norm
			if r < sum_p_up_to_k:
				return k
		
	def initialize(self):
		# Create global topic count table. self table only has one row, which
		# contains counts for all topics.
		PsTableGroup.createDenseIntTable(TOPIC_TABLE, self.staleness, self.numTopics)
		# Create global word-topic table. self table contains numWords rows, each
		# of which has numTopics columns.
		PsTableGroup.createDenseIntTable(WORD_TOPIC_TABLE, self.staleness, self.numTopics)
  
	def runWorkerThread(self, threadId):
		clientId = PsTableGroup.getClientId()

		# Load data for this thread
		print("Client %d thread %d loading data..." % (clientId, threadId))
		part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId
		numParts = PsTableGroup.getNumTotalWorkerThreads()
		w = self.dataLoader.load(part, numParts)
		# print "w rows = %d"%len(w)
		# print "w cols = %d"%len(w[0])

		# Get global tables
		topicTable = PsTableGroup.getIntTable(TOPIC_TABLE)
		wordTopicTable = PsTableGroup.getIntTable(WORD_TOPIC_TABLE)

		# Initialize LDA variables
		print("Client %d thread %d initializing variables..." % (clientId, threadId))
		docTopicTable = [[0] * self.numTopics for _ in range(len(w))]

		# TO DO: 
		# Initialize Sampling
		#
  		# ... fill me out ...
  		#
  		# print "Total Topics = %d"%self.numTopics
  		# print "Total Words = %d"%self.numWords
  		z = [[-1 for j in xrange(len(w[d]))] for d in xrange(len(w))]
  		for d in xrange(len(w)):
  			# print d
  			for i in xrange(len(w[d])):
  				# print i
  				word = w[d][i]
  				# print word
  				topic = random.randint(0, self.numTopics - 1)
  				z[d][i] = topic
  				docTopicTable[d][topic] += 1
  				wordTopicTable.inc(word, topic, 1)
  				topicTable.inc(0, topic, 1)
  		PsTableGroup.globalBarrier()

		# Do LDA Gibbs sampling
		print("Client %d thread %d starting gibbs sampling..." % (clientId, threadId))
		llh = [0.0] * self.numIterations
		sec = [0.0] * self.numIterations
		totalSec = 0.0
		for	iterId in range(self.numIterations):
			startTime = time.time()
			# Each iteration consists of a number of batches, and we clock
			# between each to communicate parameters according to SSP
			for batch in range(self.numClocksPerIteration):
				begin = len(w) * batch / self.numClocksPerIteration
				end = len(w) * (batch + 1) / self.numClocksPerIteration
				# TO DO:
				# Loop through each document in the current batch
				#
  				# ... fill me out ...
  				#
  				# print "begin=%d" %begin
  				# print "end=%d" %end
  				# print range(begin, end)
  				for d in xrange(begin, end):
  					# print d-
  					# print range(begin, end)
  					# for i in range(self.numTopics):
  					for i in xrange(len(w[d])):
  						word = w[d][i]
  						topic = z[d][i]
  						docTopicTable[d][topic] += -1
  						wordTopicTable.inc(word, topic, -1)
  						topicTable.inc(0, topic, -1)
  						p =[]
						norm = 0.0
						# print "topic=%d" %d
						# print len(docTopicTable[0])
						for k in xrange(self.numTopics):
							z_di_equals_k = 1 if topic == k else 0
							# print docTopicTable[d][k]
							# print "alpha = %d"%self.alpha
							ak = (docTopicTable[d][k] - z_di_equals_k + self.alpha)
							# ak = (docTopicTable[d][k] + self.alpha)
							# print wordTopicTable.get(word, k)
							bk = ((wordTopicTable.get(word, k) - z_di_equals_k + self.beta)/(topicTable.get(0, k) - z_di_equals_k + self.numWords * self.beta))
							# bk = ((wordTopicTable.get(word, k) + self.beta)/(topicTable.get(0, k) + self.numWords * self.beta))
							pk = ak*bk
							# print "bk = %.3f" %bk
							# print "ak = %.3f" %ak
							# print "pk = %.3f" %pk
							p.append(pk)
							norm += pk
  						topic = self.sample(p, norm)
  						z[d][i] = topic #New Line added as per Piazza forum
  						docTopicTable[d][topic] += 1
  						wordTopicTable.inc(word, topic, 1)
  						topicTable.inc(0, topic, 1)
  				PsTableGroup.clock()


			# Calculate likelihood and elapsed time
			totalSec += (time.time() - startTime)
			sec[iterId] = totalSec
			llh[iterId] = self.getLogLikelihood(wordTopicTable, docTopicTable)
			print("Client %d thread %d completed iteration %d" % (clientId, threadId, iterId+1))
			print("    Elapsed seconds: %f" % (sec[iterId]))
			print("    Log-likelihood: %.15e" % (llh[iterId]))

		PsTableGroup.globalBarrier()

		# Output likelihood
		print("Client %d thread %d writing likelihood to file..." % (clientId, threadId))

		try:
			with open(os.path.join(self.outputDir, "likelihood_%d-%d.csv" % (clientId, threadId)), 'w') as writer:
				for i in range(self.numIterations):
					writer.write("%d,%f,%.15e\n" % (i+1, sec[i], llh[i]))
		except Exception as detail:
			print(detail)
			sys.exit(1)

		PsTableGroup.globalBarrier()

		# Output tables
		if clientId == 0 and threadId == 0:
			print("Client %d thread %d writing word-topic table to file..." % (clientId, threadId))

			try:
				with open(os.path.join(self.outputDir, "word-topic.csv"), 'w') as writer:
					for i in range(self.numWords):
						counter = map(lambda k : str(wordTopicTable.get(i, k)), range(self.numTopics))
						writer.write(','.join(counter) + '\n')
			except Exception as detail:
				print(detail)
				sys.exit(1)

		PsTableGroup.globalBarrier()

		print("Client %d thread %d exited." % (clientId, threadId))






	
#if __name__ == "__main__":
#    lda = LdaApp()
#	config = PsConfig()
#	lda.run(config)
	