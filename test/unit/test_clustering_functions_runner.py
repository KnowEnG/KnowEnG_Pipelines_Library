# -*- coding: utf-8 -*-
"""
Demonstrate two ways to create and run a unittest.TestSuite

Created on Tue May 31 13:50:35 2016
Python Script for Unit Test of KnowEnG.py modules
KnowEnG_UnitTestsRunner.py

commandLine>> run KnowEnG_UnitTestsRunner.py

lanier4@illinois.edu
"""
#sample_clustering_unit_test_directory = '/Users/del/Sample_Clustering_Pipeline/test/uint'
import unittest
#import sys
#sys.path.extend(sample_clustering_unit_test_directory)
import test_clustering_functions as tkeg

#                               method A
#test_suite = unittest.TestSuite()
#test_suite.addTest(unittest.makeSuite(tkeg.test_clustering_functions))
#myrunner = unittest.TextTestRunner()
#myResult = myrunner.run(test_suite)

#                               method B
mySuit = tkeg.suite()
runner = unittest.TextTestRunner()
myResult2 = runner.run(mySuit)